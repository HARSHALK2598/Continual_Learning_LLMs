import torch
import os
import json
from copy import copy, deepcopy
import numpy as np

from transformers import get_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW

import evaluate
from utils import *
from custom_data_collator import CustomCollatorwithLabelPadding
import datetime

import logging
from options import Options
import pickle
from accelerate import Accelerator

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from dataloaders import *


def orthogonal_lora_loss(model, past_task_weights_A, past_task_weights_B, device, normalization_type="frobenius"):
    """
    Compute the orthogonal loss using AB matrices from past tasks and the current model.

    Args:
        model: The current model with LoRA parameters.
        past_task_weights_A (dict): Dictionary of past task LoRA_A parameters.
        past_task_weights_B (dict): Dictionary of past task LoRA_B parameters.
        device: The device to move tensors to.
        normalization_type (str): The type of normalization to use. 
                                  Options: "column" (normalize column vectors) or "frobenius" (normalize entire matrix).

    Returns:
        torch.Tensor: The computed orthogonal loss.
    """
    orth_loss = torch.tensor(0.0, device=device)

    # Store current model's LoRA A and B matrices
    model_lora_A = {name: param for name, param in model.named_parameters() if "lora_A" in name}
    # Corrected way to get LoRA B matrices
    model_lora_B = {
        name.replace("lora_A", "lora_B"): model.state_dict()[name.replace("lora_A", "lora_B")]
        for name in model_lora_A.keys()
        if name.replace("lora_A", "lora_B") in model.state_dict()
    }
    for layer_name in model_lora_A.keys():
        if layer_name in model_lora_B:
            # Retrieve current model's A and B matrices
            current_A = model_lora_A[layer_name]
            current_B = model_lora_B[layer_name]
            


            # Compute AB for the current model
            current_AB = torch.mm(current_A.T, current_B.T)
            # Check if past task weights exist for this layer
            for task_name in past_task_weights_A.keys():
                if layer_name in past_task_weights_A[task_name] and layer_name in past_task_weights_B[task_name]:
                    # Retrieve past task's A and B matrices
                    saved_A = past_task_weights_A[task_name][layer_name].to(device)
                    saved_B = past_task_weights_B[task_name][layer_name].to(device)

                    # Compute AB for the past task
                    saved_AB = torch.mm(saved_A.T, saved_B.T)
                    
                    # Normalize AB matrices based on the chosen method
                    if normalization_type == "column":
                        saved_AB_norm = saved_AB / torch.norm(saved_AB, dim=0, p=2, keepdim=True)
                        current_AB_norm = current_AB / torch.norm(current_AB, dim=0, p=2, keepdim=True)
                    elif normalization_type == "frobenius":
                        saved_AB_norm = saved_AB / torch.norm(saved_AB, p="fro")
                        current_AB_norm = current_AB / torch.norm(current_AB, p="fro")

                    # Compute dot product and accumulate loss
                    term = torch.abs(torch.dot(saved_AB_norm.view(-1), current_AB_norm.view(-1))).sum()
                    orth_loss += term
        break

    return orth_loss


def train(model, dataloader, optimizer, scheduler, lambda1, past_task_weights_A, past_task_weights_B, accelerator, normalized):
    model.train()
    train_loss, train_orth_loss = 0.0, 0.0

    for batch in dataloader:
        optimizer.zero_grad()


        # Check if 'generation_text' exists before popping
        if 'generation_text' in batch and 'generation_text_attn_mask' in batch:
            _, _ = batch.pop('generation_text'), batch.pop('generation_text_attn_mask')
        else:
            print("Error: 'generation_text' is missing from the batch")
            print(f"Available keys: {batch.keys()}")
            raise KeyError("'generation_text' or 'generation_text_attn_mask' missing from batch")

        batch = {k:v.to(accelerator.device) for k,v in batch.items()}
        labels = batch.pop('labels')
        out = model(**batch)
        logits = out.logits

        orth_loss = orthogonal_lora_loss(model, past_task_weights_A, past_task_weights_B, accelerator.device, normalized)
        ce_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        loss = ce_loss + lambda1 * orth_loss

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        train_loss += ce_loss.item()
        train_orth_loss += orth_loss.item()

    return train_loss / len(dataloader), train_orth_loss / len(dataloader)


def validate(model, dataloader, rouge_score, device):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    for batch in dataloader:
        with torch.no_grad():
            batch = {k:v.to(device) for k,v in batch.items()}
            generation_text, generation_attention = batch.pop('generation_text'), batch.pop('generation_text_attn_mask')
            labels = batch.pop('labels')
            out = model(**batch) 
            logits = out.logits
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
            val_loss = val_loss + loss.item()

            labels_np = labels.cpu().numpy()
            labels_np = np.where(labels_np != -100, labels_np, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels_np, skip_special_tokens=True)
    
            generated_tokens = model.generate( generation_text,  attention_mask=generation_attention, max_new_tokens = 32)
            decoded_generations = tokenizer.batch_decode( generated_tokens, skip_special_tokens=False)
            decoded_generations = [gen.split('### Response:\n')[1] for gen in decoded_generations]
            decoded_generations = [gen.split(tokenizer.eos_token)[0] for gen in decoded_generations]
            rouge_score.add_batch(predictions=decoded_generations, references=decoded_labels)
            
            perfect_matches = [1 if lab == gen else 0 for (lab, gen) in zip(decoded_labels, decoded_generations)]
            # print(perfect_matches)
            val_accuracy = val_accuracy + sum(perfect_matches)
            # break

    val_loss = val_loss / len(dataloader)
    val_accuracy = val_accuracy / len(dataloader.dataset)
    result = rouge_score.compute()
    result['accuracy'] = val_accuracy
    result = {key: 100*value for key, value in result.items()}


    return val_loss, result



if __name__ == '__main__':

    accelerator = Accelerator()

    args = Options(accelerator=accelerator).parse()
    if accelerator.is_main_process:
        logging.basicConfig(filename=os.path.join(args.experiment_dir, "logs.log"),
                                                format='%(asctime)s %(message)s',
                                                filemode='w')
        logger = logging.getLogger("Experiment Logger")
        logger.setLevel(logging.INFO)

    # Model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token	

    sequential_train_data, sequential_valid_data = {}, {}
    print("Preparing the datasets for continual O-LoRA training."%args.tasks)
    for task in args.tasks:
        if accelerator.is_main_process:
            logger.info("Preparing the %s dataset for joint training."%task)
        trainset, testset = GET_DATASET[task]()
        sequential_train_data[task] = preprocess_dataset_list(trainset, tokenizer)
        sequential_valid_data[task] = preprocess_dataset_list(testset, tokenizer)

    # Preparing dataloaders for training and evaluation
    data_collator = CustomCollatorwithLabelPadding(tokenizer=tokenizer)
    train_dataloaders = {task_name: DataLoader( task_valid_data, batch_size=args.bs, collate_fn=data_collator, shuffle = True) 
                                                                                    for task_name, task_valid_data in sequential_train_data.items()}     
    valid_dataloaders = {task_name: DataLoader( task_valid_data, batch_size=args.bs, collate_fn=data_collator, shuffle = False) 
                                                                                    for task_name, task_valid_data in sequential_valid_data.items()} 


    all_rouge_score_stats = {}
    olora_past_A_matrices = {}
    olora_past_B_matrices = {}
    for i, (train_task_name, train_dataloader) in enumerate(train_dataloaders.items()):
        if accelerator.is_main_process:
            logger.info("===============================================================================")
            logger.info("*******************************************************************************")
            logger.info("Training with task: %s"%train_task_name)
            logger.info("*******************************************************************************")
            logger.info("===============================================================================")

        if args.lora_target_modules:
            peft_config = LoraConfig( task_type=TaskType.CAUSAL_LM, inference_mode=False, bias = "none", target_modules=args.lora_target_modules,
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, fan_in_fan_out = args.lora_fan_in_fan_out, use_dora=True  # Enabling DoRA)
        else:
            peft_config = LoraConfig( task_type=TaskType.CAUSAL_LM, inference_mode=False, bias = "none",
                                        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, fan_in_fan_out = args.lora_fan_in_fan_out, use_dora=True  # Enabling DoRA)
        model = get_peft_model(model, peft_config)
        # model.print_trainable_parameters()

        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

        if accelerator.is_main_process:
            logger.info("Listing the trainable layers:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    #print(name)
                    logger.info(name)

        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = args.train_epochs * num_update_steps_per_epoch
        num_warmup_steps = args.warm_up_epochs * num_update_steps_per_epoch 
        lr_scheduler = get_scheduler(   args.scheduler_type,
                                        optimizer=optimizer,
                                        num_warmup_steps=num_warmup_steps,
                                        num_training_steps=num_training_steps)
           

        rouge_score_records = {}
        rouge_score = evaluate.load("rouge")
        for epoch in range(1, args.train_epochs+1):
            
            rouge_score_records[epoch] = []
            train_loss, train_orth_loss = train(model, train_dataloader, optimizer, lr_scheduler,
                                                         args.llambda, olora_past_A_matrices, olora_past_B_matrices, accelerator, args.normalized_lora)
            if accelerator.is_main_process:
                logger.info("EPOCH: %d"%epoch)
                logger.info("Train CE loss = %f"%train_loss)
                logger.info("Train Orthogonal loss = %f"%train_orth_loss)
                logger.info("Learning Rate = %f" %lr_scheduler.get_lr()[-1])

            for task_name, valid_dataloader in valid_dataloaders.items():
                val_loss, result = validate(model, valid_dataloader, rouge_score, accelerator.device)
                rouge_score_records[epoch].append({task_name: result})
                # Collecting validation ROUGE scores and reporting stats
                if accelerator.is_main_process:
                    logger.info("-----------------------------------------------------")
                    logger.info(task_name)
                    logger.info("ROUGE Scores:")
                    logger.info(result)
                    logger.info("Valid loss = %f"%val_loss)
                    logger.info("======================================================")
        if accelerator.is_main_process:
            logger.info("======================= SUMMARY =======================")
            logger.info(rouge_score_records)
        all_rouge_score_stats[train_task_name] = rouge_score_records
        

        accelerator.wait_for_everyone()
        model = accelerator.unwrap_model(model)
        accelerator.clear()
        # state._reset_state()
        del optimizer, lr_scheduler, train_dataloader
        torch.cuda.empty_cache()

        # Saving the LoRA_A adapters of the current task
        for param in model.parameters():
            param.requires_grad = False
        olora_past_A_matrices[train_task_name] = { k: param.to('cpu') for k, param in model.named_parameters() if 'lora_A' in k}
        olora_past_B_matrices[train_task_name] = { k: param.to('cpu') for k, param in model.named_parameters() if 'lora_B' in k}
        # Merging the current task LoRA layers to the model and removing the lora components.
        model = model.merge_and_unload()
        # for name,param in model.named_parameters():
        #     print(name)

        if args.save_model:
            save_name = os.path.join(args.experiment_dir,'model_params_after_task%d'%i)
            model.save_pretrained(save_name, from_pt=True) 
    
    # if accelerator.is_main_process:
    #     logger.info(olora_past_A_matrices)
    #     print(olora_past_A_matrices)

    if args.print_generations:
        for task_name, valid_dataloader in valid_dataloaders.items():
            if accelerator.is_main_process:
                logger.info(task_name)
                get_generations_on_dataloader(model, tokenizer, valid_dataloader, accelerator.device)

    with open(os.path.join(args.experiment_dir,'experiment_results.pickle'), 'wb') as handle:
        pickle.dump(all_rouge_score_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    torch.save(olora_past_A_matrices, os.path.join(args.experiment_dir,'olora_A_adapters.pt'))
    torch.save(olora_past_B_matrices, os.path.join(args.experiment_dir, 'olora_B_adapters.pt'))
