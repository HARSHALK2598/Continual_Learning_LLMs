import torch
import os
import json
from copy import copy, deepcopy
import numpy as np

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoConfig
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import get_scheduler

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW

import evaluate
from utils import *
from dataloaders import *
from custom_data_collator import CustomCollatorwithLabelPadding
import datetime

import logging
from options import Options
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


def train(model, dataloader, optimizer, scheduler):

    model.train()
    train_loss=0
    for batch in dataloader:
        optimizer.zero_grad()
        _, _ = batch.pop('generation_text'), batch.pop('generation_text_attn_mask')
        batch = {k:v.to(device) for k,v in batch.items()}
        labels = batch.pop('labels')
        out = model(**batch) 
        logits = out.logits
        # print(logits.shape[-1])
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss = train_loss + loss.item()
        # break
    train_loss = train_loss / len(dataloader)

    return train_loss
    

def validate(model, dataloader, rouge_score):
    model.eval()
    val_loss = 0
    for batch in valid_dataloader:
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
            # break

    val_loss = val_loss / len(valid_dataloader)
    result = rouge_score.compute()
    result = {key: 100*value for key, value in result.items()}


    return val_loss, result


if __name__ == '__main__':

    args = Options().parse()
    logging.basicConfig(filename=os.path.join(args.experiment_dir, "logs.log"),
                                            format='%(asctime)s %(message)s',
                                            filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Model and tokenizer
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, tokenizer = get_model_and_tokenizer(args)
    model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token	
    
    # PEFT
    if args.use_lora:
        logger.info("Using Lora!")
        if args.lora_target_modules:
            peft_config = LoraConfig( task_type="CAUSAL_LM", inference_mode=False, bias = "none", target_modules=args.lora_target_modules,
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, fan_in_fan_out = args.lora_fan_in_fan_out  )
        else:
            peft_config = LoraConfig( task_type="CAUSAL_LM", inference_mode=False, bias = "none",
                                     r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, fan_in_fan_out = args.lora_fan_in_fan_out  )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


    joint_train_data = []
    sequential_valid_data = {}
    print("Merging the tasks: ", args.tasks)
    for task in args.tasks:
        logger.info("Preparing the %s dataset for joint training."%task)
        trainset, testset = GET_DATASET[task]()
        task_preprocessed_train_data = preprocess_dataset_list(trainset, tokenizer)
        task_preprocessed_valid_data = preprocess_dataset_list(testset, tokenizer)
        joint_train_data = joint_train_data + task_preprocessed_train_data
        logger.info("Current length of the training dataset is %d."%len(joint_train_data))
        sequential_valid_data[task] = task_preprocessed_valid_data

    # Preparing dataloaders for training and evaluation
    data_collator = CustomCollatorwithLabelPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader( joint_train_data, batch_size=args.bs, collate_fn=data_collator, shuffle = True)            
    valid_dataloaders = {task_name: DataLoader( task_valid_data, batch_size=args.bs, collate_fn=data_collator, shuffle = False) 
                                                                                    for task_name, task_valid_data in sequential_valid_data.items()} 

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = args.train_epochs * num_update_steps_per_epoch
    num_warmup_steps = args.warm_up_epochs * num_update_steps_per_epoch
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps)

    rouge_score_records = {}

    rouge_score = evaluate.load("rouge")
    for epoch in range(1, args.train_epochs+1):
        
        rouge_score_records[epoch] = []
        train_loss = train(model, train_dataloader, optimizer, lr_scheduler)
        logger.info("EPOCH: %d"%epoch)
        logger.info("Train loss = %f"%train_loss)
        logger.info("Learning Rate = %f" %lr_scheduler.get_lr()[-1])

        for task_name, valid_dataloader in valid_dataloaders.items():
            val_loss, result = validate(model, valid_dataloader, rouge_score)
            rouge_score_records[epoch].append({task_name: result})
            # Collecting validation ROUGE scores and reporting stats
            logger.info("-----------------------------------------------------")
            logger.info(task_name)
            logger.info("ROUGE Scores:")
            logger.info(result)
            logger.info("Valid loss = %f"%val_loss)
        logger.info("============================================================================")
    
    logger.info("======================= SUMMARY =======================")
    logger.info(rouge_score_records)

    with open(os.path.join(args.experiment_dir,'experiment_results.pickle'), 'wb') as handle:
        pickle.dump(rouge_score_records, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.print_generations:
        for task_name, valid_dataloader in valid_dataloaders.items():
            logger.info(task_name)
            get_generations_on_dataloader(model, tokenizer, valid_dataloader, device)

    if args.save_model:
        save_name = os.path.join(args.experiment_dir,'model_params')
        model.save_pretrained(save_name, from_pt=True) 