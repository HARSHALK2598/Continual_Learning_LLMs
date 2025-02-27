import torch
import os
import json
import numpy as np

from transformers import get_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
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

from dataloaders import *
from peft import LoraConfig, get_peft_model, LoraModel, TaskType


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler
from peft import LoraConfig, get_peft_model, LoraModel, TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader

class SLoRAConfig(LoraConfig):
    def __init__(self, *args, past_task_loras=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_task_loras = past_task_loras or {}  # Store LoRA {task_name: {layer_name: (A, B, scale)}}

    def update_past_loras(self, task_name, layer_name, lora_A, lora_B, scale):
        """ Updates past LoRA parameters and scaling factors for continual learning at a per-layer level. """
        if task_name not in self.past_task_loras:
            self.past_task_loras[task_name] = {}
        self.past_task_loras[task_name][layer_name] = (lora_A, lora_B, scale)


class SLoRAModel(LoraModel):
    def __init__(self, model, peft_config):
        super().__init__(model, peft_config)

        # Attach LoRA layers for the current task
        self.model = get_peft_model(model, peft_config)

        # Dictionary to store past task LoRA parameters per layer
        self.past_loras = {}  
        for task_name, task_layers in peft_config.past_task_loras.items():
            for layer_name, (lora_A, lora_B, scale) in task_layers.items():
                if layer_name not in self.past_loras:
                    self.past_loras[layer_name] = {}
                self.past_loras[layer_name][task_name] = (
                    lora_A.clone().detach(),
                    lora_B.clone().detach(),
                    nn.Parameter(torch.tensor(scale, dtype=torch.float32, requires_grad=True))  # Trainable scale
                )

        # Trainable scaling factors for past tasks
        self.scaling_factors = nn.ModuleDict({
            layer_name: nn.ParameterDict({
                task_name: scale
                for task_name, (_, _, scale) in self.past_loras[layer_name].items()
            }) for layer_name in self.past_loras.keys()
        })

        # Trainable scaling factors for the current task per layer
        self.current_scales = nn.ParameterDict({
            layer_name: nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
            for layer_name in self.model.state_dict().keys() if "lora_A" in layer_name
        })

        # Freeze past LoRA parameters
        for layer_name, task_loras in self.past_loras.items():
            for task_name, (lora_A, lora_B, _) in task_loras.items():
                lora_A.requires_grad = False
                lora_B.requires_grad = False

    def forward(self, x):
        for layer_name, layer in self.model.named_modules():
            if hasattr(layer, "forward"):
                base_output = layer(x)  # Base model output

                # Apply past task LoRA layers per layer
                past_task_output = torch.zeros_like(base_output)
                if layer_name in self.past_loras:
                    for task_name, (lora_A, lora_B, scale) in self.past_loras[layer_name].items():
                        delta_W = torch.matmul(lora_B, lora_A)  # Compute LoRA update
                        past_task_output += self.scaling_factors[layer_name][task_name] * torch.matmul(x, delta_W.T)

                # Apply current task LoRA per layer
                if hasattr(layer, "lora_A") and hasattr(layer, "lora_B"):
                    delta_W_current = torch.matmul(layer.lora_B, layer.lora_A)
                    current_task_output = self.current_scales[layer_name] * torch.matmul(x, delta_W_current.T)
                else:
                    current_task_output = torch.zeros_like(base_output)

                # Sum all contributions
                x = base_output + past_task_output + current_task_output

        return x

def train_slora(model, dataloader, optimizer, scheduler, accelerator):
    model.train()
    train_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        _, _ = batch.pop('generation_text'), batch.pop('generation_text_attn_mask')
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        labels = batch.pop('labels')

        out = model(**batch)
        logits = out.logits
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        #break 
    return train_loss / len(dataloader)

def validate_slora(model, dataloader, rouge_score, device):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
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

def store_lora_parameters(peft_config, model, stored_loras, task_name):
    """ Store trained LoRA parameters after training each task. """
    for layer_name, layer in model.model.named_modules():
        if hasattr(layer, "lora_A") and hasattr(layer, "lora_B"):
            if layer_name not in stored_loras:
                stored_loras[layer_name] = {}
            stored_loras[layer_name][task_name] = (
                layer.lora_A.clone().detach(),
                layer.lora_B.clone().detach(),
                model.current_scales[layer_name].clone().detach()
            )


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
    stored_loras = {}
    for i, (train_task_name, train_dataloader) in enumerate(train_dataloaders.items()):
        if accelerator.is_main_process:
            logger.info("===============================================================================")
            logger.info("*******************************************************************************")
            logger.info("Training with task: %s"%train_task_name)
            logger.info("*******************************************************************************")
            logger.info("===============================================================================")

        # Load model with PEFT modifications for S-LoRA
        peft_config = SLoRAConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            bias="none",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            fan_in_fan_out=args.lora_fan_in_fan_out,
            past_task_loras=stored_loras  # Consume past LoRA matrices and scaling factors
        )

        model = SLoRAModel(model, peft_config)

        # Freeze base model
        for param in model.base_model.parameters():
            param.requires_grad = False

        # Load dataset
        train_dataloader, valid_dataloader = get_dataloaders(task_name, args)

        # Define optimizer and scheduler
        optimizer = AdamW([
            {'params': list(model.scaling_factors.parameters()), 'lr': args.lr},
            {'params': list(model.current_scales.parameters()), 'lr': args.lr},
            {'params': [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]}
        ], lr=args.lr, weight_decay=args.wd)

        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = args.train_epochs * num_update_steps_per_epoch
        num_warmup_steps = args.warm_up_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(args.scheduler_type, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

        if accelerator.is_main_process:
            logger.info("Listing the trainable layers:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name)
                    logger.info(name)

        rouge_score_records = {}
        rouge_score = evaluate.load("rouge")
        for epoch in range(1, args.train_epochs+1):

            rouge_score_records[epoch] = []
            train_loss = train_slora(model, train_dataloader, optimizer, lr_scheduler, accelerator)

            if accelerator.is_main_process:
                logger.info("EPOCH: %d"%epoch)
                logger.info("Train CE loss = %f"%train_loss)
                logger.info("Learning Rate = %f" %lr_scheduler.get_lr()[-1])

            for task_name, valid_dataloader in valid_dataloaders.items():
                val_loss, result = validate_slora(model, valid_dataloader, rouge_score, accelerator.device)
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
        del optimizer, lr_scheduler, train_dataloader
        torch.cuda.empty_cache()

        # Store LoRA parameters per layer
        for layer_name, layer in model.model.named_modules():
            if hasattr(layer, "lora_A") and hasattr(layer, "lora_B"):
                peft_config.update_past_loras(
                    task_name,
                    layer_name,
                    layer.lora_A.clone().detach(),
                    layer.lora_B.clone().detach(),
                    model.current_scales[layer_name].clone().detach()
                )
        
        # Store LoRA parameters per layer
        store_lora_parameters(peft_config, model, stored_loras, task_name)
        

    if args.print_generations:
        for task_name, valid_dataloader in valid_dataloaders.items():
            if accelerator.is_main_process:
                logger.info(task_name)
                get_generations_on_dataloader(model, tokenizer, valid_dataloader, accelerator.device)

    with open(os.path.join(args.experiment_dir,'experiment_results.pickle'), 'wb') as handle:
        pickle.dump(all_rouge_score_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
