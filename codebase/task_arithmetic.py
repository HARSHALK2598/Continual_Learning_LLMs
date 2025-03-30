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
from custom_data_collator import CustomCollatorwithLabelPadding
import datetime

import logging
from options import Options
import pickle
from accelerate import Accelerator
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from dataloaders import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train(model, dataloader, optimizer, scheduler, accelerator):

    model.train()
    train_loss=0
    for batch in dataloader:
        optimizer.zero_grad()
        _, _ = batch.pop('generation_text'), batch.pop('generation_text_attn_mask')
        batch = {k:v.to(accelerator.device) for k,v in batch.items()}
        labels = batch.pop('labels')
        out = model(**batch) 
        logits = out.logits
        # print(logits.shape[-1])
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        #loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        train_loss = train_loss + loss.item()
        # break
        
    train_loss = train_loss / len(dataloader)

    return train_loss
    

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
    
            generated_tokens = model.generate(generation_text, attention_mask=generation_attention, max_new_tokens = 32)
            decoded_generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            decoded_generations = [gen.split('### Response:\n')[1] for gen in decoded_generations]
            decoded_generations = [gen.split(tokenizer.eos_token)[0] for gen in decoded_generations]
            rouge_score.add_batch(predictions=decoded_generations, references=decoded_labels)
            
            perfect_matches = [1 if lab == gen else 0 for (lab, gen) in zip(decoded_labels, decoded_generations)]
            logger.info(f"Perfect matches: {perfect_matches}")
            val_accuracy = val_accuracy + sum(perfect_matches)
            break

    val_loss = val_loss / len(dataloader)
    val_accuracy = val_accuracy / len(dataloader.dataset)
    result = rouge_score.compute()
    result['accuracy'] = val_accuracy
    result = {key: 100*value for key, value in result.items()}

    return val_loss, result
def get_tokenizer(args):

    model_name = args.local_model_path if args.local_model_path else args.model_name
    if 'gpt2' in args.model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)    
        #model = GPT2LMHeadModel.from_pretrained(model_name)
    
    elif 'opt' in args.model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)    
        #model = OPTForCausalLM.from_pretrained(model_name)
    
    else:
        logger.error('No such model is supported now!')
        raise ValueError('Unsupported model type')

    return tokenizer


def get_model(args):

    model_name = args.local_model_path if args.local_model_path else args.model_name
    if 'gpt2' in args.model_name:
        #tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)    
        model = GPT2LMHeadModel.from_pretrained(model_name)
    
    elif 'opt' in args.model_name:
        #tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)    
        model = OPTForCausalLM.from_pretrained(model_name)
    
    else:
        logger.error('No such model is supported now!')
        raise ValueError('Unsupported model type')

    return model

def load_lora_matrices(experiment_dir):
    """ Load stored LoRA A and B matrices. """
    lora_dir = "codebase/task_arithmetic_parameters"
    lora_A_matrices = torch.load(os.path.join(lora_dir, "lora_A_matrices.pt"), map_location='cpu')
    lora_B_matrices = torch.load(os.path.join(lora_dir, "lora_B_matrices.pt"), map_location='cpu')
    
    # Convert to float32 if needed
    if isinstance(lora_A_matrices, dict):
        for key in lora_A_matrices.keys():
            if isinstance(lora_A_matrices[key], dict):
                for subkey in lora_A_matrices[key].keys():
                    if isinstance(lora_A_matrices[key][subkey], torch.Tensor):
                        tensor = lora_A_matrices[key][subkey]
                        if tensor.dtype != torch.float32:
                            lora_A_matrices[key][subkey] = tensor.float()
    
    if isinstance(lora_B_matrices, dict):
        for key in lora_B_matrices.keys():
            if isinstance(lora_B_matrices[key], dict):
                for subkey in lora_B_matrices[key].keys():
                    if isinstance(lora_B_matrices[key][subkey], torch.Tensor):
                        tensor = lora_B_matrices[key][subkey]
                        if tensor.dtype != torch.float32:
                            lora_B_matrices[key][subkey] = tensor.float()
    
    return lora_A_matrices, lora_B_matrices

def average_lora_parameters(model, past_task_weights_A, past_task_weights_B, logger):
    """ Average LoRA parameters across tasks and assign them to the model. """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" in name:
                layer_name = name
                b_layer_name = name.replace("lora_A", "lora_B")
                
                # Check if the layer exists in any task (with or without module prefix)
                found_in_tasks = []
                for task_name in past_task_weights_A.keys():
                    if layer_name in past_task_weights_A[task_name] or f"module.{layer_name}" in past_task_weights_A[task_name]:
                        found_in_tasks.append(task_name)
                
                if found_in_tasks:
                    total_A, total_B, count = 0, 0, 0
                    
                    for task_name in found_in_tasks:
                        # Try both with and without module prefix
                        task_layer_name = layer_name if layer_name in past_task_weights_A[task_name] else f"module.{layer_name}"
                        task_b_layer_name = b_layer_name if b_layer_name in past_task_weights_B[task_name] else f"module.{b_layer_name}"
                        
                        if task_b_layer_name in past_task_weights_B[task_name]:
                            # Ensure tensors are on the same device as the model
                            task_A = past_task_weights_A[task_name][task_layer_name].to(param.device)
                            task_B = past_task_weights_B[task_name][task_b_layer_name].to(param.device)
                            
                            total_A += task_A
                            total_B += task_B
                            count += 1
                    
                    if count > 0:
                        avg_A = total_A / count
                        avg_B = total_B / count
                        
                        # Update the parameters directly
                        param.copy_(avg_A)
                        # Find and update the corresponding B parameter
                        for b_name, b_param in model.named_parameters():
                            if b_name == b_layer_name:
                                b_param.copy_(avg_B)
                                break

if __name__ == '__main__':

    accelerator = Accelerator()
    args = Options(accelerator=accelerator).parse()

    # Model and tokenizer
    tokenizer = get_tokenizer(args)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token	

    if args.lora_target_modules:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, bias="none",
            target_modules=args.lora_target_modules, r=args.lora_r, lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout, fan_in_fan_out=args.lora_fan_in_fan_out)
    else:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, bias="none",
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, 
            fan_in_fan_out=args.lora_fan_in_fan_out)

    past_task_weights_A, past_task_weights_B = load_lora_matrices(args.experiment_dir)
    model = get_model(args)
    model = get_peft_model(model, peft_config)
    
    if accelerator.is_main_process:
        average_lora_parameters(model, past_task_weights_A, past_task_weights_B, logger)
        
        # Create dummy optimizer and scheduler for validation
        optimizer = AdamW(model.parameters(), lr=1e-5)
        num_training_steps = 1
        num_warmup_steps = 0
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # Prepare model, optimizer, and scheduler with accelerator
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        
        # Load validation dataloaders
        valid_dataloaders = {}
        for task in args.tasks:
            _, testset = GET_DATASET[task]()
            task_preprocessed_valid_data = preprocess_dataset_list(testset, tokenizer)
            data_collator = CustomCollatorwithLabelPadding(tokenizer=tokenizer)
            valid_dataloaders[task] = DataLoader(
                task_preprocessed_valid_data,
                batch_size=args.bs,
                collate_fn=data_collator,
                shuffle=False
            )
        
        # Run validation on first task only
        rouge_score = evaluate.load("rouge")
        for task_name, valid_dataloader in valid_dataloaders.items():
            val_loss, result = validate(model, valid_dataloader, rouge_score, accelerator.device)
            logger.info(f"Task: {task_name}")
            logger.info(f"Validation Loss: {val_loss:.4f}")
            logger.info(f"Validation Results: {result}")
            break  # Only check the first task

