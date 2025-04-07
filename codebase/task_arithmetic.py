import torch
import os
import json
import numpy as np
import logging
import pickle
from copy import deepcopy
import argparse
from datetime import timedelta

from transformers import GPT2Tokenizer, GPT2LMHeadModel, OPTForCausalLM, AutoModelForCausalLM
from transformers import get_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW

import evaluate
from utils import *
from custom_data_collator import CustomCollatorwithLabelPadding
from options import Options
from accelerate import Accelerator
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from dataloaders import *

# Set up logging globally
logging.basicConfig(
    format='%(asctime)s %(message)s',
    filemode='w'
)
logger = logging.getLogger("Task Arithmetic Logger")
logger.setLevel(logging.INFO)

def validate(model, dataloader, rouge_score, device, tokenizer):
    """Validate the model on the given dataloader."""
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
        
            generated_tokens = model.generate(generation_text, attention_mask=generation_attention, max_new_tokens=32)
            decoded_generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            decoded_generations = [gen.split('### Response:\n')[1] for gen in decoded_generations]
            decoded_generations = [gen.split(tokenizer.eos_token)[0] for gen in decoded_generations]
            rouge_score.add_batch(predictions=decoded_generations, references=decoded_labels)
            
            perfect_matches = [1 if lab == gen else 0 for (lab, gen) in zip(decoded_labels, decoded_generations)]
            val_accuracy = val_accuracy + sum(perfect_matches)

    val_loss = val_loss / len(dataloader)
    val_accuracy = val_accuracy / len(dataloader.dataset)
    result = rouge_score.compute()
    result['accuracy'] = val_accuracy
    result = {key: 100*value for key, value in result.items()}

    return val_loss, result

def get_tokenizer(args):
    # Check if local_model_path exists in args
    if hasattr(args, 'local_model_path') and args.local_model_path:
        model_name = args.local_model_path
    else:
        model_name = args.model_name
        
    if 'gpt2' in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    elif 'opt' in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        print('No such model is supported now!')
    return tokenizer

def get_model(args, use_lora=False, gradient_checkpointing=False):
    """Get the model with optional LoRA."""
    # Extract model name from args if it's an args object
    if hasattr(args, 'model_name'):
        model_name = args.model_name
    else:
        model_name = args
        
    logger.info(f"Loading model: {model_name}")
    
    # Load model based on type with half precision
    if 'gpt2' in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch.float16)
    elif 'opt' in model_name:
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    if use_lora:
        logger.info("Creating LoRA config...")
        # Get LoRA parameters from args if available
        lora_r = getattr(args, 'lora_r', 16)
        lora_alpha = getattr(args, 'lora_alpha', 16)
        lora_dropout = getattr(args, 'lora_dropout', 0.1)
        lora_target_modules = getattr(args, 'lora_target_modules', ["q_proj", "v_proj"])
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        logger.info("LoRA config created successfully")
        
        logger.info("Creating template model...")
        model = get_peft_model(model, lora_config)
        logger.info("Template model created successfully")
    
    return model

def load_lora_matrices():
    """Load LoRA matrices from the specified directory."""
    lora_dir = "codebase/task_arithmetic_parameters/"
    lora_A_matrices_path = os.path.join(lora_dir, "lora_A_matrices.pt")
    lora_B_matrices_path = os.path.join(lora_dir, "lora_B_matrices.pt")
    
    if not os.path.exists(lora_A_matrices_path) or not os.path.exists(lora_B_matrices_path):
        raise FileNotFoundError(f"LoRA matrices not found in {lora_dir}")
    
    lora_A_matrices = torch.load(lora_A_matrices_path)
    lora_B_matrices = torch.load(lora_B_matrices_path)
    
    return lora_A_matrices, lora_B_matrices

def apply_lora_parameters(model, task_name, lora_A_matrices, lora_B_matrices, logger):
    """Apply LoRA parameters for a specific task to the model."""
    with torch.no_grad():
        # Get all LoRA A parameters from the model
        lora_A_params = {name: param for name, param in model.named_parameters() if "lora_A" in name}
        
        for name, param in lora_A_params.items():
            b_layer_name = name.replace("lora_A", "lora_B")
            
            # Check if the layer exists in the task matrices
            task_layer_name = name if name in lora_A_matrices[task_name] else f"module.{name}"
            task_b_layer_name = b_layer_name if b_layer_name in lora_B_matrices[task_name] else f"module.{b_layer_name}"
            
            if task_layer_name in lora_A_matrices[task_name] and task_b_layer_name in lora_B_matrices[task_name]:
                # Get parameters and ensure they're on the same device as the target parameter
                task_A = lora_A_matrices[task_name][task_layer_name].to(param.device)
                task_B = lora_B_matrices[task_name][task_b_layer_name].to(param.device)
                
                # Update parameters
                param.data.copy_(task_A)
                model.get_parameter(b_layer_name).data.copy_(task_B)
                logger.info(f"Updated parameters for layer {name} for task {task_name} on device {param.device}")
            else:
                logger.warning(f"Layer {name} not found in task {task_name} matrices")

def create_task_model(model_template, peft_config):
    """Create a model with LoRA layers for a specific task."""
    # Create a new model with the same configuration
    if isinstance(model_template, GPT2LMHeadModel):
        task_model = GPT2LMHeadModel.from_pretrained(model_template.config._name_or_path)
    elif isinstance(model_template, OPTForCausalLM):
        task_model = OPTForCausalLM.from_pretrained(model_template.config._name_or_path)
    else:
        raise ValueError(f"Unsupported model type: {type(model_template)}")
    
    # Apply LoRA config to the task model
    task_model = get_peft_model(task_model, peft_config)
    return task_model

def create_accumulator_model(model_template, peft_config):
    """Create a model for accumulating parameters from all tasks."""
    # Create a new model with the same configuration
    # Instead of deepcopy, create a fresh instance
    if isinstance(model_template, GPT2LMHeadModel):
        accumulator_model = GPT2LMHeadModel.from_pretrained(model_template.config._name_or_path)
    elif isinstance(model_template, OPTForCausalLM):
        accumulator_model = OPTForCausalLM.from_pretrained(model_template.config._name_or_path)
    else:
        raise ValueError(f"Unsupported model type: {type(model_template)}")
    
    # Do NOT apply LoRA config to the accumulator model
    # We want a base model without LoRA layers
    return accumulator_model

def copy_parameters(source_model, target_model, logger):
    """Copy parameters from source model to target model."""
    with torch.no_grad():
        # Get all parameters from both models
        source_params = {name: param for name, param in source_model.named_parameters()}
        target_params = {name: param for name, param in target_model.named_parameters()}
        
        # Print first few parameter names for debugging
        logger.info("First few parameter names from source model:")
        for i, name in enumerate(list(source_params.keys())[:5]):
            logger.info(f"  [Source] {name}")
            
        logger.info("First few parameter names from target model:")
        for i, name in enumerate(list(target_params.keys())[:5]):
            logger.info(f"  [Target] {name}")
        
        # Create a mapping from source parameter names to target parameter names
        param_mapping = {}
        for target_name in target_params.keys():
            # Remove the 'base_model.model.' prefix from source names to match target names
            for source_name in source_params.keys():
                if source_name.replace('base_model.model.', '') == target_name:
                    param_mapping[source_name] = target_name
                    break
        
        # Copy parameters using the mapping
        copied_count = 0
        for source_name, target_name in param_mapping.items():
            source_param = source_params[source_name]
            target_param = target_params[target_name]
            
            # Ensure parameters have the same shape
            if source_param.shape != target_param.shape:
                logger.warning(f"Shape mismatch for {source_name} -> {target_name}: {source_param.shape} vs {target_param.shape}")
                continue
                
            # Copy parameters
            target_param.data.copy_(source_param.data)
            copied_count += 1
            logger.info(f"Copied parameters from {source_name} to {target_name}")
        
        logger.info(f"Copied {copied_count} parameters out of {len(target_params)} target parameters")
        
        # Check for parameters that weren't copied
        not_copied = set(target_params.keys()) - set(param_mapping.values())
        if not_copied:
            logger.warning(f"Parameters not copied: {list(not_copied)[:5]}... (total: {len(not_copied)})")

def accumulate_parameters(accumulator_model, task_model, logger):
    """Accumulate parameters from a task model into the accumulator model."""
    with torch.no_grad():
        # Get all parameters from both models
        task_params = {name: param for name, param in task_model.named_parameters()}
        acc_params = {name: param for name, param in accumulator_model.named_parameters()}
        
        # Print first few parameter names for debugging
        logger.info("First few parameter names from task model:")
        for i, name in enumerate(list(task_params.keys())[:5]):
            logger.info(f"  [Task] {name}")
            
        logger.info("First few parameter names from accumulator model:")
        for i, name in enumerate(list(acc_params.keys())[:5]):
            logger.info(f"  [Acc] {name}")
        
        # Create a mapping from task parameter names to accumulator parameter names
        param_mapping = {}
        for acc_name in acc_params.keys():
            # Remove the 'base_model.model.' prefix from task names to match accumulator names
            for task_name in task_params.keys():
                if task_name.replace('base_model.model.', '') == acc_name:
                    param_mapping[task_name] = acc_name
                    break
        
        # Accumulate parameters using the mapping
        accumulated_count = 0
        for task_name, acc_name in param_mapping.items():
            task_param = task_params[task_name]
            acc_param = acc_params[acc_name]
            
            # Ensure parameters have the same shape
            if task_param.shape != acc_param.shape:
                logger.warning(f"Shape mismatch for {task_name} -> {acc_name}: {task_param.shape} vs {acc_param.shape}")
                continue
                
            # Add task parameters to accumulator
            acc_param.data.add_(task_param.data)
            accumulated_count += 1
            logger.info(f"Accumulated parameters from {task_name} to {acc_name}")
        
        logger.info(f"Accumulated {accumulated_count} parameters out of {len(acc_params)} accumulator parameters")
        
        # Check for parameters that weren't accumulated
        not_accumulated = set(acc_params.keys()) - set(param_mapping.values())
        if not_accumulated:
            logger.warning(f"Parameters not accumulated: {list(not_accumulated)[:5]}... (total: {len(not_accumulated)})")

def average_parameters(accumulator_model, num_tasks, logger):
    """Average parameters in the accumulator model by dividing by the number of tasks."""
    with torch.no_grad():
        # Get all parameters from the accumulator model
        params = {name: param for name, param in accumulator_model.named_parameters()}
        
        # Average parameters
        for name, param in params.items():
            # Divide by the number of tasks
            param.data.div_(num_tasks)
            logger.info(f"Averaged parameters for layer {name}")

def setup_logging(experiment_dir):
    """Set up logging configuration."""
    # Create log file path
    log_file = os.path.join(experiment_dir, "task_arithmetic.log")
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    # Log initial message
    logger.info("Starting task arithmetic validation")
    logger.info(f"Log file: {log_file}")

def main():
    # Initialize accelerator with default settings
    accelerator = Accelerator()
    args = Options(accelerator=accelerator).parse()

    # Initialize logger for all processes
    if accelerator.is_main_process:
        logging.basicConfig(filename=os.path.join(args.experiment_dir, "task_arithmetic_validation.log"),
                          format='%(asctime)s %(message)s',
                          filemode='w')
        logger = logging.getLogger("Task Arithmetic Validation Logger")
        logger.setLevel(logging.INFO)
    else:
        # Create a dummy logger for non-main processes
        logger = logging.getLogger("Task Arithmetic Validation Logger")
        logger.setLevel(logging.INFO)
        # Add a null handler to prevent "No handler found" warnings
        logger.addHandler(logging.NullHandler())

    # Model and tokenizer
    tokenizer = get_tokenizer(args)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Clear memory at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if accelerator.is_main_process:
            logger.info(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create base model template
    if accelerator.is_main_process:
        logger.info("Creating base model template...")
    base_model = get_model(
        args,
        use_lora=False,  # Don't use LoRA for base model
        gradient_checkpointing=args.gradient_checkpointing
    )
    if accelerator.is_main_process:
        logger.info("Base model template created successfully")

    # Create a template model for reference
    logger.info("Creating template model...")
    template_model = get_model(args)
    logger.info("Template model created successfully")
    
    # Load LoRA matrices
    logger.info("Loading LoRA matrices...")
    lora_A_matrices, lora_B_matrices = load_lora_matrices()
    logger.info(f"Loaded LoRA matrices for tasks: {list(lora_A_matrices.keys())}")

    # Initialize dictionaries to store results
    task_results = {}
    final_results = {}
    accumulator_params = {}

    # Initialize parameter accumulator
    if accelerator.is_main_process:
        logger.info("Initializing parameter accumulator...")
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            accumulator_params[name] = torch.zeros_like(param.data)
    if accelerator.is_main_process:
        logger.info("Parameter accumulator initialized successfully")

    # Process each task
    for task in args.tasks:
        if accelerator.is_main_process:
            logger.info(f"Processing task: {task}")
        
        # Create task model
        if accelerator.is_main_process:
            logger.info(f"Creating task model for {task}...")
        task_model = get_model(
            args,
            use_lora=args.use_lora,
            gradient_checkpointing=args.gradient_checkpointing
        )
        if accelerator.is_main_process:
            logger.info(f"Task model for {task} created successfully")
        
        # Move task model to device
        if accelerator.is_main_process:
            logger.info(f"Moving task model to device: {accelerator.device}")
        task_model = task_model.to(accelerator.device)
        if accelerator.is_main_process:
            logger.info("Task model moved to device successfully")
        
        # Apply LoRA parameters for this task
        apply_lora_parameters(task_model, task, lora_A_matrices, lora_B_matrices, logger)
        
        # Merge LoRA layers into the base model before accumulation
        logger.info(f"Merging LoRA layers for task {task}")
        task_model = task_model.merge_and_unload()
        
        # Run validation for this task
        if accelerator.is_main_process:
            logger.info(f"Running validation for task {task}...")
        
        # Create a rouge score object for this task
        rouge_score = evaluate.load("rouge")
        if accelerator.is_main_process:
            logger.info("Rouge score object created")
        
        # Create a dataloader for this task
        if accelerator.is_main_process:
            logger.info(f"Loading dataset for task {task}")
        _, testset = GET_DATASET[task]()
        task_data = preprocess_dataset_list(testset, tokenizer)
        data_collator = CustomCollatorwithLabelPadding(tokenizer=tokenizer)
        task_dataloader = DataLoader(task_data, batch_size=args.bs, collate_fn=data_collator, shuffle=False)
        if accelerator.is_main_process:
            logger.info(f"Dataset for task {task} loaded successfully")
        
        # Run validation with correct parameters
        task_results[task] = validate(
            task_model,
            task_dataloader,
            rouge_score,
            accelerator.device,
            tokenizer
        )
        if accelerator.is_main_process:
            logger.info(f"Validation for task {task} completed successfully")
        
        # Accumulate parameters
        if accelerator.is_main_process:
            logger.info(f"Accumulating parameters for task {task}...")
        with torch.no_grad():
            for name, param in task_model.named_parameters():
                if name in accumulator_params:
                    # Move to CPU for accumulation
                    param_cpu = param.data.cpu()
                    accumulator_params[name].add_(param_cpu)
        if accelerator.is_main_process:
            logger.info(f"Parameters for task {task} accumulated successfully")
        
        # Clear task model from memory
        del task_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if accelerator.is_main_process:
                logger.info(f"Task model cleared from memory. Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Average accumulated parameters
    if accelerator.is_main_process:
        logger.info("Averaging accumulated parameters...")
    with torch.no_grad():
        for name in accumulator_params:
            accumulator_params[name].div_(len(args.tasks))
            base_model.state_dict()[name].copy_(accumulator_params[name])
    if accelerator.is_main_process:
        logger.info("Parameters averaged successfully")
        
    # Save the averaged model parameters
    if accelerator.is_main_process:
        logger.info("Saving averaged model parameters...")
        save_path = os.path.join(args.experiment_dir, "averaged_model")
        os.makedirs(save_path, exist_ok=True)
        base_model.save_pretrained(save_path)
        logger.info(f"Averaged model parameters saved to {save_path}")
        
        # Also save the state dict directly for easier loading
        torch.save(base_model.state_dict(), os.path.join(args.experiment_dir, "averaged_model_state_dict.pt"))
        logger.info("Averaged model state dict saved")
        
        # Save accumulator parameters for reference
        torch.save(accumulator_params, os.path.join(args.experiment_dir, "accumulator_params.pt"))
        logger.info("Accumulator parameters saved")
        
        logger.info("All parameters saved successfully. Continuing with validation.")

    # The following validation code will only run if we don't save and exit
    # Move base model to the correct device
    base_model = base_model.to(accelerator.device)
    if accelerator.is_main_process:
        logger.info(f"Base model moved to device: {accelerator.device}")

    # Run validation with averaged model
    if accelerator.is_main_process:
        logger.info("Running validation with averaged model...")
    for task in args.tasks:
        # Create a rouge score object for this task
        rouge_score = evaluate.load("rouge")
        
        # Create a dataloader for this task
        if accelerator.is_main_process:
            logger.info(f"Loading dataset for task {task}")
        _, testset = GET_DATASET[task]()
        task_data = preprocess_dataset_list(testset, tokenizer)
        data_collator = CustomCollatorwithLabelPadding(tokenizer=tokenizer)
        task_dataloader = DataLoader(task_data, batch_size=args.bs, collate_fn=data_collator, shuffle=False)
        if accelerator.is_main_process:
            logger.info(f"Dataset for task {task} loaded successfully")
        
        # Run validation with correct parameters
        final_results[task] = validate(
            base_model,
            task_dataloader,
            rouge_score,
            accelerator.device,
            tokenizer
        )
        if accelerator.is_main_process:
            logger.info(f"Validation for task {task} using base model completed successfully")
    
    # Log results
    if accelerator.is_main_process:
        logger.info("===============================================================================")
        logger.info("************************** VALIDATION RESULTS *******************************")
        logger.info("===============================================================================")
        
        logger.info("Individual Task Results:")
        for task_name, results in task_results.items():
            logger.info(f"Task: {task_name}")
            logger.info(f"Loss: {results[0]}")
            logger.info(f"Metrics: {results[1]}")
            logger.info("-----------------------------------------------------")
        
        logger.info("Averaged Model Results:")
        for task_name, results in final_results.items():
            logger.info(f"Task: {task_name}")
            logger.info(f"Loss: {results[0]}")
            logger.info(f"Metrics: {results[1]}")
            logger.info("-----------------------------------------------------")
    
    # Clear memory
    if accelerator.is_main_process:
        logger.info("Cleaning up...")
    del base_model
    del accumulator_params
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if accelerator.is_main_process:
            logger.info(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if accelerator.is_main_process:
        logger.info("Task arithmetic validation completed successfully")

if __name__ == "__main__":
    main()
