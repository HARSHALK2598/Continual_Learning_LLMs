import numpy as np
import json
from copy import copy, deepcopy
import torch
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OPTForCausalLM

def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n### Response:\n").format_map(row)

def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n### Response:\n").format_map(row)

def create_text_sample(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)

def preprocess_sample(sample, tokenizer):
    """
    Applies preprocessing on the given text sample.
    sample: tuple(string, string)
    tokenizer: tokenizer object
    """

    (inst, target) = sample
    full_text = inst + target

    generation_tokens = tokenizer(inst)
    tokens = tokenizer(full_text)
    
    inst_length = len(tokenizer(inst).input_ids)
    labels = inst_length * [-100]
    labels = labels + tokens['input_ids'][inst_length:]
    # do the shifting
    # I like calculating the loss in the old school way so I need to apply shifting manually
    preprocessed_input = {'input_ids': tokens['input_ids'][:-1], 
                            'attention_mask': tokens['attention_mask'][:-1],
                            'labels': labels[1:],
                            'generation_text': generation_tokens['input_ids'],
                            'generation_text_attn_mask': generation_tokens['attention_mask']}

    return preprocessed_input

def preprocess_dataset_list(instances, tokenizer, max_length = 512):
    """
    Forms instruction/prompt/input part from the text sample and also
    retrieves the target output text appended with the EOS token
    instances: list[dict, dict, ..., dict] -> each dict is a sample
    tokenizer: tokenizer object
    max_length: maximum length a sample can have after tokenization
    """
    instruction_list = [create_text_sample(dct) for dct in instances]  # all LLM inputs are here
    target_outputs = [dct['output'] + tokenizer.eos_token for dct in instances]
    dataset_list = [(inst, target) for (inst, target) in zip(instruction_list, target_outputs)]

    preprocessed_list = []
    for sample in dataset_list:
        preprocessed = preprocess_sample(sample, tokenizer)
        if len(preprocessed['input_ids']) < max_length:
            preprocessed_list.append(preprocessed)
        # else:
        #     print("Too long for the model!")
    return preprocessed_list



def get_generations_on_dataloader(model, tokenizer, dataloader, device):

    logger = logging.getLogger("Experiment Logger") 
    logger.info("======================================================")
    logger.info("================== GENERATION PHASE ==================")
    logger.info("======================================================")

    model.eval()
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            labels = batch.pop('labels')
            labels_np = labels.cpu().numpy()
            labels_np = np.where(labels_np != -100, labels_np, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels_np, skip_special_tokens=True)

            batch = {k:v.to(device) for k,v in batch.items()}
            generation_text, generation_attention = batch.pop('generation_text'), batch.pop('generation_text_attn_mask')
            generated_tokens = model.generate( generation_text,  attention_mask=generation_attention, max_new_tokens = 32)
            decoded_generations = tokenizer.batch_decode( generated_tokens, skip_special_tokens=False)
            decoded_generations = [gen.split('### Response:\n')[1] for gen in decoded_generations]
            decoded_generations = [gen.split(tokenizer.eos_token)[0] for gen in decoded_generations]
            logger.info("Target Labels:")
            logger.info(decoded_labels)
            logger.info("Model Generations:")
            logger.info(decoded_generations)


def load_dataset(dataset_path):
    """
    DEPRECATED: This function was written for the Natural-Instructions dataset.
                Dataloader function are in the dataloaders.py now.
    """
    with open(dataset_path) as f:
        dataset_json= json.load(f)
    # dataset_json.keys():
    # dict_keys(['Title', 'Prompt', 'Definition', 'Things to Avoid', 'Emphasis & Caution', 'Instances', 'Examples'])
    # I guess we only need to use the prompt and instances.
    prompt = dataset_json['Prompt']
    instances = dataset_json['Instances']

    for dct in instances:
        dct['instruction'] = prompt
        dct['output'] = dct['output'][0]

    return instances


def get_train_val_portions(preprocessed_data, train_ratio = 0.8, seed = 32):
    """
    DEPRECATED: This function was written for the Natural-Instructions dataset
                since it didnt have train/test portions.
    """
    np.random.seed(seed)
    index_arr = np.arange(len(preprocessed_data))
    np.random.shuffle(index_arr)
    train_idxs = index_arr[:int(train_ratio*len(preprocessed_data))]
    val_idxs = index_arr[int(train_ratio*len(preprocessed_data)):]
    preprocessed_train_data = [preprocessed_data[i] for i in train_idxs]
    preprocessed_valid_data = [preprocessed_data[i] for i in val_idxs]

    return preprocessed_train_data, preprocessed_valid_data


def get_model_and_tokenizer(args):

    model_name = args.local_model_path if args.local_model_path else args.model_name
    if 'gpt2' in args.model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)    
        model = GPT2LMHeadModel.from_pretrained(model_name)
    
    elif 'opt' in args.model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)    
        model = OPTForCausalLM.from_pretrained(model_name)
    
    else:
        print('No such model is supported now!')

    return model, tokenizer