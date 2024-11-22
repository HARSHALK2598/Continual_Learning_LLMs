import os
import json
from utils import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoConfig

"""
Dataloader functions for preprocessing and loading the datasets
in the CL Benchmark mentioned in the O-LoRA paper.
"""


def adjust_dictionary_keys_from_the_list(dataset, instruction):

    for dct in dataset:
        dct['instruction'] = instruction
        dct['output'] = dct['label']
        dct['input'] = dct['sentence']
        dct.pop('label')
        dct.pop('sentence')


def get_dbpedia_data(data_dir = 'CL_Benchmark/TC/dbpedia'):

    with open(os.path.join(data_dir, 'labels.json')) as f:
        options = json.load(f)

    options = str(options).replace('\'', '')
    instruction = "What is the topic of the following input text? Choose one from the options.\nOptions: " + options

    with open(os.path.join(data_dir, 'train.json')) as f:
        trainset = json.load(f)
    with open(os.path.join(data_dir, 'test.json')) as f:
        testset = json.load(f)

    adjust_dictionary_keys_from_the_list(trainset, instruction)
    adjust_dictionary_keys_from_the_list(testset, instruction)

    return trainset, testset


def get_amazon_data(data_dir = 'CL_Benchmark/SC/amazon'):

    with open(os.path.join(data_dir, 'labels.json')) as f:
        options = json.load(f)

    options = str(options).replace('\'', '')
    instruction = "What is the sentiment of the following input text? Choose one from the options.\nOptions: " + options

    with open(os.path.join(data_dir, 'train.json')) as f:
        trainset = json.load(f)
    with open(os.path.join(data_dir, 'test.json')) as f:
        testset = json.load(f)

    adjust_dictionary_keys_from_the_list(trainset, instruction)
    adjust_dictionary_keys_from_the_list(testset, instruction)

    return trainset, testset


def get_yahoo_data(data_dir = 'CL_Benchmark/TC/yahoo'):

    with open(os.path.join(data_dir, 'labels.json')) as f:
        options = json.load(f)

    options = str(options).replace('\'', '')
    instruction = "What is the topic of the following input text? Choose one from the options.\nOptions: " + options

    with open(os.path.join(data_dir, 'train.json')) as f:
        trainset = json.load(f)
    with open(os.path.join(data_dir, 'test.json')) as f:
        testset = json.load(f)

    adjust_dictionary_keys_from_the_list(trainset, instruction)
    adjust_dictionary_keys_from_the_list(testset, instruction)

    return trainset, testset

def get_agnews_data(data_dir = 'CL_Benchmark/TC/agnews'):

    with open(os.path.join(data_dir, 'labels.json')) as f:
        options = json.load(f)

    options = str(options).replace('\'', '')
    instruction = "What is the topic of the following input text? Choose one from the options.\nOptions: " + options

    with open(os.path.join(data_dir, 'train.json')) as f:
        trainset = json.load(f)
    with open(os.path.join(data_dir, 'test.json')) as f:
        testset = json.load(f)

    adjust_dictionary_keys_from_the_list(trainset, instruction)
    adjust_dictionary_keys_from_the_list(testset, instruction)

    return trainset, testset


GET_DATASET = {
    'dbpedia': get_dbpedia_data,
    'amazon': get_amazon_data,
    'yahoo': get_yahoo_data,
    'agnews': get_agnews_data,
}


# if __name__ == '__main__':

#     model_name = "openai-community/gpt2"
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)

#     trainset, testset = GET_DATASET['agnews']()
#     preprocessed_data = preprocess_dataset_list(trainset, tokenizer)
#     # preprocessed_train_data, preprocessed_valid_data = get_train_val_portions(preprocessed_data, train_ratio=0.8)

#     sample = preprocessed_data[1]

#     print(tokenizer.decode(sample['input_ids']))
#     print("==================================================")
#     a = sample['labels']
#     a = [tokenizer.eos_token_id if el == -100 else el for el in a]
#     print(tokenizer.decode(a, skip_special_tokens = True))
#     print("==================================================")
#     print(tokenizer.decode(sample['generation_text'], skip_special_tokens = True))
