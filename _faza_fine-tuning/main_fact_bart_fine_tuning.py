import torch
import numpy as np
import datasets

import random


import sys
sys.path.insert(0, "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/transformers/src")

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from datetime import datetime
from tabulate import tabulate
import nltk
import os
import argparse
from math import log
# import spacy

# spacy.prefer_gpu()

from rouge import Rouge

def generate_summary(test_samples, summ_model):
    inputs = tokenizer(
        test_samples["document"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(summ_model.device)
    attention_mask = inputs.attention_mask.to(summ_model.device)
    print("BEAM SINGLE: ", summ_model.generate_custom_beam(input_ids, attention_mask=attention_mask))
    outputs = summ_model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str

def flatten(example):
    return {
        "document": example["article"],
        "summary": example["highlights"],
    }

def list2samples(example):
    documents = []
    summaries = []
    for sample in zip(example["document"], example["summary"]):
        if len(sample[0]) > 0:
            documents += sample[0]
            summaries += sample[1]
    return {"document": documents, "summary": summaries}


def list2samples(example):
    print(example["article"][:2])
    documents = []
    summaries = []
    for sample in zip(example["article"], example["highlights"]):
        if len(sample[0]) > 0:
            documents += sample[0]
            summaries += sample[1]
    return {"document": documents, "summary": summaries}


def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["document"], batch["summary"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


def beam_search_expand_single_bart(summ_model, summ_paths, beam_size, input_params, **model_kwargs):
    updated_summ_paths, return_params, model_kwargs = summ_model.beam_search_expand_single(
        input_params,
        summ_paths,
        logits_processor=input_params["logits_processor"],
        stopping_criteria=input_params["stopping_criteria"],
        pad_token_id=input_params["pad_token_id"],
        eos_token_id=input_params["eos_token_id"],
        output_scores=input_params["output_scores"],
        return_dict_in_generate=input_params["return_dict_in_generate"],
        synced_gpus=input_params["synced_gpus"],
        **model_kwargs,
    )

    return updated_summ_paths, return_params, model_kwargs

def finalize_beam_search_expand_single_bart(summ_model, summ_paths, params):
    # print("input_ids: ", params['input_ids'])
    # print("summ_paths: ", summ_paths)
    
    inp_ids = []
    for _, path, _ in summ_paths:
        inp_ids.append(path.unsqueeze(0))
    input_path = torch.cat(inp_ids, dim=0)
    # print(input_path)

    params["input_ids"] = input_path
    result = summ_model.finalize_beam_search_expand_single(
        summ_paths, 
        params,
        params["beam_scorer"],
        params["input_ids"],
        params["beam_scores"],
        params["next_tokens"],
        params["next_indices"],
        params["pad_token_id"],
        params["eos_token_id"],
        params["stopping_criteria"],
        params["beam_indices"],
        params["return_dict_in_generate"],
        params["output_scores"]
    )

    return result


parser = argparse.ArgumentParser()
parser.add_argument('-visible_gpus', default='0', type=str)
parser.add_argument('-gpu_ranks', default='0', type=str)

args = parser.parse_args()

# Setup GPU for PreSumm
args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
args.world_size = len(args.gpu_ranks)
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

device = "cpu" if args.visible_gpus == '-1' else "cuda"
device_id = 1 if device == "cuda" else -1



# model_name_2 = "sshleifer/distilbart-xsum-12-3"

print("dev ",device)

# tokenization
encoder_max_length = 256  # demo
decoder_max_length = 64
dataset_name = 'xsum'
model_name = "facebook/bart-large-xsum"

if (dataset_name=='xsum'):
    model_name = "facebook/bart-large-xsum"
    
    language = 'english'
    train_data = datasets.load_dataset(dataset_name, name=language, split="train[:2000]")
    test_data = datasets.load_dataset(dataset_name, name=language, split="test")
    valid_data = datasets.load_dataset(dataset_name, name=language, split="validation")

    train_dataset = train_data.map(remove_columns=["id"])
    test_dataset = test_data.map(remove_columns=["id"])
    valid_dataset = valid_data.map(remove_columns=["id"])

    # print(train_dataset[:3])

    train_data_txt = train_dataset
    test_data_txt = test_dataset
    validation_data_txt = valid_dataset


elif (dataset_name=='cnn_dailymail'):
    model_name = "facebook/bart-large-cnn"
    
    language = '3.0.0'
    train_data = datasets.load_dataset(dataset_name, name=language, split="train[:2000]")
    test_data = datasets.load_dataset(dataset_name, name=language, split="test")
    valid_data = datasets.load_dataset(dataset_name, name=language, split="validation")

    train_dataset = train_data.map(flatten, remove_columns=["highlights","article", "id"])
    test_dataset = test_data.map(flatten, remove_columns=["highlights","article", "id"])
    valid_dataset = valid_data.map(flatten, remove_columns=["highlights","article", "id"])


    # print(train_dataset[:3])
    # print(train_dataset[:3]["summary"])

    train_data_txt = train_dataset
    test_data_txt = test_dataset
    validation_data_txt = valid_dataset

summ_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set summ_model parameters or use the default
print("torch ", torch.cuda.is_available())

summ_model.to(device)

train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)

validation_data = validation_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=validation_data_txt.column_names,
)

# Borrowed from https://github.com/huggingface/transformers/blob/master/examples/seq2seq/run_summarization.py
# nltk.download("punkt", quiet=True)

rouge = Rouge()

test_samples = validation_data_txt.select(range(50))
summaries = test_samples["summary"]

# print("len::: ", len(train_samples))

# for dat in train_data[:3]["input_ids"]:
#     print("dat")
#     print(dat)


inputs = tokenizer(
    test_samples["document"],
    padding="max_length",
    truncation=True,
    max_length=encoder_max_length,
    return_tensors="pt",
)

print("dev summ_model 2: ", summ_model.device)
input_ids = inputs.input_ids.to(summ_model.device)
attention_mask = inputs.attention_mask.to(summ_model.device)
# result = summ_model.generate_beam_expansion(input_ids, attention_mask=attention_mask)
num_beams = 4
beam_size = num_beams
result, results_ori = summ_model.generate(input_ids, num_beams=num_beams, attention_mask=attention_mask)
# print("RESULT: ")
result_str = []
for res in result:
    res_str = tokenizer.decode(res, skip_special_tokens=True)
    result_str.append(res_str)

# print("RESULT 2")


ori_seqs = [[] for _ in range(num_beams)]

print("========")
print("SEQ OUT 1 BSFT ori_seqs:")
print("---")
for i, result_ori in enumerate(results_ori):
    ori_beam = []
    for bm in range(num_beams):
        text_dec = tokenizer.decode(result_ori[bm][1], skip_special_tokens=True)
        ori_seqs[bm].append(text_dec)
        print(text_dec)
    print("---")


print("========")
# scores = rouge.get_scores(result_str, summaries, avg=True)
# print("BEST Rouge scores: ", scores)

# print("ORIGINAL SEQS =>")
# print(ori_seqs)
for i, ori_seq in enumerate(ori_seqs):
    scores = rouge.get_scores(ori_seq, summaries, avg=True)
    print(i, ". Rouge: ", scores)



inputs = tokenizer(
    test_samples["document"],
    padding="max_length",
    truncation=True,
    max_length=encoder_max_length,
    return_tensors="pt",
)

print("dev summ_model 2: ", summ_model.device)
input_ids = inputs.input_ids.to(summ_model.device)
attention_mask = inputs.attention_mask.to(summ_model.device)
# result = summ_model.generate_beam_expansion(input_ids, attention_mask=attention_mask)

### TEST WITH SINGLE BEAM EXPANSION
max_pred_len = 150
src_input_ids = input_ids
params, model_kwargs = summ_model.generate_beam_expansion(
    src_input_ids, num_beams=beam_size, attention_mask=attention_mask
)

summ_paths = []

# for input_id in zip(params["input_ids"]:
#     summ_paths.append((
#         log(1.0),
#         torch.tensor(
#             [input_id],
#             dtype=torch.long,
#             device=device
#         ),
#         0
#     ))

for input_id, beam_score in zip(params["input_ids"], params['beam_scores']):
    summ_paths.append((
        log(1.0),
        torch.tensor(
            [input_id],
            dtype=torch.long,
            device=device
        ),
        beam_score
    ))

batch_size = params['batch_size']

for step in range(max_pred_len):
    new_all_paths, params, model_kwargs = \
            beam_search_expand_single_bart(summ_model, summ_paths, beam_size, params, **model_kwargs)

    if (params['this_peer_finished']):
        
        # TODO remove below
        total_summ_paths = []
        for idx in range(batch_size):
            new_paths = new_all_paths[idx*beam_size : (idx+1)*beam_size]
            # summ_paths = new_paths 
            # print("before PATH: ", new_paths[0])
            random.shuffle(new_paths)
            new_paths = sorted(new_paths, reverse=True, key=lambda x: x[2])
            # print("after PATH: ", new_paths[0])
            
            summ_paths = new_paths
            
            summ_paths = summ_paths[:beam_size]
            total_summ_paths = total_summ_paths + summ_paths
        summ_paths = total_summ_paths
        # TODO remove above

        sequence_output = finalize_beam_search_expand_single_bart(summ_model, summ_paths, params)
        # print("input_ids: ", )
        # print("summ_paths")

        print("****************")
        print("SEQ OUT 2 BEAM EXPFT: ")
        print("---")
        last_path = []
        for ori_seq in sequence_output["original_sequences"]:
            for ori in ori_seq:
                last_path.append((ori[0], ori[1], ori[2]))
                print(tokenizer.decode(ori[1], skip_special_tokens=True) )
            print("---")
        print("****************")
        
        new_all_paths = last_path
        summ_paths = new_all_paths
        break
    
    total_summ_paths = []
    for idx in range(batch_size):
        new_paths = new_all_paths[idx*beam_size : (idx+1)*beam_size]
        # summ_paths = new_paths 
        # print("before PATH: ", new_paths[0])
        
        if (random.randint(0,9) % 3):
            random.shuffle(new_paths)
        # new_paths = sorted(new_paths, reverse=True, key=lambda x: x[2])
        # print("after PATH: ", new_paths[0])
        
        summ_paths = new_paths
        
        summ_paths = summ_paths[:beam_size]
        total_summ_paths = total_summ_paths + summ_paths
    
    summ_paths = total_summ_paths
    # TODO use above
    # summ_paths = new_all_paths

# summ_result = {}
# for idx in range(beam_size):
#     summ_result[idx] = []
#     for idy in range(batch_size):
#         id = idx * beam_size + idy
#         path_str = tokenizer.decode(summ_paths[id][1], skip_special_tokens=True)
#         summ_result[idx].append(path_str)

summ_result = [[] for _ in range(num_beams)]
for i, (lp, summ_path, beam_score) in enumerate(summ_paths):
    ori_beam = []
    text_dec = tokenizer.decode(summ_path, skip_special_tokens=True)
    summ_result[i % num_beams].append(text_dec)
    

for i in range(num_beams):
    # print("#####")
    # print("---- GOLDEN SUMM:")
    # print(summaries)
    # print("---- SYSTEM SUMM:")
    # print(summ_result[i])
    scores = rouge.get_scores(summ_result[i], summaries, avg=True)
    print("Yg ke -", i, ": ", scores)
print("SORT random then reverse AWS")