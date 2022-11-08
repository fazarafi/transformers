import torch
import numpy as np
import datasets
import datasets

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
# import spacy

# spacy.prefer_gpu()

def generate_summary(test_samples, model):
    inputs = tokenizer(
        test_samples["document"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    print("BEAM SINGLE: ", model.generate_custom_beam(input_ids, attention_mask=attention_mask))
    outputs = model.generate(input_ids, attention_mask=attention_mask)
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





parser = argparse.ArgumentParser()
parser.add_argument('-visible_gpus', default='0', type=str)
parser.add_argument('-gpu_ranks', default='0', type=str)

args = parser.parse_args()

# Setup GPU for PreSumm
args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
args.world_size = len(args.gpu_ranks)
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

device = "cpu" if args.visible_gpus == '-1' else "cuda"
device_id = 0 if device == "cuda" else -1

    



model_name = "facebook/bart-large-cnn"
# model_name = "facebook/bart-large-xsum"
# model_name_2 = "sshleifer/distilbart-xsum-12-3"

print("dev ",device)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set model parameters or use the default
print("torch ", torch.cuda.is_available())

model.to(device)

# tokenization
encoder_max_length = 256  # demo
decoder_max_length = 64
dataset_name = 'cnn_dailymail'

# print(train_data[:3])

# Take a look at the adata

if (dataset_name=='xsum'):
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

'''
Evaluation, GenerationExample
'''

model_before_tuning = AutoModelForSeq2SeqLM.from_pretrained(model_name)

test_samples = validation_data_txt.select(range(100))

train_samples = train_data_txt.select(range(100))

# print("len::: ", len(train_samples))

# for dat in train_data[:3]["input_ids"]:
#     print("dat")
#     print(dat)


# print("input: ", test_samples["document"][:1])
inputs = tokenizer(
    test_samples["document"],
    padding="max_length",
    truncation=True,
    max_length=encoder_max_length,
    return_tensors="pt",
)


print("dev model 2: ", model.device)
input_ids = inputs.input_ids.to(model.device)
# print(len(input_ids), "  ", input_ids)
attention_mask = inputs.attention_mask.to(model.device)
# result = model.generate_beam_expansion(input_ids, attention_mask=attention_mask)
result = model.generate(input_ids, attention_mask=attention_mask)
print("RESULT: ")
for res in result:
    print(tokenizer.decode(res, skip_special_tokens=True))

print("^ RESULT")












# summaries_before_tuning = generate_summary(test_samples, model_before_tuning)[1]
# summaries_after_tuning = generate_summary(test_samples, model)[1]

# print("BEFORE")
# print(str(summaries_before_tuning))
# print("AFTER")
# print(str(summaries_after_tuning))
# print("DONE")