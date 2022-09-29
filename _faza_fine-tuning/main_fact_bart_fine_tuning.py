import torch
import numpy as np
import datasets
import datasets

import sys
sys.path.insert(0, "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/transformers/src")

# >>> import sys                                                                                                                                                                                                                             
# >>> sys.path.insert(0, "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/transformers/src")                                                                                                                                            
# >>> from transformers import activations        

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from tabulate import tabulate
import nltk
from datetime import datetime

WANDB_INTEGRATION = True
if WANDB_INTEGRATION:
    import wandb

    wandb.login()


language = "english"

model_name = "sshleifer/distilbart-xsum-12-3"
# if language == "french":
#     model_name = "moussaKam/barthez-orangesum-abstract"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set model parameters or use the default
# print(model.config)

# tokenization
encoder_max_length = 256  # demo
decoder_max_length = 64

# data = datasets.load_dataset("wiki_lingua", name=language, split="train[:2000]")
train_data = datasets.load_dataset("xsum", name=language, split="train[:2000]")
test_data = datasets.load_dataset("xsum", name=language, split="test")
valid_data = datasets.load_dataset("xsum", name=language, split="validation")


# Take a look at the adata

train_dataset = train_data.map(remove_columns=["id"])
test_dataset = test_data.map(remove_columns=["id"])
valid_dataset = valid_data.map(remove_columns=["id"])

train_data_txt = train_dataset
test_data_txt = test_dataset
validation_data_txt = valid_dataset


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
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


model_before_tuning = AutoModelForSeq2SeqLM.from_pretrained(model_name)

test_samples = validation_data_txt.select(range(16))

summaries_before_tuning = generate_summary(test_samples, model_before_tuning)[1]
summaries_after_tuning = generate_summary(test_samples, model)[1]

print("BEFORE")
print(str(summaries_before_tuning))
print("AFTER")
print(str(summaries_after_tuning))
print("DONE")

