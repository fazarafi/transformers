from datasets import load_dataset, load_metric
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm
import torch
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def generate_summaries(lns, metric, batch_size=16, device=DEFAULT_DEVICE):
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn") 
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
    
    article_batches = list(chunks(lns['article'], batch_size))
    target_batches = list(chunks(lns['highlights'], batch_size))
    
    for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
        dct = tokenizer.batch_encode_plus(article_batch,
                                          max_length=1024,
                                          truncation=True,
                                          padding='max_length',
                                          return_tensors="pt")
        summaries = model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            num_beams=4,
            length_penalty=2.0,
            max_length=142,
            min_len=56,
            no_repeat_ngram_size=3,
            early_stopping=True,
            decoder_start_token_id=tokenizer.eos_token_id,
        )
        
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]

        metric.add_batch(predictions=dec, references=target_batch)
    score = metric.compute()
    return score

dataset = load_dataset("cnn_dailymail", version='3.0.0')
rouge_metric = load_metric('rouge')
print(dataset['test'][0:5])
score = generate_summaries(dataset['test'][0:100], rouge_metric)


print(score)