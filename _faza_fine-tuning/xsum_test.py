from datasets import load_dataset, load_metric
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm
import torch
import os
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUMM_RESULTS_DIR = 'dec_results/bart_xsum'

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def generate_summaries(lns, metric, batch_size=16, device=DEFAULT_DEVICE):
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-xsum") 
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum").to(device)
    
    article_batches = list(chunks(lns['document'], batch_size))
    target_batches = list(chunks(lns['summary'], batch_size))
    
    preds = []
    srcs = []
    tgts = []
    
    for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
        dct = tokenizer.batch_encode_plus(article_batch,
                                          max_length=1024,
                                          truncation=True,
                                          padding='max_length',
                                          return_tensors="pt")
        summaries = model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            num_beams=5,
            length_penalty=2.0,
            max_length=142,
            # min_len=56,
            no_repeat_ngram_size=3,
            early_stopping=True,
            decoder_start_token_id=tokenizer.eos_token_id,
        )
        
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
        
        preds.append(dec)
        srcs.append(article_batch)
        tgts.append(target_batch)
        

        metric.add_batch(predictions=dec, references=target_batch)
    # score = metric.compute()
    
    return preds, srcs, tgts

dataset = load_dataset('xsum', name='english')



rouge_metric = load_metric('rouge')
# print(dataset['test'][0:5])
preds,srcs,tgts = generate_summaries(dataset['test'], rouge_metric)

save_path = os.path.join(SUMM_RESULTS_DIR, "bart_xsum_large")
print("Saving to {}".format(save_path))
parent = os.path.abspath(os.path.join(save_path, os.pardir))
if not os.path.exists(parent):
    os.makedirs(parent)
with open(save_path, "w+", encoding='utf-8') as out_file:
    for pa in preds:
        out_file.write("".join(pa) + '\n')
        
with open(save_path + '.raw_src', "w+", encoding='utf-8') as out_file_src:
    for pa in srcs:
        out_file_src.write(str(''.join(pa.splitlines())) + '\n')

with open(save_path + '.gold', "w+", encoding='utf-8') as out_file_tgt:
    for pa in tgts:
        out_file_tgt.write(str(pa.replace('\n', ' ')) + '\n')


print("END")