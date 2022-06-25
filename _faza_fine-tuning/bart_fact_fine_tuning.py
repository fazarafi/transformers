# Code source: https://github.com/francoisstamant/Fine-tuning-for-text-summarization/blob/main/text_summarization.ipynb

!pip install ohmeow-blurr -q
!pip install bert-score -q

import pandas as pd
from fastai.text.all import *
from transformers import *
from blurr.data.all import *
from blurr.modeling.all import *

'''
Preparation
'''

#Get data
df = pd.read_csv('articles.csv', error_bad_lines=False, sep=';')
df = df.dropna().reset_index()

#Select part of data we want to keep
df = df[(df['language']=='english') & (df['type']=='bs')].reset_index()
df = df[['title','text']]

#Clean text
df['text'] = df['text'].apply(lambda x: x.replace('\n',''))

#Select only part of it (makes testing faster)
articles = df.head(100)
articles.head()

'''
Import Pretrained model
'''

#Import the pretrained model
pretrained_model_name = "facebook/bart-large-cnn"
hf_arch, hf_config, hf_tokenizer, hf_model = BLURR.get_hf_objects(pretrained_model_name, 
                                                                  model_cls=BartForConditionalGeneration)

#Create mini-batch and define parameters
hf_batch_tfm = HF_Seq2SeqBeforeBatchTransform(hf_arch, hf_config, hf_tokenizer, hf_model, 
    task='summarization',
    text_gen_kwargs=
 {'max_length': 248,'min_length': 56,'do_sample': False, 'early_stopping': True, 'num_beams': 4, 'temperature': 1.0, 
  'top_k': 50, 'top_p': 1.0, 'repetition_penalty': 1.0, 'bad_words_ids': None, 'bos_token_id': 0, 'pad_token_id': 1,
 'eos_token_id': 2, 'length_penalty': 2.0, 'no_repeat_ngram_size': 3, 'encoder_no_repeat_ngram_size': 0,
 'num_return_sequences': 1, 'decoder_start_token_id': 2, 'use_cache': True, 'num_beam_groups': 1,
 'diversity_penalty': 0.0, 'output_attentions': False, 'output_hidden_states': False, 'output_scores': False,
 'return_dict_in_generate': False, 'forced_bos_token_id': 0, 'forced_eos_token_id': 2, 'remove_invalid_values': False})


#Prepare data for training
blocks = (HF_Seq2SeqBlock(before_batch_tfm=hf_batch_tfm), noop)
dblock = DataBlock(blocks=blocks, get_x=ColReader('text'), get_y=ColReader('title'), splitter=RandomSplitter())
dls = dblock.dataloaders(articles, batch_size = 2)

'''
TRAINING
'''

#Define performance metrics
seq2seq_metrics = {
        'rouge': {
            'compute_kwargs': { 'rouge_types': ["rouge1", "rouge2", "rougeL"], 'use_stemmer': True },
            'returns': ["rouge1", "rouge2", "rougeL"]
        },
        'bertscore': {
            'compute_kwargs': { 'lang': 'fr' },
            'returns': ["precision", "recall", "f1"]}}

#Model
model = HF_BaseModelWrapper(hf_model)
learn_cbs = [HF_BaseModelCallback]
fit_cbs = [HF_Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]

#Specify training
learn = Learner(dls, model,
                opt_func=ranger,loss_func=CrossEntropyLossFlat(),
                cbs=learn_cbs,splitter=partial(seq2seq_splitter, arch=hf_arch)).to_fp16()

#Create optimizer with default hyper-parameters
learn.create_opt() 
learn.freeze()

#Training
learn.fit_one_cycle(3, lr_max=3e-5, cbs=fit_cbs)


'''
PREDICTION
'''


outputs = learn.blurr_generate(text_to_generate, early_stopping=False, num_return_sequences=1)

for idx, o in enumerate(outputs):
    print(f'=== Prediction {idx+1} ===\n{o}\n')