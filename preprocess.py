
from datasets import load_dataset
from transformer_lens import HookedTransformer
import pandas as pd 
import numpy as np 
import torch

### Downloading the dataset pi
ds = load_dataset("Anthropic/model-written-evals")
df = ds['train'].to_pandas()

np.random.seed(187)
torch.manual_seed(187)
df.columns

## Create a collection of sycophantic responses and non-sycophantic responses 
def generate_sycophantic_responses_df(df): 
    
    df = df.copy()
    
    a = np.random.rand(len(df))
    df['is_syco'] = np.where(a < 0.5, 0, 1)

    df['q'] = df.apply(lambda x: x['question'] + x['answer_matching_behavior'] if x['is_syco'] > 1 
                       else x['question'] + x['answer_not_matching_behavior'], axis=1)
    return df 
## 

def extract_activations_from_prompts(df, n,  model="gpt2-small"): 
    df = df.copy()
    df_shuffled = df.sample(frac=1, random_state=18).reset_index(drop=True)
    df_sample = df_shuffled.iloc[0:n-1]
    
    samples = df_sample['q'].to_list()
    
    model = HookedTransformer.from_pretrained(model_name=model)
    _, cache = model.run_with_cache(samples)
    return cache



"""Hypothesis Generation: 
I first plot averaged saliency scores of the neural networks he goal is to  identify a transformeer layer that allows us to see where the focus is on the last few tokens. The hypothesis being that the attention layers
that focuses on the last few tokens also correspond to the layers that will be most informative on sycophancy 
"""
def visualize_saliency_score(): 
    """
    Takes the cache of the forward pass computation and visualizes saliency maps on the mean transformer attention scores. 
    Returns an *n* plots visualization of the saiency map of the averaged transformer attention scores and a sorted list of the 
    layers with highest attentions scores for the last two tokens. 
    """
    pass



df_train = generate_sycophantic_responses_df(df)

cache = extract_activations_from_prompts(df_train, n=2)

cache.keys()

cache['hook_embed']

for key, value in cache.items(): 
    print(key)
    print('------')
    print(value)

df_train['is_syco'].value_counts()