from datasets import load_dataset
from transformer_lens import HookedTransformer
import pandas as pd 
import numpy as np 
import torch
import matplotlib.pyplot as plt


ds = load_dataset("Anthropic/model-written-evals")
df = ds['train'].to_pandas()

np.random.seed(187)
torch.manual_seed(187)
df.columns

def generate_sycophantic_responses_df(df): 
    
    df = df.copy()
    
    a = np.random.rand(len(df))
    df['is_syco'] = np.where(a < 0.5, 0, 1)

    df['q'] = df.apply(lambda x: x['question'] + x['answer_matching_behavior'] if x['is_syco'] > 1 
                       else x['question'] + x['answer_not_matching_behavior'], axis=1)
    return df 

def extract_activations_from_prompts(df, n,  model="gpt2-small"): 
    df = df.copy()
    df_shuffled = df.sample(frac=1, random_state=18).reset_index(drop=True)
    df_sample = df_shuffled.iloc[0:n]
    
    samples = df_sample['q'].to_list()
    
    model = HookedTransformer.from_pretrained(model_name=model)
    _, cache = model.run_with_cache(samples)
    return cache

"""Hypothesis Generation: 
I first plot averaged saliency scores of the neural networks he goal is to identify a transformer layer that allows us to see where the focus is on the last few tokens. The hypothesis being that the attention layers
that focuses on the last few tokens also correspond to the layers that will be most informative on sycophancy 
"""
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
def visualize_attention_scores(
    cache,
    nth_layers,
    n_heads,
    kind="pattern",
    cmap=None
    ):
    """
    Visualize attention for one example across specified layers and heads.
    
    Args:
        cache: TransformerLens ActivationCache
        nth_layers: list of int, layers to visualize
        n_heads: int, number of attention heads
        kind: "pattern" (softmaxed attention) or "scores" (raw logits)
        cmap: optional colormap for visualization
    """

    if cmap is None:
        cmap = "RdBU_r" if kind == "pattern" else "RdBu_r"

    fig, axes = plt.subplots(
        len(nth_layers),
        n_heads,
        figsize=(2.2 * n_heads, 2.2 * len(nth_layers)),
        squeeze=False
    )

    for i, layer in enumerate(nth_layers):
        key = f"blocks.{layer}.attn.hook_{kind}"
        attn = cache[key][0] 

        for head in range(n_heads):
            ax = axes[i, head]
            mat = attn[head].detach().cpu().numpy()

            sns.heatmap(
                mat[::-1],
                cmap=cmap,
                ax=ax,
                cbar=False,
                center=0 if kind == "scores" else None
            )

            # Titles and labels on first row and col only
            if i == 0:
                ax.set_title(f"H{head}", fontsize=10)
            if head == 0:
                ax.set_ylabel(f"L{layer}", fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(
        f"Attention {'Patterns' if kind == 'pattern' else 'Scores'} (One Example)",
        y=1.02
    )
    plt.tight_layout()
    # You could optionally return fig here to save the figure from outside
    return fig


df_train = generate_sycophantic_responses_df(df)
cache = extract_activations_from_prompts(df_train, n=1)

for l in range(0, 12):
    fig = visualize_attention_scores(cache=cache, nth_layers=[l], n_heads=12, kind="pattern")
    fig.savefig(f'visualizations/attention_layer_{l}_heads_12_attn_scores.png')
    plt.close(fig)  # free memory
    

import os
os.makedirs('visualizations')
plt.save('visualizations/attention_12_layers_12_heads_heatmaps')



cache = extract_activations_from_prompts(df=df_train, n=3)

cache.keys()

fig, ax = plt.subplots()
sns.heatmap(cache['blocks.0.attn.hook_pattern'][0][0, :].detach().cpu().tolist(), cmap='cividis', ax=ax)

plt.xlabel("Key tokens")
plt.ylabel("Query tokens")
plt.show()

cache['blocks.0.attn.hook_pattern'][0].shape

plt.show()

cache['hook_embed']

visualize_saliency_score(cache=cache)


cache['blocks.0.attn.hook_attn_scores'].size()


for key, value in cache.items(): 
    print(key)
    print('------')
    print(value)

df_train['is_syco'].value_counts()