import numpy as np 
import pandas as pd 
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import seaborn as sns 

def generate_sycophantic_responses_df(df): 
    df_syco_0 = df.copy()
    df_syco_0['is_syco'] = 0
    df_syco_1 = df.copy()
    df_syco_1['is_syco'] = 1

    df_contrastive = pd.concat([df_syco_0, df_syco_1], ignore_index=True)

    df_contrastive['q'] = df_contrastive.apply(
        lambda x: x['question'] + x['answer_matching_behavior'] if x['is_syco'] == 1 else x['question'] + x['answer_not_matching_behavior'], axis=1)
    
    df_contrastive['index'] = np.repeat(df.index.values, 2)
    return df_contrastive.sort_values(by='index', ascending=True)


def extract_activations_from_prompts(df, n, model="gpt2-small"): 
    df = df.copy()
    df_shuffled = df.sample(frac=1, random_state=18).reset_index(drop=True)
    df_sample = df_shuffled.iloc[0:n]
    
    samples = df_sample['q'].to_list()
    
    model = HookedTransformer.from_pretrained(model_name=model)
    logits, cache = model.run_with_cache(samples)
    return logits, cache


def visualize_attention_internals(cache, nth_layers, n_heads, kind="pattern", cmap=None):
    """
    Visualize attention for one example across specified layers and heads.
    
    Args:
        cache: TransformerLens ActivationCache
        nth_layers: list of int, layers to visualize
        n_heads: int, number of attention heads
        kind: "pattern" (post softmax) or "scores" (raw attention logits)
        cmap: optional colormap for visualization
    """
    
    kind_dict = {
        "logits": "attn_scores", 
        "pattern": "pattern", 
        "pre-softmax": "attn_scores", 
        "post-softmax": "pattern"
    }
    kind = kind_dict.get(kind)

    if cmap is None:
        cmap = "Blues" if kind == "pattern" else "Blues_r"

    fig, axes = plt.subplots(len(nth_layers), n_heads, figsize=(2.2 * n_heads, 2.2 * len(nth_layers)), squeeze=False)
    
    for i, layer in enumerate(nth_layers):
        key = f"blocks.{layer}.attn.hook_{kind}"
        attn = cache[key][0] 

        for head in range(n_heads):
            ax = axes[i, head]
            mat = attn[head].detach().cpu().numpy()

            sns.heatmap(mat, cmap=cmap, ax=ax, cbar=False, center=0 if kind == "scores" else None)

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
    return fig