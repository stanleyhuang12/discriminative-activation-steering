import numpy as np 
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import seaborn as sns 

def generate_sycophantic_responses_df(df): 
    
    df = df.copy()
    
    a = np.random.rand(len(df))
    df['is_syco'] = np.where(a < 0.5, 0, 1)

    df['q'] = df.apply(lambda x: x['question'] + x['answer_matching_behavior'] if x['is_syco'] > 1 
                       else x['question'] + x['answer_not_matching_behavior'], axis=1)
    return df 


def extract_activations_from_prompts(df, n, model="gpt2-small"): 
    df = df.copy()
    df_shuffled = df.sample(frac=1, random_state=18).reset_index(drop=True)
    df_sample = df_shuffled.iloc[0:n]
    
    samples = df_sample['q'].to_list()
    
    model = HookedTransformer.from_pretrained(model_name=model)
    _, cache = model.run_with_cache(samples)
    return cache


def visualize_attention_scores(cache, nth_layers, n_heads, kind="pattern", cmap=None):
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

    fig, axes = plt.subplots(len(nth_layers), n_heads, figsize=(2.2 * n_heads, 2.2 * len(nth_layers)), squeeze=False)
    
    for i, layer in enumerate(nth_layers):
        key = f"blocks.{layer}.attn.hook_{kind}"
        attn = cache[key][0] 

        for head in range(n_heads):
            ax = axes[i, head]
            mat = attn[head].detach().cpu().numpy()

            sns.heatmap(mat[::-1], cmap=cmap, ax=ax, cbar=False, center=0 if kind == "scores" else None)

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