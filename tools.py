import numpy as np 
import pandas as pd 
from transformer_lens import HookedTransformer, ActivationCache
import matplotlib.pyplot as plt
import seaborn as sns 
import tiktoken 
from typing import List




def generate_sycophantic_responses_df(df):
    df['responses'] = df.apply(
        lambda x: [x['answer_not_matching_behavior'], x['answer_matching_behavior']], axis=1
    )
    df_contrastive = df.explode('responses').reset_index(drop=True)
    df_contrastive['pair_id'] = df_contrastive.index // 2
    df_contrastive['prompt'] = df_contrastive['question'] + df_contrastive['responses']
    df_contrastive['is_syco'] = (df['responses'] == df['answer_matching_behavior']).astype(int)
    return df_contrastive


class ContrastiveActivator(): 
    pass 

class DiscriminativeActivator(): 
    pass 

def extract_activations_from_prompts(df, 
                                     n_pairs, 
                                     model="gpt2-small"):
    """
    Extract activations from prompts in a pairwise manner.

    Args:
        df: DataFrame with columns ['pair_id', 'prompt', 'response', 'is_syco'].
        n_pairs: Number of pairs to sample.
        model: Model name for HookedTransformer.
    
    Returns:
        logits, cache
    """
    df = df.copy()
    
    unique_pairs = df['pair_id'].unique()
    sampled_pairs = pd.Series(unique_pairs).sample(n=n_pairs, random_state=18).tolist()
    
    df_sample = df[df['pair_id'].isin(sampled_pairs)]
    df_sample = df_sample.sort_values('pair_id').reset_index(drop=True)
    
    prompts = df_sample['prompt'].to_list()
    
    model = HookedTransformer.from_pretrained(model_name=model)
    
    logits, cache = model.run_with_cache(prompts)
    
  
    return logits, cache


def retrieve_residual_stream_for_contrastive_pair(cache: ActivationCache, 
                            positions_to_analyze: List[int],
                            layers: int,
                            decompose_residual_stream=True,
                            normalize_streams=True
                            ): 
    assert cache['hook_embed'].size()[0] == 2, "Contrastive pairs must have a batch dimension of two."
    
    if decompose_residual_stream: 
        accum_resids  = cache.decompose_resid(layers=layers, 
                                     return_labels=True, 
                                     apply_ln=normalize_streams)
    
    else: 
        accum_resids = cache.accumulated_resid(layer=layers, 
                                               return_labels=True, 
                                               apply_ln=normalize_streams)
    
    token_resids = {}
    
    for position in positions_to_analyze: 
        token_resids["positive"] = accum_resids[:, 0, position, :] # [layer, 0th batch, pos, dimensions-of_model]
        token_resids["negative"] = accum_resids[:, 1, position, :] 
    
    return token_resids



def visualize_attention_internals(cache: ActivationCache, nth_layers: int, n_heads: int, kind:str="pattern", cmap:str=None):
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


"""
TO DO: 

Look at which sequence length is the last token? 

"""

def evaluate_sequence_length_of_prompt(prompt, model="gpt2"): 
    """
    Takes a prompt and model name and returns a list of tuples of the order and token position
    
    e.g., 
        (0, b'Hello')
        (1, b' world')
        (2, b'! ')  
    """
    encoder = tiktoken.get_encoding(model)
    bytes_list = encoder.encode(prompt)
    
    decoded_bytes = encoder.decode_tokens_bytes(bytes_list)
    length_of_bytes = range(len(decoded_bytes))
    
    return list(zip(length_of_bytes, decoded_bytes))


    