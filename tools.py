import numpy as np 
import pandas as pd 
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint
import matplotlib.pyplot as plt
import seaborn as sns 
import tiktoken 
from typing import List, Optional, Dict
import torch 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os 


def generate_sycophantic_responses_df(df):
    """
    Generate the dataset for forward pass and activation storing.
    """
    df['responses'] = df.apply(
        lambda x: [x['answer_not_matching_behavior'], x['answer_matching_behavior']], axis=1
    )
    
    df_contrastive = df.explode('responses').reset_index(drop=True)
   
    df_contrastive['pair_id'] = df_contrastive.index // 2
    df_contrastive['prompt'] = df_contrastive['question'] + df_contrastive['responses']
    df_contrastive['is_syco'] = (df['responses'] == df['answer_matching_behavior']).astype(int)
    
    df_contrastive = df_contrastive.sort_values(by=['pair_id', 'is_syco'], ascending=True)
    
    return df_contrastive


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
  
    return logits, cache, model 


def retrieve_residual_stream_for_contrastive_pair(cache: ActivationCache, 
                                                  layers: List[int],
                                                  positions_to_analyze: int=-1,
                                                  decompose_residual_stream=True, 
                                                  normalize_streams=True): 
    """_summary_

   Parameters:
        - cache (ActivationCache): The cache object containing the model activations (residuals, embeddings, etc.).
        - layers (List[int]): A list of layer indices to extract from the residual stream.
        - positions_to_analyze (List[int], default=-1): The token positions to analyze within the sequence. -1 typically refers to the last token.
        - decompose_residual_stream (bool, default=True): If True, decomposes the residuals into their components per layer. If False, returns the accumulated residual stream as-is.
        - normalize_streams (bool, default=True): If True, applies layer normalization to the residual streams. optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    batch_size = cache['hook_embed'].size()[0]
    assert batch_size % 2 == 0, "Contrastive pairs must have a batch dimension of two."
    
    n_pairs = batch_size // 2
    
    if decompose_residual_stream: 
        accum_resids  = cache.decompose_resid(layer=max(layers), 
                                              apply_ln=normalize_streams)
    
    else: 
        accum_resids = cache.accumulated_resid(layer=max(layers), 
                                               apply_ln=normalize_streams)


    resids = accum_resids[:, :, positions_to_analyze, :]
    resids = resids.view(  # Reshapes into n_layers, n_pairs, 2, 768 
        resids.size(0),
        n_pairs,
        2,
        resids.size(-1),
    )
    
    pos = resids[:, :, 1, :] 
    neg = resids[:, :, 0, :] 
    
    return np.array(pos)[layers, :, :], np.array(neg)[layers, :, :]


def compute_diff_vector(pos, neg): 
    """
    Computes the difference of the matrix. 
    """
    return pos - neg

def compute_steering_vector(pos, 
                            neg, 
                            steering_coeffs, 
                            normalize:bool=False): 
    """
    Takes the activations of the positive example and negative examples and computes a steering vector. 
    """
    diff = compute_diff_vector(pos, neg)
    average_diff = np.average(diff[0], axis=0).reshape(1, -1)
    
    if normalize: 
        steer_vec = (steering_coeffs * average_diff) / np.linalg.norm(diff)
    else: 
        steer_vec = steering_coeffs * average_diff 
    return steer_vec 


def initialize_perma_hook_model(model, 
                                layer, 
                                steering_vector): 
    """
    Registers a permanent forward hook that injects the steering vector to the layer of choice. 
    """
    if isinstance(steering_vector, np.ndarray): 
        print("Converting steering vector to a tensor.")
        steering_vector = torch.Tensor(steering_vector)
        steering_vector = steering_vector.detach()
    
    def steering_hook(activations, hook: HookPoint): 
        print("Running perma hook function ")

        return activations + steering_vector
   
    model = HookedTransformer.from_pretrained(model_name=model)
    model.add_perma_hook(name=layer, hook=steering_hook)
    
    return model 


"""

Takes a matrix of 

[ # of layers, n_pairs, 2 (pos is 1 neg is 0), 768 (model dimensions)] 

For every single layer: 

    We enter the matrix, and for each of the n_pair_matrix, we extract the positive and negative layers, and concatenate it into a dataset with features as 1 if pos
    Then, we run Linear Discriminant Analysis. 
    Project the matrix onto the discriminant components 
    Save the results in a txt file 
    Save the results out in a DF 
        with the layer #, how accurate are the discrimination scores, 
        save a plot of the visualization in a distributions 

"""


def linear_discriminative_projections(cache, layer): 
    
    pos, neg = retrieve_residual_stream_for_contrastive_pair(cache=cache, 
                                                            layers=[layer], 
                                                            positions_to_analyze=-1, 
                                                            decompose_residual_stream=True,
                                                            normalize_streams=False)      
    pos_df = pd.DataFrame(pos[0])
    pos_df['is_syco'] = 1
    neg_df = pd.DataFrame(neg[0])
    neg_df['is_syco'] = 0

    df = pd.concat([pos_df, neg_df], axis=0)
    
    is_syco_vec = df.loc[:, 'is_syco']
    X = df.drop(['is_syco'], axis=1).values
    
    lda = LinearDiscriminantAnalysis("svd")
    lda.fit_transform(X=X, y=is_syco_vec)
    
    X_proj = lda.transform(X)
    y_pred = lda.predict(X=X)
    
    print("Accuracy:", accuracy_score(is_syco_vec, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(is_syco_vec, y_pred))
    print("Classification Report:\n", classification_report(is_syco_vec, y_pred))
    
    return {
        "projected": X_proj,
        "coeffs": lda.coef_,
        "norm_lda_coeffs": np.linalg.norm(lda.coef_),
        "explained_variance": lda.explained_variance_ratio_
    }
    

def compute_orthogonal_discriminative_projections(cache, layer): 
    
    ret_dct = linear_discriminative_projections(cache=cache, layer=layer)
    np

def plot_linear_discriminants(projected_data: np.ndarray, 
                              labels: np.ndarray,
                              plot_title: str,
                              label_dict: Optional[Dict[int, str]] = None,
                              alpha: float = 0.85):
    """
    Plot LDA-projected data, color by original class labels.

    Args:
        projected_data: (N, D) array from lda.transform(X)
        labels: (N,) array of class labels (e.g. 0 / 1)
        plot_title: Title of the plot
        label_dict: Optional mapping {0: "negative", 1: "positive"}
        alpha: Scatter transparency
    """
    if label_dict is None:
        label_dict = {label: str(label) for label in np.unique(labels)}

    projected_data = np.asarray(projected_data)
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)

    if projected_data.shape[1] == 1:
        x = projected_data[:, 0]

        for label in unique_labels:
            mask = labels == label
            plt.scatter(x[mask], np.zeros(mask.sum()), alpha=alpha, label=label_dict.get(label, str(label)))

            class_mean = x[mask].mean()
            plt.scatter(class_mean, 0, color="black", zorder=5)
            plt.text(class_mean, 0.02, f"μ = {class_mean:.2f}", ha="center")

        if len(unique_labels) == 2:
            mean_0 = x[labels == unique_labels[0]].mean()
            mean_1 = x[labels == unique_labels[1]].mean()
            decision_boundary = 0.5 * (mean_0 + mean_1)
            plt.axvline(decision_boundary, linestyle="--", color="black", label="Decision midpoint")

        plt.yticks([])
        plt.xlabel("Linear Discriminant 1")


    else:
        for label in unique_labels:
            mask = labels == label
            plt.scatter(projected_data[mask, 0], projected_data[mask, 1], alpha=alpha, label=label_dict.get(label, str(label)), edgecolors="none")

            centroid = projected_data[mask, :2].mean(axis=0)
            plt.scatter(centroid[0],centroid[1],color="black",zorder=5)

        plt.xlabel("Linear Discriminant 1")
        plt.ylabel("Linear Discriminant 2")

    plt.title(plot_title)
    plt.legend()
    plt.ylim(-0.4, len(unique_labels) - 1 + 0.4)
    plt.grid(False)
    plt.tight_layout()
    plt.show()



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


    