import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt 

def extract_activations(
    model, 
    baseline_prompt, 
    inject_noise=True,
    pos_to_inject=-1,
    noise_mean=0.0,
    noise_variance=0.5, 
    random_seed=12,
    save_logits=True
): 
    torch.manual_seed(random_seed)

    activation_outputs = []
    mlp_hiddens = []
    mlp_acts = []

    activation_outputs_inj = []
    mlp_hiddens_inj = []
    mlp_acts_inj = []
    
    with model.trace() as tracer: 
        
        with tracer.invoke(baseline_prompt) as invoker_baseline:
            for l in model.transformer.h: 
                activations = l.attn.output[0]
                mlp_hidden = l.mlp.c_fc.output[0]
                mlp_gelu = l.mlp.act.output[0]
                activation_outputs.append(activations.save())
                mlp_hiddens.append(mlp_hidden.save())
                mlp_acts.append(mlp_gelu.save())
            
            if save_logits:        
                logits = model.lm_head.output[0].save()
        
        with tracer.invoke(baseline_prompt) as invoker_injected:
            for l in model.transformer.h:
                activations = l.attn.output[0]
                if inject_noise:
                    noise = torch.normal(noise_mean, noise_variance, size=activations[:, pos_to_inject].size())
                    activations[:, pos_to_inject] += noise
                
                mlp_hidden = l.mlp.c_fc.output[0]
                mlp_gelu = l.mlp.act.output[0]

                activation_outputs_inj.append(activations.save())
                mlp_hiddens_inj.append(mlp_hidden.save())
                mlp_acts_inj.append(mlp_gelu.save()) 
            
            if save_logits: 
                logits_inj = model.lm_head.output[0].save()
                
    return {
        "act_out": activation_outputs,
        "act_out_inj": activation_outputs_inj,
        "mlp_hid": mlp_hiddens,
        "mlp_hid_inj": mlp_hiddens_inj,
        "mlp_acts": mlp_acts,
        "mlp_acts_inj": mlp_acts_inj
    }, logits, logits_inj




def compute_layerwise_cosine_similarity(activations_dict, token_pos=-1):
    """
    Compute cosine similarity for all original/injected activation pairs in a dictionary.

    For each key in activations_dict that does NOT end with '_inj', it finds the corresponding
    injected key (key + '_inj') and computes layerwise cosine similarity.

    Returns:
        dict: keys are original keys with '_corr' appended, values are lists of cosine similarities
    """
    correlations = {}

    for key in activations_dict:
        if key.endswith('_inj'):
            continue
        
        inj_key = key + '_inj'
        if inj_key not in activations_dict:
            continue

        sims = []
        for orig, inj in zip(activations_dict[key], activations_dict[inj_key]):
            # Handle both 2D (token, hidden) or 3D (batch, token, hidden)
            if orig.ndim == 3:  # (batch, token, hidden)
                orig_vec = orig[0, token_pos, :].flatten().unsqueeze(0)
                inj_vec = inj[0, token_pos, :].flatten().unsqueeze(0)
            else:  # (token, hidden)
                orig_vec = orig[token_pos, :].flatten().unsqueeze(0)
                inj_vec = inj[token_pos, :].flatten().unsqueeze(0)
            
            cos = F.cosine_similarity(orig_vec, inj_vec).item()
            sims.append(cos)
        
        correlations[f"{key}_corr"] = sims

    return correlations


def plot_layerwise_correlations(corr_dict):
    """
    Plot all correlations in a dictionary with proper labels.

    Args:
        corr_dict (dict): Dictionary of correlations where keys are labels and values are lists of floats
    """
    plt.figure(figsize=(10, 5))
    layers = range(len(next(iter(corr_dict.values()))))  # assume all lists have same length

    # Predefined markers/colors for common keys
    markers = ['o', 's', '^', 'x', 'd', 'v']
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

    for i, (key, values) in enumerate(corr_dict.items()):
        plt.plot(layers, values, marker=markers[i % len(markers)], color=colors[i % len(colors)], label=key)

    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.title('Layerwise Cosine Similarity between Original and Injected Activations')
    plt.xticks(layers)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sorted_logit_distribution(
    logits,
    tokenizer=None,
    top_k=None,
    title="Sorted Logit Distribution"
):
    """
    Visualize sorted logits from highest to lowest.

    Args:
        logits (Tensor): shape [vocab] or [batch, vocab]
        tokenizer: optional tokenizer for decoding top tokens
        top_k (int): if provided, plot only top-k logits
        title (str): plot title
    """
    # Handle batch dimension
    if logits.ndim == 2:
        logits = logits[0]

    logits = logits.detach().cpu()

    # Sort logits descending
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    if top_k is not None:
        sorted_logits = sorted_logits[:top_k]
        sorted_indices = sorted_indices[:top_k]

    x = range(len(sorted_logits))

    plt.figure(figsize=(10, 5))
    plt.plot(x, sorted_logits)
    plt.xlabel("Token rank (sorted by logit)")
    plt.ylabel("Logit value")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Annotate top tokens if tokenizer provided
    if tokenizer is not None:
        top_tokens = [
            tokenizer.decode([idx.item()]) for idx in sorted_indices[:10]
        ]
        for i, tok in enumerate(top_tokens):
            plt.annotate(
                tok.replace("\n", "\\n"),
                (i, sorted_logits[i].item()),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9
            )

    plt.tight_layout()
    plt.show()


def compare_sorted_logit_distributions(
    logits_a,
    logits_b,
    top_k=None,
    labels=("Baseline", "Injected"),
    title="Sorted Logit Distribution Comparison",
    compute_metrics=True
):
    """
    Compare two logit distributions by sorting each independently.

    Args:
        logits_a (Tensor): shape [vocab] or [batch, vocab]
        logits_b (Tensor): shape [vocab] or [batch, vocab]
        top_k (int): plot only top-k logits
        labels (tuple): labels for legend
        title (str): plot title
        compute_metrics (bool): whether to compute summary statistics

    Returns:
        dict (optional): summary metrics if compute_metrics=True
    """
    # Handle batch dimension
    if logits_a.ndim == 2:
        logits_a = logits_a[0]
    if logits_b.ndim == 2:
        logits_b = logits_b[0]

    logits_a = logits_a.detach().cpu()
    logits_b = logits_b.detach().cpu()

    # Sort independently
    a_sorted, _ = torch.sort(logits_a, descending=True)
    b_sorted, _ = torch.sort(logits_b, descending=True)

    if top_k is not None:
        a_sorted = a_sorted[:top_k]
        b_sorted = b_sorted[:top_k]

    x = range(len(a_sorted))

    plt.figure(figsize=(10, 5))
    plt.plot(x, a_sorted, label=labels[0])
    plt.plot(x, b_sorted, label=labels[1])
    plt.xlabel("Token rank (sorted independently)")
    plt.ylabel("Logit value")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if not compute_metrics:
        return None

    # ---- Quantitative diagnostics ----
    metrics = {}

    # Cosine similarity of sorted logits
    metrics["cosine_similarity"] = F.cosine_similarity(
        a_sorted.unsqueeze(0),
        b_sorted.unsqueeze(0)
    ).item()

    # Mean absolute difference
    metrics["mean_abs_diff"] = torch.mean(
        torch.abs(a_sorted - b_sorted)
    ).item()

    # Top-1 gap difference
    metrics["top1_gap_diff"] = (a_sorted[0] - b_sorted[0]).item()

    # Tail mass comparison
    tail_start = int(0.9 * len(a_sorted))
    metrics["tail_mean_diff"] = (
        torch.mean(a_sorted[tail_start:]) -
        torch.mean(b_sorted[tail_start:])
    ).item()

    return metrics
