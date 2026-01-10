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
    random_seed=12
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
    return {
        "act_out": activation_outputs,
        "act_out_inj": activation_outputs_inj,
        "mlp_hid": mlp_hiddens,
        "mlp_hid_inj": mlp_hiddens_inj,
        "mlp_act": mlp_acts,
        "mlp_acts_inj": mlp_acts_inj
    }




def compute_layerwise_cosine_similarity_general(activations_dict, token_pos=-1):
    """
    Compute cosine similarity for all original/injected activation pairs in a dictionary.

    For each key in activations_dict that does NOT end with '_inj', it finds the corresponding
    injected key (key + '_inj') and computes layerwise cosine similarity.

    Returns:
        dict: keys are original keys with '_corr' appended, values are lists of cosine similarities
    """
    correlations = {}

    for key in activations_dict:
        # Skip keys that are already injected
        if key.endswith('_inj'):
            continue
        
        inj_key = key + '_inj'
        if inj_key not in activations_dict:
            # Skip if no injected pair
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