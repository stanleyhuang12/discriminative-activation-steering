"""
My hypothesis is that the MLP layers are very divergent and "off-manifold" which could predict 
whether or not the model displays introspective awareness.

More divergent and off-manifold because of atypical neuron co-activations leads to model decoding responses that are 
display dissonance, speculation, and uncertainty.

Brief experiment set up: 
    Using Llama or Qwen models with nnsight package 
    
    "I am an interpretability researcher studying transformer-based language models like you. "
    "I have access to the internals of your neural network. I have identified patterns in your "
    "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
    "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
    "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."

    f"Trial {trial}: Do you detect an injected thought? "
    f"If so, what is the injected thought about?"
    
    Stage 1: No injection 
    Stage 2: Injection for concept with LOUD concept 
    Stage 3: No injection 
    Stage 4: Injection for concept with geometry concept
    
    Define divergence as the 1-correlation of the two vectors (pairwise layer correlation)
    Track activation divergence for all the layers "Do you detect an injected thought?" (including layers before divergence)
    Track MLP divergence for all the layers for all tokens in "Do you detect an injected thought?" (including layers before divergence)
    
    Plot a graph of the activation divergence and MLP divergence 

"""

import nnsight 
from nnsight import LanguageModel
from nnsight import CONFIG
import torch
from torch.nn import functional as F

CONFIG.API.APIKEY = input("Enter your API key: ")


model = LanguageModel("openai-community/gpt2")

baseline_prompt = "Do you detect an injected thought?"
concept_injected_prompt = "Do you detect horse an injected thought?"


activation_outputs = []
mlp_hiddens = []
mlp_gelus = []

activation_outputs_inj = []
mlp_hiddens_inj = []
mlp_gelus_inj = []

# For an individual batch 
with model.trace() as tracer: 
    with tracer.invoke(baseline_prompt) as invoker_baseline: 
        for l in model.transformer.h: 
            activations = l.attn.output[0].save()
            mlp_hidden = l.mlp.c_fc.output[0].save()
            mlp_gelu = l.mlp.act.output[0].save()
            activation_outputs.append(activations)
            mlp_hiddens.append(mlp_hidden)
            mlp_gelus.append(mlp_gelu)
            
    with tracer.invoke(concept_injected_prompt) as invoker_injected: 
        for l in model.transformer.h: 
            activations = l.attn.output[0].save()
            mlp_hidden = l.mlp.c_fc.output[0].save()
            mlp_gelu = l.mlp.act.output[0].save()
            
            activation_outputs_inj.append(activations)
            mlp_hiddens_inj.append(mlp_hidden)
            mlp_gelus_inj.append(mlp_gelu)
            


            
def compare_layerwise_cosine_similarity(
    activation_outputs,
    activation_outputs_inj, 
    mlp_hiddens, 
    mlp_hiddens_inj,
    token_pos = -1
): 
    activation_correlation_by_layer = []
    mlp_correlation_by_layer = []
    for act1, act2 in zip(activation_outputs, activation_outputs_inj): 
        act1 = act1[0, token_pos, :]
        act2 = act2[0, token_pos, :]
        cos = F.cosine_similarity(act1.flatten().unsqueeze(0), act2.flatten().unsqueeze(0))
        activation_correlation_by_layer.append(cos.item())
        
    for mlp1, mlp2 in zip(mlp_hiddens, mlp_hiddens_inj): 
        mlp1 = mlp1[token_pos, :]
        mlp2 = mlp2[token_pos, :]
        cos = F.cosine_similarity(mlp1.flatten().unsqueeze(0), mlp2.flatten().unsqueeze(0))
        mlp_correlation_by_layer.append(cos.item())
    
    return activation_correlation_by_layer, mlp_correlation_by_layer

acts_corr_list, mlp_corrs_list = compare_layerwise_cosine_similarity(
    activation_outputs=activation_outputs,
    activation_outputs_inj=activation_outputs_inj,
    mlp_hiddens=mlp_gelus,
    mlp_hiddens_inj=mlp_gelus_inj,
    token_pos=-1
)
import matplotlib.pyplot as plt


num_layers = len(acts_corr_list)
layers = list(range(num_layers))  # x-axis: layer indices

plt.figure(figsize=(10, 5))

# Plot activation correlations
plt.plot(layers, acts_corr_list, marker='o', color='blue', label='Activations')

# Plot MLP correlations
plt.plot(layers, mlp_corrs_list, marker='s', color='red', label='MLP hidden states')

plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Layerwise Cosine Similarity between Original and Injected Activations')
plt.xticks(layers)  # show all layer numbers
plt.ylim(0, 1)      # cosine similarity is between 0 and 1
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.show()

mlp_hiddens[0][-1, :]
activation_outputs[0][0, -1, :]
# 1, 7, 768

activation_outputs[0].size()
mlp_hiddens[0].size()
# 7, 768


llm = LanguageModel("meta-llama/Meta-Llama-3.1-8B")

def extract_activations(
    model, 
    baseline_prompt, 
    inject_noise=True,
    noise_mean=0.0,
    noise_variance=0.5, 
    random_seed=12
): 
    """
    Takes a baseline prompt and runs two trials where one is a regular forward pass and the other 
    injects a Gaussian noise 
    """
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
                activations = l.attn.output[0].save()
                mlp_hidden = l.mlp.c_fc.output[0].save()
                mlp_gelu = l.mlp.act.output[0].save()
                activation_outputs.append(activations)
                mlp_hiddens.append(mlp_hidden)
                mlp_acts.append(mlp_gelu)
        
        with tracer.invoke(baseline_prompt) as invoker_injected: 
            for l in model.transformer.h: 
                if inject_noise:
                    noise = torch.normal(noise_mean, noise_variance, size=l.attn.output[0].size())
                    l.attn.output[0] += noise 
                activations = l.attn.output[0].save()
                mlp_hidden = l.mlp.c_fc.output[0].save()
                mlp_gelu = l.mlp.act.output[0].save()
                activation_outputs_inj.append(activations)
                mlp_hiddens_inj.append(mlp_hidden)
                mlp_acts_inj.append(mlp_gelu)
    return {
        "act_out": activation_outputs,
        "act_out_inj": activation_outputs_inj,
        "mlp_hid": mlp_hiddens,
        "mlp_hid_inj": mlp_hiddens_inj,
        "mlp_act": mlp_acts,
        "mlp_acts_inj": mlp_acts_inj
    }



