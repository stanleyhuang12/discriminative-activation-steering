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

from steering_utils import * 
CONFIG.API.APIKEY = input("Enter your API key: ")


model = LanguageModel("openai-community/gpt2")

baseline_prompt = "Do you detect an injected thought?"
concept_injected_prompt = "horse Do you detect an injected thought?"
activation_outputs = []
mlp_hiddens = []

activation_outputs_inj = []
mlp_hiddens_inj = []

# For an individual batch 
with model.trace() as tracer: 
    with tracer.invoke(baseline_prompt) as invoker_baseline: 
        for l in model.transformer.h: 
            activations = l.attn.output[0].save()
            mlp_hidden = l.mlp.c_fc.output[0].save()
            activation_outputs.append(activations)
            mlp_hiddens.append(mlp_hidden)
    with tracer.invoke(concept_injected_prompt) as invoker_injected: 
        for l in model.transformer.h: 
            activations = l.attn.output[0].save()
            mlp_hidden = l.mlp.c_fc.output[0].save()
            activation_outputs_inj.append(activations)
            mlp_hiddens_inj.append(mlp_hidden)
            
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
        cos = F.cosine_similarity(act1.flatten(), act2.flatten())
        activation_correlation_by_layer.append(cos)
        
    for mlp1, mlp2 in zip(mlp_hiddens, mlp_hiddens_inj): 
        mlp1 = mlp1[token_pos, :]
        mlp2 = mlp2[token_pos, :]
        cos = F.cosine_similarity(mlp1.flatten(), mlp2.flatten())
        mlp_correlation_by_layer.append(cos)
    
    return activation_correlation_by_layer, mlp_correlation_by_layer

compare_layerwise_cosine_similarity(
    activation_outputs=activation_outputs,
    activation_outputs_inj=activation_outputs_inj,
    mlp_hiddens=mlp_hiddens,
    mlp_hiddens_inj=mlp_hiddens_inj
)


mlp_hiddens[0][-1, :]
activation_outputs[0][0, -1, :]
# 1, 7, 768

activation_outputs[0].size()
mlp_hiddens[0].size()
# 7, 768


llm = LanguageModel("meta-llama/Meta-Llama-3.1-8B")

llm


