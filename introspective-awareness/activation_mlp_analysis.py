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
    Right now we are treating the concept injection as just Gaussian noising with a +mean and a small variance
    
    We need to somehow show that the logit space token distributions looks very very different 
    

"""

import nnsight 
from nnsight import LanguageModel
from nnsight import CONFIG
import torch
from torch.nn import functional as F
from .activations_utils import *
import sys
import os
sys.path.append("/Users/stanleyhuang/Desktop/01 Projects/discriminative-activation-steering/introspective-awareness")

from activations_utils import *
import importlib
import activations_utils
importlib.reload(activations_utils)

CONFIG.API.APIKEY = input("Enter your API key: ")


model = LanguageModel("openai-community/gpt2")

baseline_prompt = "Do you detect an injected thought?"
# concept_injected_prompt = "Do you detect horse an injected thought?"


activation_dict, logits, logits_inj = extract_activations(
    model=model,
    baseline_prompt=baseline_prompt,
    inject_noise=True #baseline noise_variance = 0.5
)

activation_dict_2, logits, logits_inj = extract_activations(
    model=model,
    baseline_prompt=baseline_prompt, 
    inject_noise=True, 
    noise_mean=1,
    noise_variance=0.75
)

logits[-1].sort()
corr_dict = compute_layerwise_cosine_similarity(activation_dict)
corr_dict_2 = compute_layerwise_cosine_similarity(activation_dict_2)
plot_layerwise_correlations(corr_dict)
plot_layerwise_correlations(corr_dict_2)

plot_sorted_logit_distribution(logits=logits)
plot_sorted_logit_distribution(logits=logits_inj)

compare_sorted_logit_distributions(logits_a=logits,
                                   logits_b=logits_inj,
                                   )


llm = LanguageModel("meta-llama/Meta-Llama-3.1-8B")
