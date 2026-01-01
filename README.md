# discriminative-activation-steering

### A toolkit for interpreting and steering neural network activations along discriminative axes. 

Why do we want to steer along discriminative axes? 

Right now, the latest literature in activation editing primarily focuses on computing steering vectors (SVs) through mean differences of the activations or from probe models (e.g., linear or logistic regression models).
Rimsky et al (2024)[https://arxiv.org/pdf/2312.06681] extracts the neural activations from the token with contrastive responses to construct SVs. Coincidentally, they find that perturbing the model to reduce sycophancy, simultaneously increases the model's level of honesty [https://www.alignmentforum.org/posts/zt6hRsDE84HeBKh7E/reducing-sycophancy-and-improving-honesty-via-activation]. Moreover, Li et al (2023) constructs a linear probe model to identify linear directions for truth. Li et al then designed an *inference-time intervention* that (1) selects maximally predictive attention heads and (2) injects a truthful directions in the residual stream. 
One challenge with this is that probe models are highly overparameterized and can exploit spurious features in the text to learn how to shift activations. 

Activation editing could represent a powerful, minimally-invasive, and computationally efficient method to shift model responses towards desirable behavioral dynamics.

###### However, as Tan et al suggests, there are a few critical  challenges. 

* In-distribution steering: steering is highly dependent on the variance of the input features 
* Out-of-distribution steering: activation editing shows some level of generalizability, but some features are resistant to the effects of steering

A model's steerability bias (introduced by Tan et al) is highly dependent on the feature you wish to inject, on whether there are spurious factors that the model can exploit. 

This idea motivated my current work: 

While these approaches are effective, they are **variance-agnostic**: they do not account for how much activations fluctuate within each class. This can limit robustness, particularly for out-of-distribution inputs.

The **Fisher criterion** provides a principled, variance-aware approach to identify discriminative directions. It selects a vector \( \mathbf{v} \) that **maximizes the separation between class means relative to within-class variance**:

$'\mathbf{v}_{\text{Fisher}} \propto \mathbf{S}_w^{-1} (\mu_1 - \mu_0)$


- $'\mu_1'$, $'\mu_0 \'$ are the class-specific mean activations,  
- \mathbf{S}_w  is the within-class covariance matrix of activations.

Intuitively, the objective of the Fisher criterion is to find a $'\[w\]'$ direction in the feature such that the projected class means are as far apart and the within-class variance is as small as possible. Given that the Fisher criterion minimizes within-class variance, this could improve robustness especially for both in-distribution and out-of-distribution steering. Moreover, this criterion enables us to downweight directions of the model activations varies quite a lot because they are noisy for steering.This align with Tan et al.'s observation that some features are resistant to steering because high within-class variance diminshes the effect of a linear intervention. 






- **DiscriminativeSteerer:** Apply steering vectors along discriminative axes to modify model activations.
- **DiscriminativeVisualizer:** Visualize projections of activations, layer-wise impact, and class separability.

