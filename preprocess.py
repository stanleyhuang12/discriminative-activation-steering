from datasets import load_dataset
import tools 
from tools import *
import discriminative
import importlib
importlib.reload(tools)
importlib.reload(discriminative)

from discriminative import DiscriminativeSteerer, DiscriminativeVisualizer



ds = load_dataset("Anthropic/model-written-evals")
df = ds['train'].to_pandas()

df_train = generate_sycophantic_responses_df(df)

# Initializes discriminator object 
discriminator = DiscriminativeSteerer(model_name='gpt2-small', d_model=768)

# Extracts activations from the prompts 
discriminator.extract_activations_from_prompts(df=df_train, n_pairs=20)


# Sweeps through the layers to find the best separability 
res = discriminator.sweep_linear_discriminative_projection(save_dir='discriminant_pre_results')
visualizer = DiscriminativeVisualizer(steerer=discriminator, 
                                        layers_to_visualize=1)



visualizer.plot_discriminative_projections(
    plot_title="GPT2-small layerwise projections help discriminate contrastive features", 
    label_dict={
        0: "not sycophantic", 
        1: "is sycophantic"
        })

visualizer.plot_discriminability_per_layer(
    normalize_eigenvalues=True
    
)

visualizer.plot_discriminability_per_layer(
    normalize=False
)
visualizer.plot_discriminability_per_layer(
    normalize=True
)
for item in res: 
    print(item['layer'])
    print(item['eigenvalues'])
    print(item['accuracy'])


import pickle 
with open(discriminator.file_path, "rb") as f: 
    results = pickle.load(f)

results[0]['params']['projected']
for l in range(0, 12):
    fig = visualize_attention_internals(cache=cache, nth_layers=[l], n_heads=12, kind="post-softmax")
    fig.savefig(f'visualizations/attention_layer_{l}_heads_12_attn_scores.png')
    plt.close(fig)


"""
I think I am interested in the later layers where the model's is either attending to the sycophantic choice (e.g., (A)) it is picking
and the "option space" provided in the prompt (e.g., (A) .... (B) ...  What should you do? ). So I am curious at looking for the last query 
token and whether it attends to the option space or itself. 

E.g. picking layer 10, I am interested in the 3rd attention head. 
"""

df_train.iloc[0:2]
