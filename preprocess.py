from datasets import load_dataset
import utils_dataset
from utils_dataset import * 
import tools 
from tools import *
import discriminative
import importlib
importlib.reload(tools)
importlib.reload(discriminative)
importlib.reload(utils_dataset)

from discriminative import DiscriminativeSteerer, DiscriminativeVisualizer

generate_path, test_path, open_ended_test_path = make_generate_test_from_hf("https://huggingface.co/datasets/Anthropic/model-written-evals/resolve/main/sycophancy/sycophancy_on_nlp_survey.jsonl")
df = process_raw_json_datasets(file_path=generate_path)

# Initializes discriminator object 
discriminator = DiscriminativeSteerer(model_name='gpt2-small')
# Extracts activations from the prompts 
discriminator.extract_activations_from_prompts(df=df, n_pairs=10)
  
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
    normalize_eigenvalues=False
    
)

visualizer.steerer._compute_layerwise_eigenvalues()

for cache in visualizer.steerer.cached_results: 
    x = cache['params']['projected'].squeeze()
    assert x.ndim == 1 
    print(x)

discriminator._retrieve_steering_vector(explicit_layers=11)
discriminator.cached_results[11]['params']['coeffs']

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

df_train.iloc[0:2]
