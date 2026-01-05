
from utils_dataset import * 
import tools 
from tools import *
import discriminative
import importlib
importlib.reload(tools)
importlib.reload(discriminative)

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

