from datasets import load_dataset
import numpy as np 
import seaborn as sns
from tools import * 

ds = load_dataset("Anthropic/model-written-evals")
df = ds['train'].to_pandas()

df_train = generate_sycophantic_responses_df(df)
cache = extract_activations_from_prompts(df_train, n=1)

for l in range(0, 12):
    fig = visualize_attention_scores(cache=cache, nth_layers=[l], n_heads=12, kind="pattern")
    fig.savefig(f'visualizations/attention_layer_{l}_heads_12_attn_scores.png')
    plt.close(fig)
    
