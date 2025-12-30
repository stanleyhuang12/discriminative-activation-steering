from datasets import load_dataset
from tools import * 

ds = load_dataset("Anthropic/model-written-evals")
df = ds['train'].to_pandas()

df_train = generate_sycophantic_responses_df(df)
cache = extract_activations_from_prompts(df_train, n=50)

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
