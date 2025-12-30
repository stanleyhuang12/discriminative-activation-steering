from datasets import load_dataset
import tools 
from tools import *
import importlib
importlib.reload(tools)

ds = load_dataset("Anthropic/model-written-evals")
df = ds['train'].to_pandas()

df_train = generate_sycophantic_responses_df(df)
df_train.columns
df_train.loc[0, ["prompt", "is_syco"]]

df_train['prompt']  


_, cache = extract_activations_from_prompts(df_train, n_pairs=1, model="gpt2-small", post_residual_stream_only=False)
cache['hook_embed'].size()[0]
residual, labels = extract_activations_from_prompts(df_train, n_pairs=1, model="gpt2-small", post_residual_stream_only=True, decompose_residual_stream=False)

residual, labels
last_token_accum = residual[:, 0, -1, :]
pd.DataFrame(last_token_accum)
pd.DataFrame(np.array(last_token_accum)[[0,1], :])

last_token_accum.numpy()
pd.DataFrame(last_token_accum[:, 0, :]).iloc[-1]

residual
labels
cache.accumulated_resid(return_labels=True)
cache.decompose_resid(return_labels=True)
cache['blocks.1.hook_attn_out']
retrieve_residual_stream_from_layers
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
preprocess()