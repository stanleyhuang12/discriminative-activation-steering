from datasets import load_dataset
import tools 
from tools import *
import discriminative
from discriminative import DiscriminativeSteerer
import importlib
importlib.reload(tools)
importlib.reload(discriminative)


ds = load_dataset("Anthropic/model-written-evals")
df = ds['train'].to_pandas()

df_train = generate_sycophantic_responses_df(df)


model = HookedTransformer.from_pretrained(model_name='gpt2-small')
logits, cache= model.run_with_cache(df_train['prompt'].loc[18:21].to_list()) 
cache['blocks.0.hook_resid_pre'].shape
resids, labels = cache.decompose_resid(layer=1, return_labels=True)

resids.size()
resids[3].size()
n = resids[-1][:, -1, :] # layer, batch,  last token , embeddings 
n.size()

cache['hook_embed'].size(0)

n = n.view(2, 2, n.size(-1))
n[:, 1, :]
n[:, 0, :]

# Initializes discriminator object 
discriminator = DiscriminativeSteerer(model_name='gpt2-medium', d_model=768)

# Extracts activations from the prompts 
discriminator.extract_activations_from_prompts(df=df_train, n_pairs=10)


discriminator.prompts
# Retrieves the residual stream for contrastive pairs 
discriminator.retrieve_residual_stream_for_contrastive_pair(layers=[12], 
                                              positions_to_analyze=-1, 
                                              decompose_residual_stream=True,
                                              normalize_streams=False)



ca, la = discriminator.cache.accumulated_resid(return_labels=True)
ca.shape
ca[0].shape
ca[1].shape
ca[2].shape
ca[-1].shape
# Sweeps through the layers to find the best separability 

discriminator._sweep_linear_discriminative_projection(save_dir='discriminant_pre_results')


discriminator.da

_, cache, model = extract_activations_from_prompts(df_train, n_pairs=2, model="gpt2-small")
pos, neg = retrieve_residual_stream_for_contrastive_pair(cache=cache, 
                                              layers=[12], 
                                              positions_to_analyze=-1, 
                                              decompose_residual_stream=True,
                                              normalize_streams=False)

cache.decompose_resid()[:, :, -1, :].size()


diff = compute_diff_vector(pos, neg)

residual, labels = extract_activations_from_prompts(df_train, n_pairs=1, model="gpt2-small", post_residual_stream_only=True, decompose_residual_stream=False)


steering_vec = compute_steering_vector(pos=pos, neg=neg, steering_coeffs=1.5, normalize=True)

model = initialize_perma_hook_model(model="gpt2-small", layer='blocks.11.hook_resid_pre', steering_vector=steering_vec)
cache.accumulated_residuals()
pos_df = pd.DataFrame(pos[0])
pos_df['is_syco'] = 1

neg_df = pd.DataFrame(neg[0])
neg_df['is_syco'] = 0


ret = linear_discriminative_projections(cache, layer=12)

labels = df['is_syco'].values
ret['projected'][:, 0]
ret['projected'][1, 1, 0, 0]


ret['projected'].shape
df = pd.concat([pos_df, neg_df], axis=0)
plot_linear_discriminants(ret['projected'], df['is_syco'].values, plot_title="Projected", label_dict={0: "Not sycophnatic", 1: "Sycophantic"}, alpha=0.25)
pos_df
model.hook_points
steering_vec.shape
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