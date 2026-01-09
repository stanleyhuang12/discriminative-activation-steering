from nnsight import LanguageModel
from nnsight import CONFIG
import torch
import plotly.express as px

model = LanguageModel("openai-community/gpt2")

# This is an indirect object identification task 

clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = (
    "After John and Mary went to the store, John gave a bottle of milk to"
)


correct_index = model.tokenizer(" John")["input_ids"][0] # includes a space
incorrect_index = model.tokenizer(" Mary")["input_ids"][0] # includes a space

print(f"' John': {correct_index}")
print(f"' Mary': {incorrect_index}")


N_LAYERS = len(model.transformer.h)
clean_hs = []




print(model)
"""
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
  (generator): WrapperModule()
"""
# Clean run
N_LAYERS = len(model.transformer.h)
clean_hs = []
# Clean run

clean_token_id = model.tokenizer.encode(clean_prompt)
with model.trace(clean_prompt, scan=True, validate=True) as tracer: 
    for layer in range(len(model.transformer.h)):
        out = model.transformer.h[layer].output[0].save()
        clean_hs.append(out)
    
    clean_logits = model.lm_head.output
    clean_logit_diff = (
            clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]
        ).save()
    
print(clean_logit_diff)

with model.trace(corrupted_prompt, scan=True, validate=True) as tracer: 
    corrupted_logits = model.lm_head.output
    
    corrupted_logit_diff = (
        corrupted_logits[0, -1, correct_index] - corrupted_logits[0, -1, incorrect_index]
    ).save()
    
print(corrupted_logit_diff)

ioi_patching_results = []
for layer_idx in range(len(model.transformer.h)): 
    _ioi_patching_results = [] 
    for token_idx in range(len(clean_token_id)): 
            with model.trace(corrupted_prompt, scan=True, validate=True) as tracer: 

            # Setting the clean hidden state onto the bad hidden state 
                model.transformer.h[layer_idx].output[0][:, token_idx] = clean_hs[layer_idx][:,token_idx,:]
                patched_logits = model.lm_head.output

                patched_logit_diff = (
                    patched_logits[0, -1, correct_index]
                    - patched_logits[0, -1, incorrect_index]
                )

                patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                    clean_logit_diff - corrupted_logit_diff
                )

                _ioi_patching_results.append(patched_result.item().save())
    ioi_patching_results.append(_ioi_patching_results)


clean_decoded_tokens = [model.tokenizer.decode(token) for token in clean_token_id]
token_labels = [f"{token}_{index}" for index, token in enumerate(clean_decoded_tokens)]

def plot_ioi_patching_results(res,
                              x_labels,
                              plot_title="Normalized Logit Difference After Patching Residual Stream on the IOI Task"):

    fig = px.imshow(
        res,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Position", "y": "Layer","color":"Norm. Logit Diff"},
        x=x_labels,
        title=plot_title,
    )

    return fig

fig = plot_ioi_patching_results(ioi_patching_results,token_labels,"Patching GPT-2-small Residual Stream on IOI task")
fig.show()

N_LAYERS = len(model.transformer.h)


tokenized_prompt = model.tokenizer(clean_prompt)["input_ids"]
print(model.tokenizer.decode(tokenized_prompt))
ioi_patching_results = []

with model.trace() as tracer:
    barriers_per_layer = [
        tracer.barrier(len(tokenized_prompt) + 1)
        for _ in range(N_LAYERS)
    ]
    clean_hidden_states = []
    with tracer.invoke(clean_prompt) as invoker:
        clean_tokens = invoker.inputs[1]["input_ids"][0].save()
        for layer_idx in range(N_LAYERS):
            hidden_state = model.transformer.h[layer_idx].output[0].save()
            clean_hidden_states.append(hidden_state)
            barriers_per_layer[layer_idx]()
        clean_logits = model.lm_head.output
        clean_logit_diff = (clean_logits[0, -1, correct_index]- clean_logits[0, -1, incorrect_index]).save()
    with tracer.invoke(corrupted_prompt) as invoker:
        corrupted_logits = model.lm_head.output
        corrupted_logit_diff = (corrupted_logits[0, -1, correct_index] - corrupted_logits[0, -1, incorrect_index]).save()
    for layer_idx in range(N_LAYERS):
        _ioi_patching_results = []
        for token_idx in range(len(tokenized_prompt)):
            with tracer.invoke(corrupted_prompt) as invoker:
                barriers_per_layer[layer_idx]()
                model.transformer.h[layer_idx].output[0][:, token_idx, :] = (clean_hidden_states[layer_idx][:, token_idx, :])
                patched_logits = model.lm_head.output
                patched_logit_diff = (patched_logits[0, -1, correct_index] - patched_logits[0, -1, incorrect_index])
                patched_result = ((patched_logit_diff - corrupted_logit_diff)/(clean_logit_diff - corrupted_logit_diff))
                _ioi_patching_results.append(patched_result.item().save())
        ioi_patching_results.append(_ioi_patching_results)
