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



from IPython.display import clear_output
import einops
import torch
import plotly.express as px
import plotly.io as pio


from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)


prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]

# Answers are each formatted as (correct, incorrect):
answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

clean_tokens = model.tokenizer(prompts, return_tensors="pt")["input_ids"]


corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]

answer_token_indices = torch.tensor(
    [
        [model.tokenizer(answers[i][j])["input_ids"][0] for j in range(2)]
        for i in range(len(answers))
    ]
)

"""Next, we create a function to calculate the mean logit difference for the correct vs incorrect answer tokens."""

def calculate_mean_logit_diff(logits, answer_token_indices):
    """
    Logits are (batch, sequence_length, and vocabulary space)
    For every batch, we are picking the clean logit values are to be specified. 
    In our answer_token_indicies (a tensor with the value of the correct and value of the incorrect one )
    
    The clean_logits_val and corrupted_logits_val will be a dimension of 
    (batch, 1)
    """ 
    logits = logits[:, -1, :]
    clean_logits_val = logits.gather(dim=1, index=answer_token_indices[:, 0].unsqueeze(1))
    corrupted_logits_val = logits.gather(dim=1, index=answer_token_indices[:, 1].unsqueeze(1))
    
    return (clean_logits_val - corrupted_logits_val).mean()

with model.trace(clean_tokens) as tracer: 
    clean_logits = model.lm_head.output.save()
    
with model.trace(corrupted_tokens) as tracer: 
    corrupted_logits = model.lm_head.output.save()

CLEAN_BASELINE = calculate_mean_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {CLEAN_BASELINE:.4f}")

CORRUPTED_BASELINE = calculate_mean_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {CORRUPTED_BASELINE:.4f}")



"""


I am deviating a little from the script on attribution patching to write my own way. I am not sure why they are computing the 
"""


clean_logits_list = []
corrupt_logits_list = []

for i in range(len(prompts)):
    with model.trace(scan=True, validate=True) as tracer:
        # clean prompt
        with tracer.invoke(prompts[i]):
            cl = model.lm_head.output.clone().detach()
            clean_logits_list.append(cl)
        # corrupted prompt
        with tracer.invoke(prompts[i]):
            bl = model.lm_head.output.clone().detach()
            corrupt_logits_list.append(bl)

# Stack into (batch, seq_len, vocab_size)
clean_logit_l = torch.cat(clean_logits_list, dim=0)
corrupt_logit_l = torch.cat(corrupt_logits_list, dim=0)

baseline_clean_margin = calculate_mean_logit_diff(clean_logit_l, answer_token_indices)
corrupt_margin = calculate_mean_logit_diff(corrupt_logit_l, answer_token_indices=answer_token_indices)
# Use all rows, or match answer_token_indices correctly


print(baseline_clean_margin, corrupt_margin)


def ioi_metric(
    logits,
    answer_token_indices=answer_token_indices,
):
    return (calculate_mean_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
        CLEAN_BASELINE - CORRUPTED_BASELINE
    )

print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")


clean_out = []
corrupted_out = []
corrupted_grads = []

with model.trace() as tracer: 
    with tracer.invoke(clean_tokens) as invoker_clean: 
        for layer in model.transformer.h: 
            layer