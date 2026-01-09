
from nnsight import LanguageModel


model = LanguageModel("openai-community/gpt2", device_map="auto")

import nnsight
from nnsight import CONFIG


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
N_LAYERS = len(model.transformer.h)
clean_hs = []

# Clean run
with model.trace() as tracer:
    with tracer.invoke(clean_prompt) as invoker:
        print(invoker.inputs[0]['input_ids'])
        clean_tokens = invoker.inputs[0]['input_ids'][0]

        # Get hidden states of all layers in the network.
        # We index the output at 0 because it's a tuple where the first index is the hidden state.
        for layer_idx in range(N_LAYERS):
            clean_hs.append(model.transformer.h[layer_idx].output[0].save())

        # Get logits from the lm_head.
        clean_logits = model.lm_head.output

        # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
        clean_logit_diff = (
            clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]
        ).save()
"""
My test to see what the model response is 
"""
    
with model.trace(corrupted_prompt) as tracer: 
    tok_index = model.lm_head.output[0].argmax().save()
    model.tokenizer.decode(tok_index.item())
    
"the"

print(corrupted_logits_ls[0].size()) # batch (only 1), token, logit space (what is the model's best guess)
corrupted_logit_diff
print("tensor(-2.2725, grad_fn=<SubBackward0>)")


## Patching intervention 

ioi_patching_results = []

"""
For every layer, we have a dimension of [batch_idx, seq_len, hidden_state]
For every layer, we want to index the batch, go through each seq_len and patch that hidden state with the clean one 
Note that we stored the clean hidden states in clean_hs [batch, seq_len, hidden_state]
We patch it by just setting those values
"""

clean_hs[0][:, 0, :]

idx = []

idx[0].size()

ioi_patching_results = []

# Iterate through all the layers
for layer_idx in range(len(model.transformer.h)):
    _ioi_patching_results = []

    # Iterate through all tokens
    for token_idx in range(len(clean_tokens)):
        # Patching corrupted run at given layer and token
        with model.trace(corrupted_prompt) as tracer:
            # Apply the patch from the clean hidden states to the corrupted hidden states.
            model.transformer.h[layer_idx].output[0][:, token_idx] = clean_hs[layer_idx][:,token_idx,:]

            patched_logits = model.lm_head.output

            patched_logit_diff = (
                patched_logits[0, -1, correct_index]
                - patched_logits[0, -1, incorrect_index]
            )

            # Calculate the improvement in the correct token after patching.
            patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                clean_logit_diff - corrupted_logit_diff
            )

            _ioi_patching_results.append(patched_result.item())
            _ioi_patching_results.save()

    ioi_patching_results.append(_ioi_patching_results)

    


    
model.transformer.h[0].__sizeof__()

## The difference 
clean_logits[0].size() # SIZE: num_tokens, num of words 
clean_hs[0].size()  # SIZE: layer, num_tokens, hidden state
            
            
clean_logits[0].size()



tokenized_prompt = model.tokenizer(clean_prompt)["input_ids"]
print(model.tokenizer.decode(tokenized_prompt))

N_LAYERS = len(model.transformer.h)

ioi_patching_results = []
with model.trace() as tracer:
    # define barriers for each layer and token
    barriers_per_layer = [tracer.barrier(len(tokenized_prompt)+1) for _ in range(N_LAYERS)]
    # Clean run
    with tracer.invoke(clean_prompt) as invoker:
        clean_tokens = invoker.inputs[1]["input_ids"][0].save()

        # Get hidden states of all layers in the network.
        # We index the output at 0 because it's a tuple where the first index is the hidden state.
        for layer_idx in range(N_LAYERS):
            hidden_state = model.transformer.h[layer_idx].output[0]
            # call barrier so nnsight can prepare the hidden states for the patching invoke
            barriers_per_layer[layer_idx]()

        # Get logits from the lm_head.
        clean_logits = model.lm_head.output

        # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
        clean_logit_diff = (
            clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]
        ).save()

    # Corrupted run
    with tracer.invoke(corrupted_prompt) as invoker:
        corrupted_logits = model.lm_head.output

        # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
        corrupted_logit_diff = (
            corrupted_logits[0, -1, correct_index]
            - corrupted_logits[0, -1, incorrect_index]
        ).save()


    # Activation Patching Intervention – across all layers & token positions
    for layer_idx in range(len(model.transformer.h)):
        _ioi_patching_results = []

        for token_idx in range(len(tokenized_prompt)):
            # Patching corrupted run at given layer and token

            with tracer.invoke(corrupted_prompt) as invoker:
                # call barrier so nnsight can grab the hidden states for this invoke
                barriers_per_layer[layer_idx]()

                # Patch in the clean hidden states over the corrupted hidden states.
                model.transformer.h[layer_idx].output[0][:, token_idx, :] = hidden_state[:, token_idx, :]

                patched_logits = model.lm_head.output

                patched_logit_diff = (
                    patched_logits[0, -1, correct_index]
                    - patched_logits[0, -1, incorrect_index]
                )

                # Calculate the improvement in the correct token after patching.
                patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                    clean_logit_diff - corrupted_logit_diff
                )

                _ioi_patching_results.append(patched_result.item())

        ioi_patching_results.append(_ioi_patching_results)
        ioi_patching_results.save()
        