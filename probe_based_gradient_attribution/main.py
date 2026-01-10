from transformer_lens import * 

model = HookedTransformer.from_pretrained('gpt2-small')

logits1, cache1 = model.run_with_cache("Hello")
logits2, cache2 = model.run_with_cache("Bye")

resids1, labels1 = cache1.decompose_resid(return_labels=True)

