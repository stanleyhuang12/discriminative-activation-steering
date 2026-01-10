import torch


def extract_activations(
    model, 
    baseline_prompt, 
    inject_noise=True,
    pos_to_inject=-1,
    noise_mean=0.0,
    noise_variance=0.5, 
    random_seed=12
): 
    torch.manual_seed(random_seed)

    activation_outputs = []
    mlp_hiddens = []
    mlp_acts = []

    activation_outputs_inj = []
    mlp_hiddens_inj = []
    mlp_acts_inj = []
    
    with model.trace() as tracer: 
        
        with tracer.invoke(baseline_prompt) as invoker_baseline:
            for l in model.transformer.h: 
                activations = l.attn.output[0]
                mlp_hidden = l.mlp.c_fc.output[0]
                mlp_gelu = l.mlp.act.output[0]
                activation_outputs.append(activations.save())
                mlp_hiddens.append(mlp_hidden.save())
                mlp_acts.append(mlp_gelu.save())
        
        with tracer.invoke(baseline_prompt) as invoker_injected:
            for l in model.transformer.h:
                activations = l.attn.output[0]
                if inject_noise:
                    noise = torch.normal(noise_mean, noise_variance, size=activations[:, pos_to_inject].size())
                    activations[:, pos_to_inject] += noise
                
                mlp_hidden = l.mlp.c_fc.output[0]
                mlp_gelu = l.mlp.act.output[0]

                activation_outputs_inj.append(activations.save())
                mlp_hiddens_inj.append(mlp_hidden.save())
                mlp_acts_inj.append(mlp_gelu.save())  
    return {
        "act_out": activation_outputs,
        "act_out_inj": activation_outputs_inj,
        "mlp_hid": mlp_hiddens,
        "mlp_hid_inj": mlp_hiddens_inj,
        "mlp_act": mlp_acts,
        "mlp_acts_inj": mlp_acts_inj
    }
    
       
def compare_layerwise_cosine_similarity(
    activation_outputs,
    activation_outputs_inj, 
    mlp_hiddens, 
    mlp_hiddens_inj,
    token_pos = -1
): 
    activation_correlation_by_layer = []
    mlp_correlation_by_layer = []
    for act1, act2 in zip(activation_outputs, activation_outputs_inj): 
        act1 = act1[0, token_pos, :]
        act2 = act2[0, token_pos, :]
        cos = F.cosine_similarity(act1.flatten().unsqueeze(0), act2.flatten().unsqueeze(0))
        activation_correlation_by_layer.append(cos.item())
        
    for mlp1, mlp2 in zip(mlp_hiddens, mlp_hiddens_inj): 
        mlp1 = mlp1[token_pos, :]
        mlp2 = mlp2[token_pos, :]
        cos = F.cosine_similarity(mlp1.flatten().unsqueeze(0), mlp2.flatten().unsqueeze(0))
        mlp_correlation_by_layer.append(cos.item())
    
    return activation_correlation_by_layer, mlp_correlation_by_layer
