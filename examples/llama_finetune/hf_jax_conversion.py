import transformers

def import_hf_model(model_name_or_path):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
    )
    return model

def hf_to_jax_weight(hf_model):
    state_dict = hf_model.state_dict()
    jax_weights = {
        'transformer': {
            'wte': {'embedding': state_dict['model.embed_tokens.weight'].numpy()},
            'ln_f': {'kernel': state_dict['model.norm.weight'].numpy()},
            'h': {
                '%d' % (layer): {
                    'attention': {
                        'wq': {'kernel': state_dict['model.layers.%d.self_attn.q_proj.weight' % (layer)].numpy().transpose()},
                        'wk': {'kernel': state_dict['model.layers.%d.self_attn.k_proj.weight' % (layer)].numpy().transpose()},
                        'wv': {'kernel': state_dict['model.layers.%d.self_attn.v_proj.weight' % (layer)].numpy().transpose()},
                        'wo': {'kernel': state_dict['model.layers.%d.self_attn.o_proj.weight' % (layer)].numpy().transpose()},
                    },
                    'feed_forward': {
                        'w1': {'kernel': state_dict['model.layers.%d.mlp.gate_proj.weight' % (layer)].numpy().transpose()},
                        'w2': {'kernel': state_dict['model.layers.%d.mlp.down_proj.weight' % (layer)].numpy().transpose()},
                        'w3': {'kernel': state_dict['model.layers.%d.mlp.up_proj.weight' % (layer)].numpy().transpose()},
                    },
                    'attention_norm': {'kernel': state_dict['model.layers.%d.input_layernorm.weight' % (layer)].numpy()},
                    'ffn_norm': {'kernel': state_dict['model.layers.%d.post_attention_layernorm.weight' % (layer)].numpy()},
                }
            for layer in range(hf_model.config.num_hidden_layers)},
        },
        'lm_head': {'kernel': state_dict["lm_head.weight"].numpy().transpose()},
    }
    return jax_weights

if __name__ == "__main__":
    hf_model = import_hf_model("./llama-7b")
    jax_params = hf_to_jax(hf_model)
    # EasyLM uses fout.write(flax.serialization.msgpack_serialize(jax_weights, in_place=True)) to store the param
