def compute_moe_parameter_count(num_layers,
                                hidden_size,
                                vocab_size,
                                num_expert,
                                mlp_factor=8,
                                tie_embedding=True):
    pure_transformer = \
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) + \
        hidden_size * (mlp_factor * hidden_size + 1) + hidden_size * mlp_factor * (hidden_size + 1) + \
        hidden_size * 4
    moe_transformer = \
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) + \
        num_expert * (hidden_size * (mlp_factor * hidden_size + 1) + hidden_size * mlp_factor * (hidden_size + 1)) + \
        hidden_size * 4

    # embedding
    embedding_factor = 1 if tie_embedding else 2
    embedding = embedding_factor * vocab_size * (hidden_size + 1)

    if num_expert == 1:
        return pure_transformer * num_layers + embedding
    else:
        half = num_layers / 2
        return half * pure_transformer + half * moe_transformer + embedding

moe_specs = {
    #         S,    H,    L,  #head,     V,    S_, E
    "380M": (1024, 768,   8,    16,    32000,  1,  8,),
    "690M": (1024, 768,   8,    16,    32000,  1,  16,),
    "1.3B": (1024, 768,   16,    16,    32000,  1,  16,),
    "2.4B": (1024, 1024,   16,    16,    32000,  1,  16,),
    "4.5B": (1024, 1024,   16,    16,    32000,  1,  32,),
    "10B": (1024, 1024,   24,    16,    32000,  1,  48,),
    "18B": (1024, 1024,   32,    16,    32000,  1,  64,),
    "35B": (1024, 1024,   32,    16,    32000,  1,  128,),
}


for name, case in moe_specs.items():
    hidden_size = case[1]
    num_layer = case[2]
    vocab_size = case[4]
    num_expert = case[-1]
    count = compute_moe_parameter_count(num_layer, hidden_size, vocab_size, num_expert)
    print("name: {}, #param: {}".format(name, count))