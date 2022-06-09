opt_specs = {
         #S,    H,     L,   head,  V
"125M": (2048,  768,   4,   12,  50272),
"350M": (2048,  1024,  24,   16,  50272),
"1.3B": (2048,  2048,  24,   32,  50272),
"2.7B": (2048,  2560,  32,   32,  50272),
"6.7B": (2048,  4096,  32,   32,  50272),
"13B": (2048,  5120,  40,   40,  50272),
"30B": (2048,  7168,  48,   56,  50272),
"175B": (2048,  12288,  96,   96,  50272),
}

def gpt_flops(
    input_seq_len: int,
    output_seq_len: int,
    vocab_size: int = 50272,
    num_heads: int = 96,
    d_model: int = 12288,
    d_ff: int = 49152,
    num_layers: int = 96,
):
    assert input_seq_len + output_seq_len <= 2049
    d_head = int(d_model / num_heads)

    # Here, we count the number of MACs instead of FLOPs.
    # Then, we will mutiply it by 2 (i.e., FLOPs = 2 * MACs).
    total_macs = 0

    # Processing the input tokens.
    q = num_heads * d_head * d_model * input_seq_len
    k = num_heads * d_head * d_model * input_seq_len
    v = num_heads * d_head * d_model * input_seq_len
    attn = 2 * num_heads * (input_seq_len * d_head * input_seq_len)
    ffn1 = d_ff * d_model * input_seq_len
    ffn2 = d_model * d_ff * input_seq_len
    macs_per_layer = q + k +v + attn + ffn1 + ffn2

    sampling = vocab_size * d_model * 1 # last input token
    total_macs += macs_per_layer * num_layers + sampling

    # Processing the output sequence.
    for i in range(1, output_seq_len):
        # Process 1 token at a time.
        q = num_heads * d_head * d_model * 1
        k = num_heads * d_head * d_model * 1
        v = num_heads * d_head * d_model * 1

        seq_len = input_seq_len + i
        attn = 2 * num_heads * (seq_len * d_head * 1)
        ffn1 = d_ff * d_model * 1
        ffn2 = d_model * d_ff * 1
        macs_per_layer = q + k + v + attn + ffn1 + ffn2

        sampling = vocab_size * d_model * 1
        total_macs += macs_per_layer * num_layers + sampling

    total_flops = 2 * total_macs
    return total_flops

def compute_gpt_tflops_with_padding(batch_size, gen_len, seq_len, num_layers, hidden_size, vocab_size,
                                    num_gpus, latency, checkpoint_activations=False):
    factor = 96 if checkpoint_activations else 72
    total_flop = factor * batch_size * gen_len * (hidden_size ** 2) * num_layers * \
          (1 + seq_len / (6 * hidden_size)) \
          + 10 * batch_size * gen_len * hidden_size * vocab_size
    # Note: if we use dot to compute forward embedding
    # then the last term in total_flops should be
    # "+ 10 * batch_size * seq_len * hidden_size * vocab_size".
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops