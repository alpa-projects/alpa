from transformers.generation_utils import dataclass

opt_specs = {
         #S,    H,     L,   head,  V
"125M": (2048,  768,   12,   12,  50272),
"350M": (2048,  1024,  24,   16,  50272),
"1.3B": (2048,  2048,  24,   32,  50272),
"2.7B": (2048,  2560,  32,   32,  50272),
"6.7B": (2048,  4096,  32,   32,  50272),
"13B": (2048,  5120,  40,   40,  50272),
"30B": (2048,  7168,  48,   56,  50272),
"175B": (2048,  12288,  96,   96,  50272),
}


@dataclass
class TransformerModelConfig:
    # hidden size
    H: int = 768
    # number of layers
    L: int = 12
    # number of attention heads
    n_head: int = 12
    seq_len: int = 2048
    vocab_size: int = 50272


def compute_gpt_tflops_inference_with_padding(batch_size, gen_len, seq_len, num_layers,
                                              hidden_size, vocab_size, num_gpus, latency):
    """This calculation assumes that each code decoded attend to seq_len number tokens."""
    factor = 24
    total_flop = factor * batch_size * gen_len * (hidden_size ** 2) * num_layers * \
          (1 + seq_len / (6 * hidden_size)) \
          + 2 * batch_size * gen_len * hidden_size * vocab_size
    # Note (Hao): it should be 4 here because of input embedding, but we will
    # respect Deepak's eq. instead.
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops

test_prompts = [
    "Computer science is the study of computation and",
    "Ion Stoica is a Romanian-American computer scientist specializing in",
    "The University of California, Berkeley is a public",
    "Today is a good day and I want to",
    "What is the valuation of Databricks?",
    "Paris is the capital city of",
    "Which country has the most population?",
    "What do you think about the future of Cryptocurrency?",
    "What do you think about the meaning of life?",
    "Donald Trump is the president of",
    "GPT-3 is a large language model that is capable of"
]
