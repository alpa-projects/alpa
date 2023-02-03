import dataclasses
from dataclasses import dataclass
import jax.numpy as jnp
import alpa

@dataclass(frozen=True)
class OPTConfig:
    # Inherited from OPT
    num_hidden_layers: int = 12
    max_seq_len: int = 2048
    hidden_size: int = 768
    n_head: int = 12
    input_dim: int = 768
    ffn_embed_dim: int = 3072
    pad: int = 1
    activation_fn: str = 'relu'
    dtype: any = jnp.float16
    use_stable_embedding: bool = False
    no_scale_embedding: bool = True
    decoder_learned_pos: bool = True
    decoder_normalize_before: bool = True
    share_decoder_input_output_embed: bool = True
    # Added
    version: int = 1
    vocab_size: int = 50272
    layer_norm_eps: float = 0.00001
    num_pp_stages: int = None
    # parallelize
    mark_boundary: bool = True


def get_config(name, **kwargs):
    if name == "opt-125m":
        config = OPTConfig(
            max_seq_len=2048, num_hidden_layers=12, n_head=12,
            hidden_size=768, input_dim=768, ffn_embed_dim=768 * 4,
            version=3,
        )
    elif name == "opt-350m":
        config = OPTConfig(
            max_seq_len=2048, num_hidden_layers=24, n_head=16,
            hidden_size=1024, input_dim=1024, ffn_embed_dim=1024 * 4,
            version=2,
        )
        raise NotImplementedError("Not implemented because this model "
                                  "has a different architecture")
    elif name == "opt-1.3b":
        config = OPTConfig(
            max_seq_len=2048, num_hidden_layers=24, n_head=32,
            hidden_size=2048, input_dim=2048, ffn_embed_dim=2048 * 4,
            version=3,
        )
    elif name == "opt-2.7b":
        config = OPTConfig(
            max_seq_len=2048, num_hidden_layers=32, n_head=32,
            hidden_size=2560, input_dim=2560, ffn_embed_dim=2560 * 4,
            version=3,
        )
    elif name == "opt-6.7b":
        config = OPTConfig(
            max_seq_len=2048, num_hidden_layers=32, n_head=32,
            hidden_size=4096, input_dim=4096, ffn_embed_dim=4096 * 4,
            version=3,
        )
    elif name == "opt-30b":
        config = OPTConfig(
            max_seq_len=2048, num_hidden_layers=48, n_head=56,
            hidden_size=7168, input_dim=7168, ffn_embed_dim=7168 * 4,
            version=3,
        )
    elif name == "opt-66b":
        config = OPTConfig(
            max_seq_len=2048, num_hidden_layers=64, n_head=72,
            hidden_size=9216, input_dim=9216, ffn_embed_dim=9216 * 4,
            version=3,
        )
    elif name == "opt-175b":
        config = OPTConfig(
            max_seq_len=2048, num_hidden_layers=96, n_head=96,
            hidden_size=12288, input_dim=12288, ffn_embed_dim=12288 * 4,
            version=3,
        )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)


def estimate_throughput(model_name: str, gbs: int, seq_len: int, num_device: int, latency: float):
    config = get_config(model_name)
    n_layer = config.num_hidden_layers
    h = config.hidden_size

    throughput_tflops = alpa.util.compute_gpt_tflops(
        batch_size=gbs,
        seq_len=seq_len,
        num_layers=n_layer,
        hidden_size=h,
        vocab_size=50272,
        num_gpus=num_device,
        latency=latency)
    print(f"Model {model_name}, gbs: {gbs}, seq_len: {seq_len}, Model TFlops: {throughput_tflops}, HW TFlops: {throughput_tflops * 4 / 3}..")


estimate_throughput("opt-175b", 1536, 2048, 128, 207)