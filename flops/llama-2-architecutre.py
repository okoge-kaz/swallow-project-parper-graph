import argparse

parser = argparse.ArgumentParser(description='Llama 2 architecture')
parser.add_argument("--hidden-size", type=int)
parser.add_argument("--num-layers", type=int)
parser.add_argument("--num-attention-heads", type=int)
parser.add_argument("--vocab-size", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--seq-len", type=int)
parser.add_argument("--intermediate-size", type=int)
parser.add_argument("--num-query-groups", type=int)
parser.add_argument("--iteration", type=int)

args = parser.parse_args()

hidden_size: int = args.hidden_size
num_layers: int = args.num_layers
num_attention_heads: int = args.num_attention_heads
vocab_size: int = args.vocab_size
batch_size: int = args.batch_size
seq_len: int = args.seq_len
intermediate_size: int = args.intermediate_size
num_query_groups = args.num_query_groups

activation_function_factor = 4 + 2  # SWiGLU (upscaling + down scaling)
selective_recompute_factor = 2
checkpoint_activations_factor = 3

gqa_grouped_query = num_attention_heads // num_query_groups

flops_per_iteration: float = checkpoint_activations_factor * ((
    (2 + (2 * 3) + activation_function_factor * (intermediate_size / hidden_size)) * batch_size * seq_len * num_layers * (hidden_size**2)
) + (
    ((  # Attention matrix & attention over values
        4 * batch_size * (seq_len ** 2) * hidden_size
    ) / gqa_grouped_query * selective_recompute_factor
    ) +  # noqa: W504
    # lm-head: logit layer
    2 * batch_size * seq_len * hidden_size * vocab_size)
)

total_flops = flops_per_iteration * args.iteration

print(
    f"FLOPS: {total_flops}",
)
