import argparse

parser = argparse.ArgumentParser(description="Llama 2 architecture")
parser.add_argument("--hidden-size", type=int)
parser.add_argument("--num-layers", type=int)
parser.add_argument("--num-attention-heads", type=int)
parser.add_argument("--vocab-size", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--seq-len", type=int)
parser.add_argument("--intermediate-size", type=int)
parser.add_argument("--num-query-groups", type=int)
parser.add_argument("--iteration", type=int)
parser.add_argument("--num-experts", type=int, default=None)
parser.add_argument("--moe-router-topk", type=int, default=1)
parser.add_argument("--swiglu", action="store_true", help="Use SWiGLU activation function")

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

# Attention projection size.
kv_channels = hidden_size // num_attention_heads
query_projection_size = kv_channels * args.num_attention_heads
query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size

# MoE.
num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
gated_linear_multiplier = 3 / 2 if args.swiglu else 1

flops_per_iteration: float = (
    12
    * batch_size
    * seq_len
    * num_layers
    * hidden_size
    * hidden_size
    * (
        # Attention.
        (
            (1 + (num_query_groups / args.num_attention_heads) + (args.seq_len / args.hidden_size))
            * query_projection_to_hidden_size_ratio
        )
        # MLP.
        + ((args.intermediate_size / args.hidden_size) * num_experts_routed_to * gated_linear_multiplier)
        # Logit.
        + (args.vocab_size / (2 * args.num_layers * args.hidden_size))
    )
)

total_flops = flops_per_iteration * args.iteration

print(
    f"FLOPS: {total_flops}",
)
