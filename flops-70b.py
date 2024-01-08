def print_old_tflops():
    # flops calculator
    hidden_size: int = 8192
    num_layers: int = 80
    vocab_size: int = 43176

    checkpoint_activations_factor = 4
    batch_size: int = 1024

    seq_len: int = 4096

    flops_per_iteration: float = (24 * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))

    print(f"FLOPs per iteration (old): {flops_per_iteration}")


def print_new_tflops() -> None:
    hidden_size: int = 8192
    num_layers: int = 80
    vocab_size: int = 43176

    activation_function_factor = 4 + 2  # SWiGLU (upscaling + down scaling)
    batch_size: int = 1024
    seq_len: int = 4096

    intermediate_size: int = 28672
    selective_recompute_factor = 2
    checkpoint_activations_factor = 3

    num_query_groups = 8

    # 2: post-attention linear projection
    # 2 * 3: Key, Query, and Value transformation
    # / args.num_query_groups : GQA: Grouped Query Attention (default: num_query_groups=1)
    flops_per_iteration: float = checkpoint_activations_factor * ((
        (2 + (2 * 3) + activation_function_factor * (intermediate_size / hidden_size)) * batch_size * seq_len * num_layers * (hidden_size**2)
    ) + (
        ((  # Attention matrix & attention over values
            4 * batch_size * (seq_len ** 2) * hidden_size
        ) / num_query_groups * selective_recompute_factor
        ) +  # noqa: W504
        # lm-head: logit layer
        2 * batch_size * seq_len * hidden_size * vocab_size)
    )

    print(f"FLOPs per iteration (new): {flops_per_iteration}")


if __name__ == "__main__":
    print_old_tflops()
    print_new_tflops()
