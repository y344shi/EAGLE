#!/usr/bin/env python3
"""Run CostDraftTree CUDA score kernel once and dump a golden case for HLS comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


DEFAULT_REL_KERNEL = (
    "sglang-eagle4/sglang-v0.5.6/python/sglang/srt/speculative/cost_draft_tree_kernel.cu"
)


def _default_kernel_src() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    return repo_root / DEFAULT_REL_KERNEL


def _write_float_line(f, name: str, tensor: torch.Tensor) -> None:
    flat = tensor.detach().cpu().reshape(-1).to(torch.float32)
    values = " ".join(f"{float(x):.17g}" for x in flat.tolist())
    f.write(f"{name} {flat.numel()} {values}\n")


def _write_int_line(f, name: str, tensor: torch.Tensor) -> None:
    flat = tensor.detach().cpu().reshape(-1).to(torch.int64)
    values = " ".join(str(int(x)) for x in flat.tolist())
    f.write(f"{name} {flat.numel()} {values}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("cost_draft_tree_score_case.txt"))
    parser.add_argument("--kernel-src", type=Path, default=_default_kernel_src())
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tree-width", type=int, default=8)
    parser.add_argument("--node-top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--cumu-count", type=int, default=17)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument(
        "--use-hot-token-id",
        action="store_true",
        help="Apply hot-token remap inside CUDA kernel and dump remapped token tensors.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the CUDA kernel dump.")

    total_topk = args.tree_width * args.node_top_k
    if total_topk <= 0 or total_topk > 64:
        raise ValueError(f"tree_width * node_top_k must be in [1, 64], got {total_topk}")

    if args.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")

    kernel_src = args.kernel_src.resolve()
    if not kernel_src.exists():
        raise FileNotFoundError(f"kernel source not found: {kernel_src}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Loading extension from: {kernel_src}")
    kernel_mod = load(
        name="cost_draft_tree_kernel_capture",
        sources=[str(kernel_src)],
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )

    device = torch.device("cuda")

    topk_probas_sampling = torch.rand(
        (args.batch_size, total_topk), device=device, dtype=torch.float32
    )
    topk_tokens_sampling = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, total_topk),
        device=device,
        dtype=torch.int64,
    )
    topk_tokens_sampling_input = topk_tokens_sampling.clone()
    last_layer_scores = torch.rand(
        (args.batch_size, args.tree_width), device=device, dtype=torch.float32
    )

    curr_layer_scores = torch.empty_like(topk_probas_sampling)
    sort_layer_scores = torch.empty_like(topk_probas_sampling)
    sort_layer_indices = torch.empty(
        (args.batch_size, total_topk), device=device, dtype=torch.int64
    )
    cache_topk_indices = torch.empty(
        (args.batch_size, args.node_top_k), device=device, dtype=torch.int64
    )
    parent_indices_in_layer = torch.empty(
        (args.batch_size, args.node_top_k), device=device, dtype=torch.int64
    )

    input_hidden_states = torch.randn(
        (args.batch_size, args.tree_width, args.hidden_size),
        device=device,
        dtype=torch.float32,
    )
    output_hidden_states = torch.empty(
        (args.batch_size, args.node_top_k, args.hidden_size),
        device=device,
        dtype=torch.float32,
    )

    # Non-identity affine map keeps remap validation meaningful when enabled.
    hot_token_id = ((torch.arange(args.vocab_size, device=device, dtype=torch.int64) * 7) + 3) % args.vocab_size

    kernel_mod.draft_tree_layer_score_index_gen_op(
        topk_probas_sampling,
        topk_tokens_sampling,
        last_layer_scores,
        curr_layer_scores,
        sort_layer_scores,
        sort_layer_indices,
        cache_topk_indices,
        parent_indices_in_layer,
        input_hidden_states,
        output_hidden_states,
        hot_token_id,
        args.batch_size,
        args.node_top_k,
        args.tree_width,
        args.cumu_count,
        args.use_hot_token_id,
    )
    torch.cuda.synchronize()

    expected_output_tokens = torch.gather(
        topk_tokens_sampling, dim=1, index=sort_layer_indices[:, : args.node_top_k]
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# cost_draft_tree score kernel capture v2\n")
        f.write(
            f"meta {args.batch_size} {args.node_top_k} {args.tree_width} {args.hidden_size} {args.cumu_count}\n"
        )
        _write_int_line(
            f,
            "use_hot_token_id",
            torch.tensor([1 if args.use_hot_token_id else 0], dtype=torch.int64),
        )
        _write_float_line(f, "topk_probas_sampling", topk_probas_sampling)
        _write_int_line(f, "topk_tokens_sampling", topk_tokens_sampling_input)
        _write_float_line(f, "last_layer_scores", last_layer_scores)
        _write_float_line(f, "input_hidden_states", input_hidden_states)
        if args.use_hot_token_id:
            _write_int_line(f, "hot_token_id", hot_token_id)
        _write_float_line(f, "expected_curr_layer_scores", curr_layer_scores)
        _write_float_line(f, "expected_sort_layer_scores", sort_layer_scores)
        _write_int_line(f, "expected_sort_layer_indices", sort_layer_indices)
        _write_int_line(f, "expected_cache_topk_indices", cache_topk_indices)
        _write_int_line(
            f, "expected_parent_indices_in_layer", parent_indices_in_layer
        )
        _write_float_line(f, "expected_output_hidden_states", output_hidden_states)
        _write_int_line(f, "expected_topk_tokens_sampling", topk_tokens_sampling)
        _write_int_line(f, "expected_output_tokens", expected_output_tokens)

    print(f"Wrote golden case: {output_path}")
    print("Use this to run HLS check:")
    print(
        "  g++ -std=c++17 -O2 -I. cost_draft_tree_score_tb.cpp -o /tmp/cdt_score_tb "
        f"&& /tmp/cdt_score_tb {output_path}"
    )


if __name__ == "__main__":
    main()
