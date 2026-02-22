#!/usr/bin/env python3
"""Run CostDraftTree CUDA score+update kernels and dump an update-state golden case."""

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


def _make_initial_sort_scores(
    batch_size: int,
    max_verify_num: int,
    verify_num: int,
    cumu_count: int,
    device: torch.device,
) -> torch.Tensor:
    sort_scores = torch.full(
        (batch_size, max_verify_num), -1234.0, dtype=torch.float32, device=device
    )
    ws0 = min(verify_num, cumu_count)
    if ws0 <= 0:
        return sort_scores
    for b in range(batch_size):
        base = 2.5 - float(b) * 0.1
        vals = [base - float(i) * 0.01 for i in range(ws0)]
        sort_scores[b, :ws0] = torch.tensor(vals, dtype=torch.float32, device=device)
    return sort_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("cost_draft_tree_update_case.txt"))
    parser.add_argument("--kernel-src", type=Path, default=_default_kernel_src())
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--node-top-k", type=int, default=8)
    parser.add_argument("--tree-width", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--input-count", type=int, default=17)
    parser.add_argument("--cumu-count", type=int, default=21)
    parser.add_argument("--verify-num", type=int, default=64)
    parser.add_argument("--curr-depth", type=int, default=3)
    parser.add_argument("--max-input-size", type=int, default=64)
    parser.add_argument("--max-node-count", type=int, default=1024)
    parser.add_argument("--max-verify-num", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--use-hot-token-id",
        action="store_true",
        help="Enable token remap in score stage before feeding update stage.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the CUDA-backed update dump.")

    total_topk = args.tree_width * args.node_top_k
    if total_topk <= 0 or total_topk > 64:
        raise ValueError(f"tree_width * node_top_k must be in [1, 64], got {total_topk}")

    if args.max_input_size < args.input_count - 1:
        raise ValueError("max_input_size must be >= input_count - 1")

    kernel_src = args.kernel_src.resolve()
    if not kernel_src.exists():
        raise FileNotFoundError(f"kernel source not found: {kernel_src}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Loading extension from: {kernel_src}")
    kernel_mod = load(
        name="cost_draft_tree_kernel_capture_update",
        sources=[str(kernel_src)],
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )

    device = torch.device("cuda")

    # Inputs for score stage; score output feeds update stage.
    topk_probas = torch.rand((args.batch_size, total_topk), device=device, dtype=torch.float32)
    topk_tokens = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, total_topk),
        device=device,
        dtype=torch.int64,
    )
    topk_tokens_input = topk_tokens.clone()
    last_layer_scores = torch.rand(
        (args.batch_size, args.tree_width), device=device, dtype=torch.float32
    )
    input_hidden_states = torch.randn(
        (args.batch_size, args.tree_width, args.hidden_size), device=device, dtype=torch.float32
    )

    hot_token_id = (
        (torch.arange(args.vocab_size, device=device, dtype=torch.int64) * 7 + 3)
        % args.vocab_size
    )

    curr_layer_scores = torch.empty_like(topk_probas)
    sort_layer_scores = torch.empty_like(topk_probas)
    sort_layer_indices = torch.empty_like(topk_tokens)
    cache_topk_indices = torch.empty(
        (args.batch_size, args.node_top_k), device=device, dtype=torch.int64
    )
    parent_indices_in_layer = torch.empty(
        (args.batch_size, args.node_top_k), device=device, dtype=torch.int64
    )
    output_hidden_states = torch.empty(
        (args.batch_size, args.node_top_k, args.hidden_size), device=device, dtype=torch.float32
    )

    kernel_mod.draft_tree_layer_score_index_gen_op(
        topk_probas,
        topk_tokens,
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

    topk_indexs_prev = torch.empty(
        (args.batch_size, args.tree_width), device=device, dtype=torch.int64
    )
    for b in range(args.batch_size):
        for i in range(args.tree_width):
            # Keep parent ids unique within a batch to avoid ambiguous next-pointer races.
            topk_indexs_prev[b, i] = (b * args.tree_width + i) % max(1, args.cumu_count)

    input_tree_mask = (
        torch.randint(
            0,
            2,
            (args.batch_size, args.tree_width, args.input_count - 1),
            device=device,
            dtype=torch.int64,
        )
        != 0
    )

    node_n = args.batch_size * args.max_node_count
    out_n = args.batch_size * args.node_top_k
    work_n = args.batch_size * (args.max_verify_num + args.node_top_k)
    sort_n = args.batch_size * args.max_verify_num
    out_mask_n = args.batch_size * args.node_top_k * (args.max_input_size + 1)

    initial_cumu_tokens = torch.full((node_n,), -777, dtype=torch.int64, device=device)
    initial_cumu_scores = torch.full((node_n,), -3.0, dtype=torch.float32, device=device)
    initial_cumu_deltas = torch.full((node_n,), -1, dtype=torch.int64, device=device)
    initial_prev_indexs = torch.full((node_n,), -1, dtype=torch.int64, device=device)
    initial_next_indexs = torch.full((node_n,), -2, dtype=torch.int64, device=device)
    initial_side_indexs = torch.full((node_n,), -3, dtype=torch.int64, device=device)

    initial_output_scores = torch.full((out_n,), -999.0, dtype=torch.float32, device=device)
    initial_output_tokens = torch.full((out_n,), -999, dtype=torch.int64, device=device)

    initial_work_scores = torch.full((work_n,), -123.0, dtype=torch.float32, device=device)
    initial_sort_scores = _make_initial_sort_scores(
        args.batch_size,
        args.max_verify_num,
        args.verify_num,
        args.cumu_count,
        device,
    ).reshape(-1)

    initial_output_tree_mask = (
        (torch.arange(out_mask_n, device=device, dtype=torch.int64) % 3) == 0
    )

    # Mutable state fed to CUDA update op.
    cumu_tokens = initial_cumu_tokens.clone()
    cumu_scores = initial_cumu_scores.clone()
    cumu_deltas = initial_cumu_deltas.clone()
    prev_indexs = initial_prev_indexs.clone()
    next_indexs = initial_next_indexs.clone()
    side_indexs = initial_side_indexs.clone()
    output_scores = initial_output_scores.clone()
    output_tokens = initial_output_tokens.clone()
    work_scores = initial_work_scores.clone()
    sort_scores = initial_sort_scores.clone()
    output_tree_mask = initial_output_tree_mask.clone()

    kernel_mod.update_cumu_draft_state(
        topk_probas,
        topk_tokens,
        sort_layer_scores,
        sort_layer_indices,
        parent_indices_in_layer,
        topk_indexs_prev,
        input_tree_mask,
        args.batch_size,
        args.node_top_k,
        args.tree_width,
        args.input_count,
        args.cumu_count,
        args.verify_num,
        args.curr_depth,
        args.max_input_size,
        args.max_node_count,
        args.max_verify_num,
        cumu_tokens,
        cumu_scores,
        cumu_deltas,
        prev_indexs,
        next_indexs,
        side_indexs,
        output_scores,
        output_tokens,
        work_scores,
        sort_scores,
        output_tree_mask,
    )
    torch.cuda.synchronize()

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# cost_draft_tree update-state CUDA capture v2\n")
        _write_int_line(
            f,
            "meta",
            torch.tensor(
                [
                    args.batch_size,
                    args.node_top_k,
                    args.tree_width,
                    args.input_count,
                    args.cumu_count,
                    args.verify_num,
                    args.curr_depth,
                    args.max_input_size,
                    args.max_node_count,
                    args.max_verify_num,
                ],
                dtype=torch.int64,
            ),
        )

        _write_float_line(f, "topk_probas", topk_probas)
        _write_int_line(f, "topk_tokens", topk_tokens)
        _write_float_line(f, "sorted_scores", sort_layer_scores)
        _write_int_line(f, "sorted_indexs", sort_layer_indices)
        _write_int_line(f, "parent_indexs", parent_indices_in_layer)
        _write_int_line(f, "topk_indexs", topk_indexs_prev)
        _write_int_line(f, "input_tree_mask", input_tree_mask.to(torch.int64))

        _write_int_line(f, "initial_cumu_tokens", initial_cumu_tokens)
        _write_float_line(f, "initial_cumu_scores", initial_cumu_scores)
        _write_int_line(f, "initial_cumu_deltas", initial_cumu_deltas)
        _write_int_line(f, "initial_prev_indexs", initial_prev_indexs)
        _write_int_line(f, "initial_next_indexs", initial_next_indexs)
        _write_int_line(f, "initial_side_indexs", initial_side_indexs)
        _write_float_line(f, "initial_output_scores", initial_output_scores)
        _write_int_line(f, "initial_output_tokens", initial_output_tokens)
        _write_float_line(f, "initial_work_scores", initial_work_scores)
        _write_float_line(f, "initial_sort_scores", initial_sort_scores)
        _write_int_line(f, "initial_output_tree_mask", initial_output_tree_mask.to(torch.int64))

        _write_int_line(f, "expected_cumu_tokens", cumu_tokens)
        _write_float_line(f, "expected_cumu_scores", cumu_scores)
        _write_int_line(f, "expected_cumu_deltas", cumu_deltas)
        _write_int_line(f, "expected_prev_indexs", prev_indexs)
        _write_int_line(f, "expected_next_indexs", next_indexs)
        _write_int_line(f, "expected_side_indexs", side_indexs)
        _write_float_line(f, "expected_output_scores", output_scores)
        _write_int_line(f, "expected_output_tokens", output_tokens)
        _write_float_line(f, "expected_work_scores", work_scores)
        _write_float_line(f, "expected_sort_scores", sort_scores)
        _write_int_line(f, "expected_output_tree_mask", output_tree_mask.to(torch.int64))

        # Additional traceability fields; ignored by current HLS parser but useful for debug.
        _write_float_line(f, "score_curr_layer_scores", curr_layer_scores)
        _write_float_line(f, "score_output_hidden_states", output_hidden_states)
        _write_int_line(f, "topk_tokens_sampling_input", topk_tokens_input)

    print(f"Wrote update-state case: {out_path}")
    print("Use this to run HLS check:")
    print(
        "  g++ -std=c++17 -O2 -I. cost_draft_tree_update_tb.cpp -o /tmp/cdt_update_tb "
        f"&& /tmp/cdt_update_tb --case-file {out_path}"
    )


if __name__ == "__main__":
    main()
