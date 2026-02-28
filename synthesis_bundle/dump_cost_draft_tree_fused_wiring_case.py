#!/usr/bin/env python3
"""Dump fused wiring case with CUDA-backed score/update expected tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


DEFAULT_REL_KERNEL = (
    "sglang-eagle4/sglang-v0.5.6/python/sglang/srt/speculative/cost_draft_tree_kernel.cu"
)
MAX_CONTROLLER_DEPTH = 64


def _default_kernel_src() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    return repo_root / DEFAULT_REL_KERNEL


def _write_line(f, key: str, values) -> None:
    vals = list(values)
    if vals:
        payload = " ".join(str(int(v)) for v in vals)
        f.write(f"{key} {len(vals)} {payload}\n")
    else:
        f.write(f"{key} 0\n")


def _write_float_line_tensor(f, key: str, tensor: torch.Tensor) -> None:
    flat = tensor.detach().cpu().reshape(-1).to(torch.float32)
    payload = " ".join(f"{float(v):.17g}" for v in flat.tolist())
    f.write(f"{key} {flat.numel()} {payload}\n")


def _write_int_line_tensor(f, key: str, tensor: torch.Tensor) -> None:
    flat = tensor.detach().cpu().reshape(-1).to(torch.int64)
    payload = " ".join(str(int(v)) for v in flat.tolist())
    f.write(f"{key} {flat.numel()} {payload}\n")


def seed_controller_state(cfg: dict) -> dict:
    batch_size = cfg["batch_size"]
    parent_width = cfg["parent_width"]
    max_tree_width = cfg["max_tree_width"]
    max_node_count = cfg["max_node_count"]

    node_count = [0] * batch_size
    frontier = [-1] * (batch_size * max_tree_width)
    node_token = [-1] * (batch_size * max_node_count)
    node_parent = [-1] * (batch_size * max_node_count)
    node_first_child = [-1] * (batch_size * max_node_count)
    node_last_child = [-1] * (batch_size * max_node_count)
    node_next_sibling = [-1] * (batch_size * max_node_count)
    node_depth = [-1] * (batch_size * max_node_count)
    node_cache = [-1] * (batch_size * max_node_count)

    for b in range(batch_size):
        base = b * max_node_count
        prev_seed_id = -1
        for i in range(parent_width):
            nid = node_count[b]
            node_count[b] += 1
            frontier[b * max_tree_width + i] = nid
            node_token[base + nid] = 100 + b * 10 + i
            node_parent[base + nid] = -1
            node_first_child[base + nid] = -1
            node_last_child[base + nid] = -1
            node_next_sibling[base + nid] = -1
            node_depth[base + nid] = 0
            node_cache[base + nid] = 1000 + b * 100 + i
            if prev_seed_id >= 0:
                node_next_sibling[base + prev_seed_id] = nid
            prev_seed_id = nid

    return {
        "controller_node_count": node_count,
        "controller_frontier_in": frontier,
        "controller_node_token_ids": node_token,
        "controller_node_parent_ids": node_parent,
        "controller_node_first_child_ids": node_first_child,
        "controller_node_last_child_ids": node_last_child,
        "controller_node_next_sibling_ids": node_next_sibling,
        "controller_node_depths": node_depth,
        "controller_node_cache_locs": node_cache,
    }


def seed_legacy_state(cfg: dict, device: torch.device) -> dict:
    batch_size = cfg["batch_size"]
    max_node_count = cfg["max_node_count"]
    node_top_k = cfg["node_top_k"]
    max_verify_num = cfg["max_verify_num"]
    verify_num = cfg["verify_num"]
    cumu_count = cfg["cumu_count"]
    max_input_size = cfg["max_input_size"]

    node_n = batch_size * max_node_count
    out_n = batch_size * node_top_k
    work_n = batch_size * (max_verify_num + node_top_k)
    sort_n = batch_size * max_verify_num
    out_mask_n = batch_size * node_top_k * (max_input_size + 1)

    legacy = {
        "legacy_cumu_tokens": torch.full((node_n,), -999, dtype=torch.int64, device=device),
        "legacy_cumu_scores": torch.full((node_n,), -5.0, dtype=torch.float32, device=device),
        "legacy_cumu_deltas": torch.full((node_n,), -1, dtype=torch.int64, device=device),
        "legacy_prev_indexs": torch.full((node_n,), -1, dtype=torch.int64, device=device),
        "legacy_next_indexs": torch.full((node_n,), -1, dtype=torch.int64, device=device),
        "legacy_side_indexs": torch.full((node_n,), -1, dtype=torch.int64, device=device),
        "legacy_output_scores": torch.full((out_n,), -7.0, dtype=torch.float32, device=device),
        "legacy_output_tokens": torch.full((out_n,), -3, dtype=torch.int64, device=device),
        "legacy_work_scores": torch.full((work_n,), -9.0, dtype=torch.float32, device=device),
        "legacy_sort_scores": torch.full((sort_n,), -11.0, dtype=torch.float32, device=device),
        "legacy_output_tree_mask": torch.zeros((out_mask_n,), dtype=torch.bool, device=device),
    }

    ws0 = min(verify_num, cumu_count)
    for b in range(batch_size):
        base = 2.0 - 0.1 * b
        vals = [base - 0.01 * i for i in range(ws0)]
        legacy["legacy_sort_scores"][b * max_verify_num : b * max_verify_num + ws0] = torch.tensor(
            vals, dtype=torch.float32, device=device
        )

    return legacy


def _clamp(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def controller_expand_frontier(
    cfg: dict,
    controller_state: dict,
    parent_slots: list[int],
    child_tokens: list[int],
    child_cache_locs: list[int],
) -> tuple[list[int], dict]:
    batch_size = cfg["batch_size"]
    parent_width = _clamp(cfg["parent_width"], 1, cfg["max_tree_width"])
    next_tree_width = _clamp(cfg["next_tree_width"], 0, cfg["max_tree_width"])
    max_tree_width = cfg["max_tree_width"]
    max_node_count = cfg["max_node_count"]

    node_count = controller_state["controller_node_count"]
    frontier_in = controller_state["controller_frontier_in"]
    node_token = controller_state["controller_node_token_ids"]
    node_parent = controller_state["controller_node_parent_ids"]
    node_first_child = controller_state["controller_node_first_child_ids"]
    node_last_child = controller_state["controller_node_last_child_ids"]
    node_next_sibling = controller_state["controller_node_next_sibling_ids"]
    node_depth = controller_state["controller_node_depths"]
    node_cache = controller_state["controller_node_cache_locs"]

    frontier_out = [-1] * (batch_size * max_tree_width)

    for b in range(batch_size):
        base = b * max_node_count
        for i in range(next_tree_width):
            in_idx = b * next_tree_width + i
            slot = _clamp(parent_slots[in_idx], 0, parent_width - 1)
            parent_nid = frontier_in[b * max_tree_width + slot]
            if parent_nid < 0 or parent_nid >= max_node_count:
                parent_nid = -1

            nid = node_count[b]
            if nid >= max_node_count:
                frontier_out[b * max_tree_width + i] = -1
                continue

            node_count[b] = nid + 1
            frontier_out[b * max_tree_width + i] = nid

            node_token[base + nid] = child_tokens[in_idx]
            node_parent[base + nid] = parent_nid
            node_first_child[base + nid] = -1
            node_last_child[base + nid] = -1
            node_next_sibling[base + nid] = -1
            node_cache[base + nid] = child_cache_locs[in_idx]

            depth = 0
            if parent_nid >= 0:
                depth = node_depth[base + parent_nid] + 1
            node_depth[base + nid] = depth

            if parent_nid >= 0:
                pidx = base + parent_nid
                last_child = node_last_child[pidx]
                if last_child < 0:
                    node_first_child[pidx] = nid
                elif last_child < max_node_count:
                    node_next_sibling[base + last_child] = nid
                node_last_child[pidx] = nid

        for i in range(next_tree_width, max_tree_width):
            frontier_out[b * max_tree_width + i] = -1

    expected_state = {
        "expected_controller_node_count": list(node_count),
        "expected_controller_node_token_ids": list(node_token),
        "expected_controller_node_parent_ids": list(node_parent),
        "expected_controller_node_first_child_ids": list(node_first_child),
        "expected_controller_node_last_child_ids": list(node_last_child),
        "expected_controller_node_next_sibling_ids": list(node_next_sibling),
        "expected_controller_node_depths": list(node_depth),
        "expected_controller_node_cache_locs": list(node_cache),
    }

    return frontier_out, expected_state


def controller_build_parent_visible_kv(
    cfg: dict,
    frontier_node_ids: list[int],
    prefix_kv_locs: list[int],
    prefix_lens: list[int],
    node_parent_ids: list[int],
    node_cache_locs: list[int],
) -> tuple[list[int], list[int], list[int]]:
    batch_size = cfg["batch_size"]
    width = _clamp(cfg["next_tree_width"], 0, cfg["max_tree_width"])
    max_tree_width = cfg["max_tree_width"]
    max_prefix_len = cfg["max_prefix_len"]
    max_input_size = cfg["max_input_size"]
    max_node_count = cfg["max_node_count"]

    kv_indices = [-1] * (batch_size * max_tree_width * max_input_size)
    kv_mask = [0] * (batch_size * max_tree_width * max_input_size)
    kv_lens = [0] * (batch_size * max_tree_width)

    for b in range(batch_size):
        base = b * max_node_count
        pfx_len = _clamp(prefix_lens[b], 0, max_prefix_len)
        for q in range(max_tree_width):
            if q >= width:
                continue
            out_base = (b * max_tree_width + q) * max_input_size
            nid = frontier_node_ids[b * max_tree_width + q]
            if nid < 0 or nid >= max_node_count:
                continue

            chain: list[int] = []
            for _ in range(MAX_CONTROLLER_DEPTH):
                if nid < 0 or nid >= max_node_count:
                    break
                chain.append(nid)
                nid = node_parent_ids[base + nid]

            out_len = 0
            for i in range(pfx_len):
                if out_len >= max_input_size:
                    break
                kv_indices[out_base + out_len] = prefix_kv_locs[b * max_prefix_len + i]
                kv_mask[out_base + out_len] = 1
                out_len += 1

            for node_id in reversed(chain):
                if out_len >= max_input_size:
                    break
                kv_indices[out_base + out_len] = node_cache_locs[base + node_id]
                kv_mask[out_base + out_len] = 1
                out_len += 1

            kv_lens[b * max_tree_width + q] = out_len

    return kv_indices, kv_mask, kv_lens


def controller_export_frontier(
    cfg: dict,
    frontier_node_ids: list[int],
    node_token_ids: list[int],
    node_parent_ids: list[int],
    node_depths: list[int],
    node_cache_locs: list[int],
) -> tuple[list[int], list[int], list[int], list[int]]:
    batch_size = cfg["batch_size"]
    width = _clamp(cfg["next_tree_width"], 0, cfg["max_tree_width"])
    max_tree_width = cfg["max_tree_width"]
    max_node_count = cfg["max_node_count"]

    out_tokens = [-1] * (batch_size * max_tree_width)
    out_parents = [-1] * (batch_size * max_tree_width)
    out_depths = [-1] * (batch_size * max_tree_width)
    out_cache = [-1] * (batch_size * max_tree_width)

    for b in range(batch_size):
        base = b * max_node_count
        for i in range(max_tree_width):
            if i >= width:
                continue
            nid = frontier_node_ids[b * max_tree_width + i]
            if nid < 0 or nid >= max_node_count:
                continue
            out_tokens[b * max_tree_width + i] = node_token_ids[base + nid]
            out_parents[b * max_tree_width + i] = node_parent_ids[base + nid]
            out_depths[b * max_tree_width + i] = node_depths[base + nid]
            out_cache[b * max_tree_width + i] = node_cache_locs[base + nid]

    return out_tokens, out_parents, out_depths, out_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("cost_draft_tree_fused_wiring_case.txt"))
    parser.add_argument("--kernel-src", type=Path, default=_default_kernel_src())
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate CUDA-backed fused wiring case.")

    kernel_src = args.kernel_src.resolve()
    if not kernel_src.exists():
        raise FileNotFoundError(f"kernel source not found: {kernel_src}")

    cfg = {
        "batch_size": 2,
        "node_top_k": 4,
        "tree_width": 4,
        "hidden_size": 16,
        "cumu_count": 8,
        "input_count": 6,
        "verify_num": 8,
        "curr_depth": 2,
        "max_input_size": 16,
        "max_node_count": 128,
        "max_verify_num": 16,
        "max_tree_width": 4,
        "max_prefix_len": 8,
        "parent_width": 4,
        "next_tree_width": 4,
        "hot_vocab_size": 512,
        "use_hot_token_id": 1,
    }

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    print(f"Loading extension from: {kernel_src}")
    kernel_mod = load(
        name="cost_draft_tree_kernel_capture_fused",
        sources=[str(kernel_src)],
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )

    total_topk = cfg["tree_width"] * cfg["node_top_k"]

    topk_probas = torch.rand((cfg["batch_size"], total_topk), device=device, dtype=torch.float32)
    topk_tokens = torch.randint(
        0,
        cfg["hot_vocab_size"],
        (cfg["batch_size"], total_topk),
        device=device,
        dtype=torch.int64,
    )
    topk_tokens_input = topk_tokens.clone()
    last_layer_scores = torch.rand(
        (cfg["batch_size"], cfg["tree_width"]), device=device, dtype=torch.float32
    )
    input_hidden_states = torch.randn(
        (cfg["batch_size"], cfg["tree_width"], cfg["hidden_size"]),
        device=device,
        dtype=torch.float32,
    )

    hot_token_id = (
        (torch.arange(cfg["hot_vocab_size"], device=device, dtype=torch.int64) * 7 + 3)
        % cfg["hot_vocab_size"]
    )

    topk_indexs_prev = torch.empty(
        (cfg["batch_size"], cfg["tree_width"]), device=device, dtype=torch.int64
    )
    for b in range(cfg["batch_size"]):
        for i in range(cfg["tree_width"]):
            topk_indexs_prev[b, i] = i

    input_tree_mask = (
        torch.arange(
            cfg["batch_size"] * cfg["tree_width"] * (cfg["input_count"] - 1),
            device=device,
            dtype=torch.int64,
        )
        .reshape(cfg["batch_size"], cfg["tree_width"], cfg["input_count"] - 1)
        % 2
        == 0
    )

    prefix_kv_locs = torch.full(
        (cfg["batch_size"], cfg["max_prefix_len"]), -1, device=device, dtype=torch.int32
    )
    prefix_lens = torch.full((cfg["batch_size"],), 3, device=device, dtype=torch.int32)
    for b in range(cfg["batch_size"]):
        prefix_kv_locs[b, 0] = 11 + b * 10
        prefix_kv_locs[b, 1] = 12 + b * 10
        prefix_kv_locs[b, 2] = 13 + b * 10

    selected_cache_locs = torch.empty(
        (cfg["batch_size"], cfg["node_top_k"]), device=device, dtype=torch.int32
    )
    for b in range(cfg["batch_size"]):
        for i in range(cfg["node_top_k"]):
            selected_cache_locs[b, i] = 2100 + b * 100 + i

    # Score stage outputs.
    curr_layer_scores = torch.empty_like(topk_probas)
    sort_layer_scores = torch.empty_like(topk_probas)
    sort_layer_indices = torch.empty_like(topk_tokens)
    cache_topk_indices = torch.empty(
        (cfg["batch_size"], cfg["node_top_k"]), device=device, dtype=torch.int64
    )
    parent_indices_in_layer = torch.empty(
        (cfg["batch_size"], cfg["node_top_k"]), device=device, dtype=torch.int64
    )
    output_hidden_states = torch.empty(
        (cfg["batch_size"], cfg["node_top_k"], cfg["hidden_size"]),
        device=device,
        dtype=torch.float32,
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
        cfg["batch_size"],
        cfg["node_top_k"],
        cfg["tree_width"],
        cfg["cumu_count"],
        bool(cfg["use_hot_token_id"]),
    )

    selected_output_tokens = torch.gather(
        topk_tokens, dim=1, index=sort_layer_indices[:, : cfg["node_top_k"]]
    )

    # Update stage outputs (legacy state).
    legacy_initial = seed_legacy_state(cfg, device)
    legacy_runtime = {k: v.clone() for k, v in legacy_initial.items()}

    kernel_mod.update_cumu_draft_state(
        topk_probas,
        topk_tokens,
        sort_layer_scores,
        sort_layer_indices,
        parent_indices_in_layer,
        topk_indexs_prev,
        input_tree_mask,
        cfg["batch_size"],
        cfg["node_top_k"],
        cfg["tree_width"],
        cfg["input_count"],
        cfg["cumu_count"],
        cfg["verify_num"],
        cfg["curr_depth"],
        cfg["max_input_size"],
        cfg["max_node_count"],
        cfg["max_verify_num"],
        legacy_runtime["legacy_cumu_tokens"],
        legacy_runtime["legacy_cumu_scores"],
        legacy_runtime["legacy_cumu_deltas"],
        legacy_runtime["legacy_prev_indexs"],
        legacy_runtime["legacy_next_indexs"],
        legacy_runtime["legacy_side_indexs"],
        legacy_runtime["legacy_output_scores"],
        legacy_runtime["legacy_output_tokens"],
        legacy_runtime["legacy_work_scores"],
        legacy_runtime["legacy_sort_scores"],
        legacy_runtime["legacy_output_tree_mask"],
    )
    torch.cuda.synchronize()

    # Controller stage expected outputs from deterministic Python implementation.
    controller_init = seed_controller_state(cfg)
    controller_work = {k: list(v) for k, v in controller_init.items()}

    frontier_out, controller_state_expected = controller_expand_frontier(
        cfg,
        controller_work,
        parent_indices_in_layer.detach().cpu().reshape(-1).to(torch.int64).tolist(),
        selected_output_tokens.detach().cpu().reshape(-1).to(torch.int64).tolist(),
        selected_cache_locs.detach().cpu().reshape(-1).to(torch.int64).tolist(),
    )

    kv_indices, kv_mask, kv_lens = controller_build_parent_visible_kv(
        cfg,
        frontier_out,
        prefix_kv_locs.detach().cpu().reshape(-1).to(torch.int64).tolist(),
        prefix_lens.detach().cpu().reshape(-1).to(torch.int64).tolist(),
        controller_state_expected["expected_controller_node_parent_ids"],
        controller_state_expected["expected_controller_node_cache_locs"],
    )

    frontier_tokens, frontier_parent_ids, frontier_depths, frontier_cache_locs = (
        controller_export_frontier(
            cfg,
            frontier_out,
            controller_state_expected["expected_controller_node_token_ids"],
            controller_state_expected["expected_controller_node_parent_ids"],
            controller_state_expected["expected_controller_node_depths"],
            controller_state_expected["expected_controller_node_cache_locs"],
        )
    )

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# cost_draft_tree fused wiring CUDA capture v2\n")
        _write_line(
            f,
            "meta",
            [
                cfg["batch_size"],
                cfg["node_top_k"],
                cfg["tree_width"],
                cfg["hidden_size"],
                cfg["cumu_count"],
                cfg["input_count"],
                cfg["verify_num"],
                cfg["curr_depth"],
                cfg["max_input_size"],
                cfg["max_node_count"],
                cfg["max_verify_num"],
                cfg["max_tree_width"],
                cfg["max_prefix_len"],
                cfg["parent_width"],
                cfg["next_tree_width"],
                cfg["hot_vocab_size"],
                cfg["use_hot_token_id"],
            ],
        )

        _write_float_line_tensor(f, "topk_probas_sampling", topk_probas)
        _write_int_line_tensor(f, "topk_tokens_sampling", topk_tokens_input)
        _write_float_line_tensor(f, "last_layer_scores", last_layer_scores)
        _write_float_line_tensor(f, "input_hidden_states", input_hidden_states)
        _write_int_line_tensor(f, "hot_token_id", hot_token_id)
        _write_int_line_tensor(f, "topk_indexs_prev", topk_indexs_prev)
        _write_int_line_tensor(f, "input_tree_mask", input_tree_mask.to(torch.int64))
        _write_int_line_tensor(f, "prefix_kv_locs", prefix_kv_locs.to(torch.int64))
        _write_line(f, "prefix_lens", prefix_lens.detach().cpu().tolist())
        _write_int_line_tensor(f, "selected_cache_locs", selected_cache_locs.to(torch.int64))

        # Controller initial state.
        for key in (
            "controller_node_count",
            "controller_frontier_in",
            "controller_node_token_ids",
            "controller_node_parent_ids",
            "controller_node_first_child_ids",
            "controller_node_last_child_ids",
            "controller_node_next_sibling_ids",
            "controller_node_depths",
            "controller_node_cache_locs",
        ):
            _write_line(f, key, controller_init[key])

        # Legacy initial state.
        _write_int_line_tensor(f, "legacy_cumu_tokens", legacy_initial["legacy_cumu_tokens"])
        _write_float_line_tensor(f, "legacy_cumu_scores", legacy_initial["legacy_cumu_scores"])
        _write_int_line_tensor(f, "legacy_cumu_deltas", legacy_initial["legacy_cumu_deltas"])
        _write_int_line_tensor(f, "legacy_prev_indexs", legacy_initial["legacy_prev_indexs"])
        _write_int_line_tensor(f, "legacy_next_indexs", legacy_initial["legacy_next_indexs"])
        _write_int_line_tensor(f, "legacy_side_indexs", legacy_initial["legacy_side_indexs"])
        _write_float_line_tensor(f, "legacy_output_scores", legacy_initial["legacy_output_scores"])
        _write_int_line_tensor(f, "legacy_output_tokens", legacy_initial["legacy_output_tokens"])
        _write_float_line_tensor(f, "legacy_work_scores", legacy_initial["legacy_work_scores"])
        _write_float_line_tensor(f, "legacy_sort_scores", legacy_initial["legacy_sort_scores"])
        _write_int_line_tensor(
            f,
            "legacy_output_tree_mask",
            legacy_initial["legacy_output_tree_mask"].to(torch.int64),
        )

        # Stage-1 expected (score).
        _write_float_line_tensor(f, "expected_dbg_curr_layer_scores", curr_layer_scores)
        _write_float_line_tensor(f, "expected_dbg_sort_layer_scores", sort_layer_scores)
        _write_int_line_tensor(f, "expected_dbg_sort_layer_indices", sort_layer_indices)
        _write_int_line_tensor(
            f, "expected_dbg_parent_indices_in_layer", parent_indices_in_layer
        )
        _write_int_line_tensor(f, "expected_dbg_remapped_topk_tokens", topk_tokens)
        _write_int_line_tensor(f, "expected_cache_topk_indices", cache_topk_indices)
        _write_float_line_tensor(f, "expected_output_hidden_states", output_hidden_states)

        # Stage-2 expected (update).
        _write_int_line_tensor(
            f, "expected_legacy_cumu_tokens", legacy_runtime["legacy_cumu_tokens"]
        )
        _write_float_line_tensor(
            f, "expected_legacy_cumu_scores", legacy_runtime["legacy_cumu_scores"]
        )
        _write_int_line_tensor(
            f, "expected_legacy_cumu_deltas", legacy_runtime["legacy_cumu_deltas"]
        )
        _write_int_line_tensor(
            f, "expected_legacy_prev_indexs", legacy_runtime["legacy_prev_indexs"]
        )
        _write_int_line_tensor(
            f, "expected_legacy_next_indexs", legacy_runtime["legacy_next_indexs"]
        )
        _write_int_line_tensor(
            f, "expected_legacy_side_indexs", legacy_runtime["legacy_side_indexs"]
        )
        _write_float_line_tensor(
            f, "expected_legacy_output_scores", legacy_runtime["legacy_output_scores"]
        )
        _write_int_line_tensor(
            f, "expected_legacy_output_tokens", legacy_runtime["legacy_output_tokens"]
        )
        _write_float_line_tensor(
            f, "expected_legacy_work_scores", legacy_runtime["legacy_work_scores"]
        )
        _write_float_line_tensor(
            f, "expected_legacy_sort_scores", legacy_runtime["legacy_sort_scores"]
        )
        _write_int_line_tensor(
            f,
            "expected_legacy_output_tree_mask",
            legacy_runtime["legacy_output_tree_mask"].to(torch.int64),
        )

        # Stage-3/4 expected (controller).
        _write_line(f, "expected_controller_frontier_out", frontier_out)
        for key in (
            "expected_controller_node_count",
            "expected_controller_node_token_ids",
            "expected_controller_node_parent_ids",
            "expected_controller_node_first_child_ids",
            "expected_controller_node_last_child_ids",
            "expected_controller_node_next_sibling_ids",
            "expected_controller_node_depths",
            "expected_controller_node_cache_locs",
        ):
            _write_line(f, key, controller_state_expected[key])

        _write_line(f, "expected_controller_kv_indices", kv_indices)
        _write_line(f, "expected_controller_kv_mask", kv_mask)
        _write_line(f, "expected_controller_kv_lens", kv_lens)
        _write_line(f, "expected_controller_frontier_tokens", frontier_tokens)
        _write_line(f, "expected_controller_frontier_parent_ids", frontier_parent_ids)
        _write_line(f, "expected_controller_frontier_depths", frontier_depths)
        _write_line(f, "expected_controller_frontier_cache_locs", frontier_cache_locs)

    print(f"Wrote fused wiring case: {out_path}")


if __name__ == "__main__":
    main()
