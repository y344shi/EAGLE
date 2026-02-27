#!/usr/bin/env python3
"""Compose a multilayer orchestrator TB case from existing captures with synthetic fallback.

This script is SM75-safe: it never compiles or runs CUDA kernels.
It reuses already captured CostDraftTree case files when present, and fills
missing tensors deterministically.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_SEARCH_DIRS = [
    Path("/home/y344shi/workspace/eagle4_adaptation/sglang-eagle4/capture/cases"),
    Path("/home/y344shi/workspace/eagle4_adaptation/capture/cases"),
    Path("hardware/EAGLE/eagle/hls_hw/synthesis_bundle"),
]


def _clamp(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _parse_key_count_file(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"invalid line in {path}: {line}")
            key = parts[0]
            try:
                count = int(parts[1])
            except ValueError as e:
                raise ValueError(f"invalid count for key {key} in {path}") from e
            payload = parts[2:]
            if len(payload) != count:
                raise ValueError(
                    f"payload size mismatch for key {key} in {path}: "
                    f"got={len(payload)} expected={count}"
                )
            out[key] = payload
    return out


def _get_ints(kv: Dict[str, List[str]], key: str, required: bool = True) -> List[int]:
    if key not in kv:
        if required:
            raise KeyError(f"missing key: {key}")
        return []
    return [int(x) for x in kv[key]]


def _get_i64s(kv: Dict[str, List[str]], key: str, required: bool = True) -> List[int]:
    return _get_ints(kv, key, required=required)


def _get_floats(kv: Dict[str, List[str]], key: str, required: bool = True) -> List[float]:
    if key not in kv:
        if required:
            raise KeyError(f"missing key: {key}")
        return []
    return [float(x) for x in kv[key]]


def _slice_batch(flat: List, batch_idx: int, batch_size: int, per_batch: int) -> List:
    if batch_size <= 0:
        return []
    start = batch_idx * per_batch
    end = start + per_batch
    if end > len(flat):
        return []
    return list(flat[start:end])


def _write_line(path_f, key: str, values: Iterable) -> None:
    vals = list(values)
    if not vals:
        path_f.write(f"{key} 0\n")
        return
    payload = " ".join(str(v) for v in vals)
    path_f.write(f"{key} {len(vals)} {payload}\n")


def _write_float_line(path_f, key: str, values: Iterable[float]) -> None:
    vals = [float(v) for v in values]
    if not vals:
        path_f.write(f"{key} 0\n")
        return
    payload = " ".join(f"{v:.17g}" for v in vals)
    path_f.write(f"{key} {len(vals)} {payload}\n")


def _find_case(search_dirs: List[Path], name: str) -> Optional[Path]:
    for base in search_dirs:
        p = base / name
        if p.exists() and p.is_file():
            return p
    return None


def _simulate_expected_scalars(
    tree_depth: int,
    node_top_k: int,
    init_tree_width: int,
    init_verify_num: int,
    init_cumu_count: int,
    max_tree_width: int,
    max_verify_num: int,
    max_node_count: int,
    policy_next_tree_width: List[int],
    policy_next_verify_num: List[int],
    policy_stop_signal: List[int],
) -> tuple[int, int, int, int, int]:
    curr_tree_width = _clamp(init_tree_width, 0, max_tree_width)
    curr_tree_width = _clamp(curr_tree_width, 0, node_top_k)
    curr_verify_num = _clamp(init_verify_num, 1, max_verify_num)
    curr_cumu_count = _clamp(init_cumu_count, 0, max_node_count)
    depth_done = 0
    stopped = 0

    # InitialLoop stage
    curr_cumu_count = min(max_node_count, curr_cumu_count + node_top_k)
    depth_done += 1

    next_tree_width = _clamp(policy_next_tree_width[0], 0, max_tree_width)
    next_tree_width = _clamp(next_tree_width, 0, node_top_k)
    next_verify_num = _clamp(policy_next_verify_num[0], 1, max_verify_num)
    stop_signal = 1 if policy_stop_signal[0] else 0
    curr_tree_width = next_tree_width
    curr_verify_num = next_verify_num

    if tree_depth <= 1 or stop_signal or next_tree_width <= 0:
        stopped = 1 if (stop_signal or next_tree_width <= 0) else 0
        return curr_tree_width, curr_verify_num, curr_cumu_count, depth_done, stopped

    for d in range(1, tree_depth):
        if curr_tree_width <= 0:
            stopped = 1
            break

        curr_cumu_count = min(max_node_count, curr_cumu_count + curr_tree_width * node_top_k)
        depth_done += 1

        next_tree_width = _clamp(policy_next_tree_width[d], 0, max_tree_width)
        next_tree_width = _clamp(next_tree_width, 0, node_top_k)
        next_verify_num = _clamp(policy_next_verify_num[d], 1, max_verify_num)
        stop_signal = 1 if policy_stop_signal[d] else 0

        if d + 1 >= tree_depth or stop_signal or next_tree_width <= 0:
            # Mirrors patched orchestrator finalize semantics.
            curr_tree_width = next_tree_width
            curr_verify_num = next_verify_num
            stopped = 1 if (stop_signal or next_tree_width <= 0) else 0
            break

        curr_tree_width = next_tree_width
        curr_verify_num = next_verify_num

    return curr_tree_width, curr_verify_num, curr_cumu_count, depth_done, stopped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cost_draft_tree_multilayer_orchestrator_case.txt"),
    )
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260226)
    parser.add_argument("--eps-abs", type=float, default=1e-5)
    parser.add_argument("--eps-rel", type=float, default=1e-5)
    args = parser.parse_args()

    if args.tree_depth <= 1:
        raise ValueError("--tree-depth must be > 1")

    script_dir = Path(__file__).resolve().parent
    search_dirs = []
    for p in DEFAULT_SEARCH_DIRS:
        search_dirs.append((p if p.is_absolute() else (script_dir.parents[4] / p)).resolve())

    fused_case_path = _find_case(search_dirs, "cost_draft_tree_fused_wiring_case.txt")
    update_case_path = _find_case(search_dirs, "cost_draft_tree_update_case.txt")
    controller_case_path = _find_case(search_dirs, "cost_draft_tree_controller_case.txt")

    fused = _parse_key_count_file(fused_case_path) if fused_case_path is not None else None

    rng = random.Random(args.seed)

    batch_size = 1
    node_top_k = 4
    hidden_size = 64
    curr_depth_start = 0
    prefix_len = 8
    max_node_count = 128
    max_verify_num = 64
    max_tree_width = 4
    init_tree_width = 4
    init_verify_num = 8
    init_cumu_count = 1
    enable_initial_loop = 1
    hot_vocab_size = 8192
    use_hot_token_id = 0
    efficient_lm_rank = 128
    efficient_lm_vocab_size = 256
    max_seq_tokens = 512

    strict_recurrent_depth = [0 for _ in range(args.tree_depth)]

    if fused is not None:
        meta = _get_ints(fused, "meta")
        if len(meta) >= 17:
            f_batch = max(1, meta[0])
            f_topk = max(1, meta[1])
            f_tree = max(1, meta[2])
            f_hidden = max(1, meta[3])
            f_cumu = max(0, meta[4])
            f_verify = max(1, meta[6])
            f_max_node = max(1, meta[9])
            f_max_verify = max(1, meta[10])
            f_max_tree = max(1, meta[11])
            f_hot_vocab = max(1, meta[15])
            f_use_hot = 1 if meta[16] != 0 else 0

            node_top_k = f_topk
            hidden_size = f_hidden
            max_node_count = f_max_node
            max_verify_num = f_max_verify
            max_tree_width = f_max_tree
            init_tree_width = _clamp(f_tree, 1, min(max_tree_width, node_top_k))
            init_verify_num = _clamp(f_verify, 1, max_verify_num)
            init_cumu_count = min(1, f_cumu)
            hot_vocab_size = f_hot_vocab
            use_hot_token_id = f_use_hot
            f_batch_size = f_batch
            f_tree_width = f_tree
        else:
            f_batch_size = 0
            f_tree_width = 0
    else:
        f_batch_size = 0
        f_tree_width = 0

    tree_n = batch_size * max_tree_width
    hidden_n = batch_size * max_tree_width * hidden_size
    per_depth_topk = batch_size * max_tree_width * node_top_k
    node_n = batch_size * max_node_count
    out_n = batch_size * node_top_k
    work_n = batch_size * (max_verify_num + node_top_k)
    sort_n = batch_size * max_verify_num

    hot_token_id = list(range(hot_vocab_size))
    if fused is not None and "hot_token_id" in fused:
        src_hot = _get_i64s(fused, "hot_token_id", required=True)
        if len(src_hot) >= hot_vocab_size:
            hot_token_id = src_hot[:hot_vocab_size]

    initial_hidden_states = [rng.uniform(-1.0, 1.0) for _ in range(batch_size * hidden_size)]
    if fused is not None and "input_hidden_states" in fused and f_batch_size > 0 and f_tree_width > 0:
        src_hidden = _get_floats(fused, "input_hidden_states", required=True)
        expected_hidden = f_batch_size * f_tree_width * hidden_size
        if len(src_hidden) >= expected_hidden:
            initial_hidden_states = src_hidden[:hidden_size]

    initial_topk_probas = [0.0 for _ in range(batch_size * node_top_k)]
    initial_topk_tokens = [0 for _ in range(batch_size * node_top_k)]
    have_fused_initial = False
    if fused is not None and "topk_probas_sampling" in fused and "topk_tokens_sampling" in fused and f_batch_size > 0:
        src_p = _get_floats(fused, "topk_probas_sampling", required=True)
        src_t = _get_i64s(fused, "topk_tokens_sampling", required=True)
        src_per_batch = f_tree_width * node_top_k
        if len(src_p) >= src_per_batch and len(src_t) >= src_per_batch and src_per_batch >= node_top_k:
            initial_topk_probas = src_p[:node_top_k]
            initial_topk_tokens = src_t[:node_top_k]
            have_fused_initial = True
    if not have_fused_initial:
        probs = [0.1 + rng.random() for _ in range(node_top_k)]
        s = sum(probs)
        initial_topk_probas = [p / s for p in probs]
        initial_topk_tokens = [rng.randrange(hot_vocab_size) for _ in range(node_top_k)]

    step_input_tokens_init = [i for i in range(tree_n)]
    step_last_layer_scores_init = [1.0 - 0.05 * i for i in range(tree_n)]
    step_topk_indexs_prev_init = [i for i in range(tree_n)]
    step_input_hidden_states_init = [0.0 for _ in range(hidden_n)]
    for t in range(max_tree_width):
        dst_base = t * hidden_size
        for h in range(hidden_size):
            step_input_hidden_states_init[dst_base + h] = initial_hidden_states[h]

    policy_next_tree_width = []
    policy_next_verify_num = []
    policy_stop_signal = []
    for d in range(args.tree_depth):
        if d == 0:
            w = init_tree_width
        elif d % 2 == 1 and init_tree_width > 1:
            w = init_tree_width - 1
        else:
            w = init_tree_width
        policy_next_tree_width.append(_clamp(w, 0, min(max_tree_width, node_top_k)))
        policy_next_verify_num.append(_clamp(init_verify_num, 1, max_verify_num))
        policy_stop_signal.append(0)

    recurrent_topk_probas = [0.0 for _ in range(args.tree_depth * per_depth_topk)]
    recurrent_topk_tokens = [0 for _ in range(args.tree_depth * per_depth_topk)]
    for d in range(args.tree_depth):
        base = d * per_depth_topk
        probs = [0.2 + 0.8 * rng.random() for _ in range(per_depth_topk)]
        for i in range(per_depth_topk):
            recurrent_topk_probas[base + i] = probs[i]
            recurrent_topk_tokens[base + i] = rng.randrange(hot_vocab_size)

    if fused is not None and "topk_probas_sampling" in fused and "topk_tokens_sampling" in fused and args.tree_depth > 1:
        src_p = _get_floats(fused, "topk_probas_sampling", required=True)
        src_t = _get_i64s(fused, "topk_tokens_sampling", required=True)
        src_per_batch = f_tree_width * node_top_k
        if len(src_p) >= src_per_batch and len(src_t) >= src_per_batch and src_per_batch >= per_depth_topk:
            d = 1
            base = d * per_depth_topk
            recurrent_topk_probas[base : base + per_depth_topk] = src_p[:per_depth_topk]
            recurrent_topk_tokens[base : base + per_depth_topk] = src_t[:per_depth_topk]
            strict_recurrent_depth[d] = 1

    init_legacy_cumu_tokens = [-777 for _ in range(node_n)]
    init_legacy_cumu_scores = [-3.0 for _ in range(node_n)]
    init_legacy_cumu_deltas = [-1 for _ in range(node_n)]
    init_legacy_prev_indexs = [-1 for _ in range(node_n)]
    init_legacy_next_indexs = [-1 for _ in range(node_n)]
    init_legacy_side_indexs = [-1 for _ in range(node_n)]
    init_legacy_output_scores = [-4.0 for _ in range(out_n)]
    init_legacy_output_tokens = [-1 for _ in range(out_n)]
    init_legacy_work_scores = [-6.0 for _ in range(work_n)]
    init_legacy_sort_scores = [-2.0 for _ in range(sort_n)]

    ws0 = min(init_verify_num, init_cumu_count)
    for i in range(ws0):
        init_legacy_sort_scores[i] = 2.0 - 0.01 * i
        init_legacy_work_scores[i] = 2.0 - 0.01 * i

    if fused is not None and f_batch_size > 0:
        legacy_src_specs = [
            ("legacy_cumu_tokens", init_legacy_cumu_tokens, max_node_count, int),
            ("legacy_cumu_scores", init_legacy_cumu_scores, max_node_count, float),
            ("legacy_cumu_deltas", init_legacy_cumu_deltas, max_node_count, int),
            ("legacy_prev_indexs", init_legacy_prev_indexs, max_node_count, int),
            ("legacy_next_indexs", init_legacy_next_indexs, max_node_count, int),
            ("legacy_side_indexs", init_legacy_side_indexs, max_node_count, int),
            ("legacy_output_scores", init_legacy_output_scores, node_top_k, float),
            ("legacy_output_tokens", init_legacy_output_tokens, node_top_k, int),
            ("legacy_work_scores", init_legacy_work_scores, max_verify_num + node_top_k, float),
            ("legacy_sort_scores", init_legacy_sort_scores, max_verify_num, float),
        ]
        for key, dst, per_batch, typ in legacy_src_specs:
            if key not in fused:
                continue
            src_raw = fused[key]
            src = [typ(x) for x in src_raw]
            sliced = _slice_batch(src, 0, f_batch_size, per_batch)
            if len(sliced) == len(dst):
                dst[:] = sliced

    expected_io_tree_width, expected_io_verify_num, expected_io_cumu_count, expected_executed_depths, expected_stopped_early = _simulate_expected_scalars(
        tree_depth=args.tree_depth,
        node_top_k=node_top_k,
        init_tree_width=init_tree_width,
        init_verify_num=init_verify_num,
        init_cumu_count=init_cumu_count,
        max_tree_width=max_tree_width,
        max_verify_num=max_verify_num,
        max_node_count=max_node_count,
        policy_next_tree_width=policy_next_tree_width,
        policy_next_verify_num=policy_next_verify_num,
        policy_stop_signal=policy_stop_signal,
    )

    # Field masks order:
    # 0 io_tree_width, 1 io_verify_num, 2 io_cumu_count, 3 executed_depths, 4 stopped_early,
    # 5 cumu_tokens, 6 cumu_scores, 7 cumu_deltas, 8 output_scores, 9 output_tokens.
    expected_mask_fields = [0] * 10

    strict_count = sum(1 for x in expected_mask_fields if x != 0) + sum(
        1 for x in strict_recurrent_depth if x != 0
    )
    total_count = len(expected_mask_fields) + len(strict_recurrent_depth)
    if strict_count == 0:
        gt_mode = "synthetic"
    elif strict_count == total_count:
        gt_mode = "strict"
    else:
        gt_mode = "mixed"

    expected_cumu_tokens = [-1 for _ in range(node_n)]
    expected_cumu_scores = [0.0 for _ in range(node_n)]
    expected_cumu_deltas = [-1 for _ in range(node_n)]
    expected_output_scores = [0.0 for _ in range(out_n)]
    expected_output_tokens = [-1 for _ in range(out_n)]

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# cost_draft_tree multilayer orchestrator case v1\n")
        _write_line(
            f,
            "meta",
            [
                batch_size,
                node_top_k,
                hidden_size,
                args.tree_depth,
                curr_depth_start,
                prefix_len,
                max_node_count,
                max_verify_num,
                max_tree_width,
                init_tree_width,
                init_verify_num,
                init_cumu_count,
                enable_initial_loop,
                hot_vocab_size,
                use_hot_token_id,
                efficient_lm_rank,
                efficient_lm_vocab_size,
                max_seq_tokens,
                args.seed,
            ],
        )
        _write_float_line(f, "eps_abs", [args.eps_abs])
        _write_float_line(f, "eps_rel", [args.eps_rel])
        _write_line(f, "gt_mode", [gt_mode])
        _write_line(
            f,
            "source_flags",
            [
                1 if fused_case_path is not None else 0,
                1 if update_case_path is not None else 0,
                1 if controller_case_path is not None else 0,
            ],
        )

        _write_line(f, "step_input_tokens_init", step_input_tokens_init)
        _write_float_line(f, "step_input_hidden_states_init", step_input_hidden_states_init)
        _write_float_line(f, "step_last_layer_scores_init", step_last_layer_scores_init)
        _write_line(f, "step_topk_indexs_prev_init", step_topk_indexs_prev_init)

        _write_line(f, "hot_token_id", hot_token_id)

        _write_float_line(f, "initial_hidden_states", initial_hidden_states)
        _write_float_line(f, "initial_topk_probas", initial_topk_probas)
        _write_line(f, "initial_topk_tokens", initial_topk_tokens)

        _write_line(f, "policy_next_tree_width", policy_next_tree_width)
        _write_line(f, "policy_next_verify_num", policy_next_verify_num)
        _write_line(f, "policy_stop_signal", policy_stop_signal)

        _write_float_line(f, "recurrent_topk_probas", recurrent_topk_probas)
        _write_line(f, "recurrent_topk_tokens", recurrent_topk_tokens)
        _write_line(f, "expected_mask_recurrent_depth", strict_recurrent_depth)

        _write_line(f, "init_legacy_cumu_tokens", init_legacy_cumu_tokens)
        _write_float_line(f, "init_legacy_cumu_scores", init_legacy_cumu_scores)
        _write_line(f, "init_legacy_cumu_deltas", init_legacy_cumu_deltas)
        _write_line(f, "init_legacy_prev_indexs", init_legacy_prev_indexs)
        _write_line(f, "init_legacy_next_indexs", init_legacy_next_indexs)
        _write_line(f, "init_legacy_side_indexs", init_legacy_side_indexs)
        _write_float_line(f, "init_legacy_output_scores", init_legacy_output_scores)
        _write_line(f, "init_legacy_output_tokens", init_legacy_output_tokens)
        _write_float_line(f, "init_legacy_work_scores", init_legacy_work_scores)
        _write_float_line(f, "init_legacy_sort_scores", init_legacy_sort_scores)

        _write_line(f, "expected_io_tree_width", [expected_io_tree_width])
        _write_line(f, "expected_io_verify_num", [expected_io_verify_num])
        _write_line(f, "expected_io_cumu_count", [expected_io_cumu_count])
        _write_line(f, "expected_executed_depths", [expected_executed_depths])
        _write_line(f, "expected_stopped_early", [expected_stopped_early])

        _write_line(f, "expected_cumu_tokens", expected_cumu_tokens)
        _write_float_line(f, "expected_cumu_scores", expected_cumu_scores)
        _write_line(f, "expected_cumu_deltas", expected_cumu_deltas)
        _write_float_line(f, "expected_output_scores", expected_output_scores)
        _write_line(f, "expected_output_tokens", expected_output_tokens)
        _write_line(f, "expected_mask_fields", expected_mask_fields)

    print(f"Wrote orchestrator case: {out_path}")
    print(
        "Source coverage: "
        f"fused={'yes' if fused_case_path else 'no'}, "
        f"update={'yes' if update_case_path else 'no'}, "
        f"controller={'yes' if controller_case_path else 'no'}, "
        f"gt_mode={gt_mode}"
    )


if __name__ == "__main__":
    main()
