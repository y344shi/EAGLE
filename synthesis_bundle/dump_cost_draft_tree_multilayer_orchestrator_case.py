#!/usr/bin/env python3
"""Compose a multilayer orchestrator TB case from existing captures with synthetic fallback.

This script is SM75-safe: it never compiles or runs CUDA kernels.
It reuses already captured CostDraftTree case files when present, and fills
missing tensors deterministically.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

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


def _get_strings(
    kv: Dict[str, List[str]], key: str, required: bool = True
) -> List[str]:
    if key not in kv:
        if required:
            raise KeyError(f"missing key: {key}")
        return []
    return list(kv[key])


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


def _load_fp16_bin(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.fromfile(path, dtype=np.float16).astype(np.float32)


def _load_i32_bin(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.fromfile(path, dtype=np.int32)


def _resolve_artifact_path(base_dir: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _find_hls_hw_roots(anchor: Path) -> Optional[Tuple[Path, Path]]:
    """Find canonical runtime artifact roots under hardware/EAGLE/eagle/hls_hw."""
    for p in [anchor, *anchor.parents]:
        golden = (p / "hardware/EAGLE/eagle/hls_hw/eagle_verified_pipeline_4bit").resolve()
        packed = (p / "hardware/EAGLE/eagle/hls_hw/packed_all").resolve()
        if golden.exists() and packed.exists():
            return golden, packed
    return None


def _softmax_topk_from_stage(
    gathered_logits: np.ndarray,
    candidate_indices: np.ndarray,
    batch_size: int,
    node_top_k: int,
) -> tuple[List[float], List[int]]:
    if gathered_logits.size % batch_size != 0 or candidate_indices.size % batch_size != 0:
        raise ValueError("candidate/gathered logits size mismatch for batch reshape")
    width = gathered_logits.size // batch_size
    gathered = gathered_logits.reshape(batch_size, width)
    candidates = candidate_indices.reshape(batch_size, width)
    gathered = gathered - gathered.max(axis=1, keepdims=True)
    probs = np.exp(gathered)
    probs /= probs.sum(axis=1, keepdims=True)

    topk_idx = np.argsort(-probs, axis=1)[:, :node_top_k]
    topk_probs = np.take_along_axis(probs, topk_idx, axis=1)
    topk_tokens = np.take_along_axis(candidates, topk_idx, axis=1)
    return topk_probs.reshape(-1).astype(np.float32).tolist(), topk_tokens.reshape(-1).astype(np.int64).tolist()


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


def _build_full_path_fixtures(
    batch_size: int,
    hidden_size: int,
    prefix_len: int,
    max_node_count: int,
    max_verify_num: int,
    initial_hidden_states: List[float],
) -> Dict[str, List]:
    if batch_size != 1:
        raise ValueError("full-path fixture builder currently requires batch_size=1")
    if hidden_size <= 0 or max_node_count <= 0 or max_verify_num <= 0:
        raise ValueError("invalid dimensions for full-path fixture builder")
    if len(initial_hidden_states) < batch_size * hidden_size:
        raise ValueError("initial_hidden_states too small for full-path fixture builder")

    base_hidden = initial_hidden_states[:hidden_size]
    prefill_input_hidden_states_3h: List[float] = []
    prefill_input_hidden_states_3h.extend(base_hidden)
    prefill_input_hidden_states_3h.extend([0.5 * v for v in base_hidden])
    prefill_input_hidden_states_3h.extend([-0.25 * v for v in base_hidden])
    prefill_input_embed_states = [0.75 * v for v in base_hidden]

    accepted_count = max(1, min(2, max_verify_num, max_node_count, max(1, prefix_len)))
    accepted_draft_node_ids = list(range(accepted_count))
    node_to_hbm_slot_init = [-1 for _ in range(max_node_count)]
    for i in range(accepted_count):
        node_to_hbm_slot_init[i] = i

    return {
        "enable_prefill_stage": [1],
        "prefill_input_hidden_states_3h": prefill_input_hidden_states_3h,
        "prefill_input_embed_states": prefill_input_embed_states,
        "prefill_fixture_mode": ["synthetic"],
        "enable_accepted_kv_compact": [1],
        "accepted_draft_node_ids": accepted_draft_node_ids,
        "node_to_hbm_slot_init": node_to_hbm_slot_init,
        "compact_fixture_mode": ["synthetic"],
    }


def _merge_full_path_fixtures_from_e2e(
    e2e: Dict[str, List[str]],
    full_path: Dict[str, List],
    batch_size: int,
    hidden_size: int,
    max_node_count: int,
) -> Dict[str, List]:
    out = dict(full_path)

    enable_prefill_vals = _get_ints(e2e, "enable_prefill_stage", required=False)
    if enable_prefill_vals:
        enable_prefill_stage = 1 if enable_prefill_vals[0] != 0 else 0
    else:
        enable_prefill_stage = (
            1
            if (
                "enable_prefill_stage" in out
                and out["enable_prefill_stage"]
                and out["enable_prefill_stage"][0] != 0
            )
            else 0
        )
    if enable_prefill_stage:
        expected_hidden_3h = batch_size * 3 * hidden_size
        expected_embed = batch_size * hidden_size
        prefill_hidden = _get_floats(
            e2e, "prefill_input_hidden_states_3h", required=False
        )
        prefill_embed = _get_floats(e2e, "prefill_input_embed_states", required=False)
        if len(prefill_hidden) != expected_hidden_3h or len(prefill_embed) != expected_embed:
            print(
                "[warn] e2e requested prefill_stage=1 but prefill tensors are invalid; "
                "falling back to synthetic prefill fixtures."
            )
            out["enable_prefill_stage"] = [1]
            out["prefill_fixture_mode"] = ["synthetic"]
        else:
            out["enable_prefill_stage"] = [1]
            out["prefill_input_hidden_states_3h"] = prefill_hidden
            out["prefill_input_embed_states"] = prefill_embed
            out["prefill_fixture_mode"] = ["e2e"]
    else:
        out["enable_prefill_stage"] = [0]
        out["prefill_input_hidden_states_3h"] = []
        out["prefill_input_embed_states"] = []
        out["prefill_fixture_mode"] = ["disabled"]

    enable_compact_vals = _get_ints(e2e, "enable_accepted_kv_compact", required=False)
    if enable_compact_vals:
        enable_compact = 1 if enable_compact_vals[0] != 0 else 0
    else:
        enable_compact = (
            1
            if (
                "enable_accepted_kv_compact" in out
                and out["enable_accepted_kv_compact"]
                and out["enable_accepted_kv_compact"][0] != 0
            )
            else 0
        )
    node_to_hbm_slot_init = _get_i64s(e2e, "node_to_hbm_slot_init", required=False)
    if len(node_to_hbm_slot_init) != max_node_count:
        if enable_compact:
            raise ValueError(
                "enable_accepted_kv_compact=1 requires node_to_hbm_slot_init "
                f"size={max_node_count}, got={len(node_to_hbm_slot_init)}"
            )
        node_to_hbm_slot_init = [-1 for _ in range(max_node_count)]

    accepted_draft_node_ids = _get_i64s(e2e, "accepted_draft_node_ids", required=False)
    if enable_compact:
        if not accepted_draft_node_ids:
            raise ValueError(
                "enable_accepted_kv_compact=1 requires non-empty accepted_draft_node_ids"
            )
        for node_id in accepted_draft_node_ids:
            if node_id < 0 or node_id >= max_node_count:
                raise ValueError(
                    f"accepted_draft_node_ids contains out-of-range id {node_id} "
                    f"(max_node_count={max_node_count})"
                )
        out["compact_fixture_mode"] = ["e2e"]
    else:
        accepted_draft_node_ids = []
        out["compact_fixture_mode"] = ["disabled"]

    out["enable_accepted_kv_compact"] = [enable_compact]
    out["accepted_draft_node_ids"] = accepted_draft_node_ids
    out["node_to_hbm_slot_init"] = node_to_hbm_slot_init
    return out


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
    parser.add_argument("--e2e-case", type=Path, default=None)
    parser.add_argument(
        "--allow-e2e-fallback",
        action="store_true",
        help="Allow fallback to synthetic/mixed generation if e2e case is invalid.",
    )
    parser.add_argument(
        "--strict-classic",
        action="store_true",
        help="For eagle4_classic e2e input, keep strict masks/expected vectors from capture.",
    )
    args = parser.parse_args()

    if args.tree_depth <= 1 and args.e2e_case is None:
        raise ValueError("--tree-depth must be > 1")

    script_dir = Path(__file__).resolve().parent
    search_dirs = []
    for p in DEFAULT_SEARCH_DIRS:
        search_dirs.append((p if p.is_absolute() else (script_dir.parents[4] / p)).resolve())

    e2e_case_path: Optional[Path]
    if args.e2e_case is not None:
        e2e_case_path = args.e2e_case.resolve()
    else:
        e2e_case_path = _find_case(search_dirs, "cost_draft_tree_draft_e2e_case.txt")

    # Fallback path may still emit optional diagnostic fields; keep a safe default.
    e2e: Dict[str, List[str]] = {}

    if e2e_case_path is not None and e2e_case_path.exists():
        try:
            e2e = _parse_key_count_file(e2e_case_path)
            capture_backend = _get_strings(e2e, "capture_backend", required=False)
            if args.strict_classic and (
                not capture_backend
                or capture_backend[0] not in ("eagle4_classic", "classic_eagle")
            ):
                raise ValueError(
                    "strict-classic requires capture_backend=eagle4_classic|classic_eagle "
                    "in the input e2e case"
                )
            if capture_backend and capture_backend[0] in ("eagle4_classic", "classic_eagle"):
                meta = _get_ints(e2e, "meta", required=True)
                if len(meta) < 19:
                    raise ValueError("classic-eagle e2e meta must contain at least 19 ints")

                batch_size = meta[0]
                node_top_k = meta[1]
                hidden_size = meta[2]
                tree_depth = meta[3]
                curr_depth_start = meta[4]
                prefix_len = meta[5]
                max_node_count = meta[6]
                max_verify_num = meta[7]
                max_tree_width = meta[8]
                init_tree_width = meta[9]
                init_verify_num = meta[10]
                init_cumu_count = meta[11]
                enable_initial_loop = meta[12]
                hot_vocab_size = meta[13]
                use_hot_token_id = meta[14]
                efficient_lm_rank = meta[15]
                efficient_lm_vocab_size = meta[16]
                max_seq_tokens = meta[17]
                seed = meta[18]

                if batch_size != 1:
                    raise ValueError("classic-eagle case precondition failed: batch_size must be 1")
                if tree_depth <= 1:
                    raise ValueError("classic-eagle case precondition failed: tree_depth must be > 1")
                if enable_initial_loop == 0:
                    raise ValueError(
                        "classic-eagle case precondition failed: enable_initial_loop must be 1"
                    )

                eps_abs_vals = _get_floats(e2e, "eps_abs", required=False)
                eps_rel_vals = _get_floats(e2e, "eps_rel", required=False)
                eps_abs = eps_abs_vals[0] if eps_abs_vals else args.eps_abs
                eps_rel = eps_rel_vals[0] if eps_rel_vals else args.eps_rel

                golden_tensor_root_vals = _get_strings(
                    e2e, "golden_tensor_root", required=True
                )
                if len(golden_tensor_root_vals) != 1 or not golden_tensor_root_vals[0]:
                    raise ValueError("classic-eagle e2e is missing golden_tensor_root")
                golden_tensor_root = _resolve_artifact_path(
                    e2e_case_path.parent, golden_tensor_root_vals[0]
                )
                packed_dir = (golden_tensor_root.parent / "packed_all").resolve()
                # E2E metadata may point to an older capture tree. Prefer the local
                # hls_hw export roots (step-1 outputs) when available.
                hw_roots = _find_hls_hw_roots(script_dir) or _find_hls_hw_roots(
                    e2e_case_path.parent
                )
                if hw_roots is not None:
                    hw_golden_root, hw_packed_dir = hw_roots
                    if hw_golden_root != golden_tensor_root or hw_packed_dir != packed_dir:
                        print(
                            "[info] overriding classic-eagle artifact roots with local hls_hw exports: "
                            f"golden={hw_golden_root} packed={hw_packed_dir}"
                        )
                    golden_tensor_root = hw_golden_root
                    packed_dir = hw_packed_dir
                tensor_dir = golden_tensor_root / "cpmcu_tensors"

                prefix_hbm_dtype = _get_strings(e2e, "prefix_hbm_dtype", required=False)
                prefix_hbm_k_file = _get_strings(e2e, "prefix_hbm_k_file", required=False)
                prefix_hbm_v_file = _get_strings(e2e, "prefix_hbm_v_file", required=False)
                prefix_hbm_token_count = _get_ints(
                    e2e, "prefix_hbm_token_count", required=False
                )
                prefix_hbm_elems_per_token = _get_ints(
                    e2e, "prefix_hbm_elems_per_token", required=False
                )

                prefix_k_src: Optional[Path] = None
                prefix_v_src: Optional[Path] = None
                has_prefix_meta = (
                    prefix_hbm_dtype == ["fp16"]
                    and len(prefix_hbm_k_file) == 1
                    and len(prefix_hbm_v_file) == 1
                    and len(prefix_hbm_token_count) == 1
                    and len(prefix_hbm_elems_per_token) == 1
                )
                if has_prefix_meta:
                    if prefix_hbm_token_count[0] != prefix_len:
                        raise ValueError(
                            f"classic-eagle prefix token count mismatch: got={prefix_hbm_token_count[0]} expected={prefix_len}"
                        )
                    prefix_k_src = _resolve_artifact_path(
                        e2e_case_path.parent, prefix_hbm_k_file[0]
                    )
                    prefix_v_src = _resolve_artifact_path(
                        e2e_case_path.parent, prefix_hbm_v_file[0]
                    )
                    if not prefix_k_src.exists() or not prefix_v_src.exists():
                        print(
                            "[warn] classic-eagle prefix_hbm metadata exists but sidecars are "
                            f"missing: {prefix_k_src} / {prefix_v_src}; trying fallback search."
                        )
                        prefix_k_src = None
                        prefix_v_src = None

                if prefix_k_src is None or prefix_v_src is None:
                    fallback_pairs = [
                        (
                            e2e_case_path.parent
                            / "cost_draft_tree_draft_e2e_prefix_k_layer0.fp16.bin",
                            e2e_case_path.parent
                            / "cost_draft_tree_draft_e2e_prefix_v_layer0.fp16.bin",
                        ),
                        (
                            script_dir.parents[4]
                            / "capture/e2e_draft/cost_draft_tree_draft_e2e_prefix_k_layer0.fp16.bin",
                            script_dir.parents[4]
                            / "capture/e2e_draft/cost_draft_tree_draft_e2e_prefix_v_layer0.fp16.bin",
                        ),
                        (
                            script_dir.parents[4]
                            / "capture/cases/cost_draft_tree_multilayer_orchestrator_prefix_k_layer0.fp16.bin",
                            script_dir.parents[4]
                            / "capture/cases/cost_draft_tree_multilayer_orchestrator_prefix_v_layer0.fp16.bin",
                        ),
                        (
                            script_dir / "cost_draft_tree_multilayer_orchestrator_prefix_k_layer0.fp16.bin",
                            script_dir / "cost_draft_tree_multilayer_orchestrator_prefix_v_layer0.fp16.bin",
                        ),
                    ]
                    for cand_k, cand_v in fallback_pairs:
                        if cand_k.exists() and cand_v.exists():
                            prefix_k_src = cand_k.resolve()
                            prefix_v_src = cand_v.resolve()
                            break
                    if prefix_k_src is None or prefix_v_src is None:
                        raise FileNotFoundError(
                            "classic-eagle prefix sidecar missing in e2e metadata and no fallback "
                            "prefix_k/prefix_v pair was found."
                        )
                    print(
                        "[warn] classic-eagle e2e missing prefix_hbm metadata; using fallback "
                        f"prefix sidecars: {prefix_k_src} / {prefix_v_src}"
                    )
                    prefix_hbm_dtype = ["fp16"]
                    prefix_hbm_token_count = [prefix_len]
                    prefix_hbm_elems_per_token = [hidden_size]

                # Prefer SLM inputs captured during E2E runtime over golden tensor files.
                e2e_step_hidden = _get_floats(e2e, "step_input_hidden_states_init", required=False)
                e2e_step_embed = _get_floats(e2e, "step_input_prev_embed_init", required=False)
                e2e_initial_hidden = _get_floats(e2e, "initial_hidden_states", required=False)

                if e2e_initial_hidden and len(e2e_initial_hidden) >= batch_size * hidden_size:
                    initial_hidden_states = e2e_initial_hidden[: batch_size * hidden_size]
                    print("[info] using initial_hidden_states from E2E capture")
                else:
                    tensor_007 = tensor_dir / "tensor_007_EAGLE_INPUT_prev_hidden_ALL.bin"
                    initial_hidden_raw = _load_fp16_bin(tensor_007)
                    if initial_hidden_raw.size < batch_size * hidden_size:
                        raise ValueError(
                            f"tensor_007 too small: got={initial_hidden_raw.size} "
                            f"need>={batch_size * hidden_size}"
                        )
                    initial_hidden_states = (
                        initial_hidden_raw[: batch_size * hidden_size]
                        .astype(np.float32)
                        .tolist()
                    )
                    print("[info] using initial_hidden_states from golden tensor_007")

                # Load prev_embed (token embeddings) — separate from prev_hidden.
                initial_embed_states: Optional[List[float]] = None
                if e2e_step_embed:
                    # Embed was captured during E2E runtime — extract first token.
                    initial_embed_states = e2e_step_embed[:hidden_size]
                    print("[info] using initial_embed_states from E2E capture")
                else:
                    tensor_006 = tensor_dir / "tensor_006_EAGLE_INPUT_prev_embed_ALL.bin"
                    if tensor_006.exists():
                        initial_embed_raw = _load_fp16_bin(tensor_006)
                        if initial_embed_raw.size >= batch_size * hidden_size:
                            initial_embed_states = (
                                initial_embed_raw[: batch_size * hidden_size]
                                .astype(np.float32)
                                .tolist()
                            )
                            print("[info] using initial_embed_states from golden tensor_006")
                        else:
                            print(
                                f"[warn] tensor_006 too small ({initial_embed_raw.size}), "
                                f"falling back to hidden-as-embed"
                            )

                initial_topk_probas = _get_floats(
                    e2e, "initial_topk_probas", required=False
                )
                initial_topk_tokens = _get_i64s(
                    e2e, "initial_topk_tokens", required=False
                )
                out_n = batch_size * node_top_k
                if len(initial_topk_probas) != out_n or len(initial_topk_tokens) != out_n:
                    tensor_133 = tensor_dir / "tensor_133_EAGLE_LM_candidate_indices.bin"
                    tensor_134 = tensor_dir / "tensor_134_EAGLE_LM_gathered_logits.bin"
                    initial_topk_probas, initial_topk_tokens = _softmax_topk_from_stage(
                        _load_fp16_bin(tensor_134), _load_i32_bin(tensor_133), batch_size, node_top_k
                    )
                    if len(initial_topk_probas) != out_n or len(initial_topk_tokens) != out_n:
                        raise ValueError("failed to reconstruct initial_topk_* from stage dumps")

                hot_token_id = _get_i64s(e2e, "hot_token_id", required=False)
                if not hot_token_id:
                    hot_token_id = list(range(max(1, hot_vocab_size)))
                if len(hot_token_id) != hot_vocab_size:
                    raise ValueError("classic-eagle hot_token_id size mismatch")

                policy_next_tree_width = _get_ints(
                    e2e, "policy_next_tree_width", required=True
                )
                policy_next_verify_num = _get_ints(
                    e2e, "policy_next_verify_num", required=True
                )
                policy_stop_signal = _get_ints(
                    e2e, "policy_stop_signal", required=True
                )
                if (
                    len(policy_next_tree_width) != tree_depth
                    or len(policy_next_verify_num) != tree_depth
                    or len(policy_stop_signal) != tree_depth
                ):
                    raise ValueError("classic-eagle policy_* size mismatch")

                tree_n = batch_size * max_tree_width
                hidden_n = tree_n * hidden_size
                per_depth_topk = batch_size * max_tree_width * node_top_k
                recurrent_n = tree_depth * per_depth_topk
                node_n = batch_size * max_node_count
                work_n = batch_size * (max_verify_num + node_top_k)
                sort_n = batch_size * max_verify_num

                recurrent_topk_probas = _get_floats(
                    e2e, "recurrent_topk_probas", required=True
                )
                recurrent_topk_tokens = _get_i64s(
                    e2e, "recurrent_topk_tokens", required=True
                )
                if (
                    len(recurrent_topk_probas) != recurrent_n
                    or len(recurrent_topk_tokens) != recurrent_n
                ):
                    raise ValueError("classic-eagle recurrent_topk_* size mismatch")

                strict_recurrent_depth = _get_ints(
                    e2e, "expected_mask_recurrent_depth", required=False
                )
                if not strict_recurrent_depth:
                    strict_recurrent_depth = [0] * tree_depth
                if len(strict_recurrent_depth) != tree_depth:
                    raise ValueError(
                        "classic-eagle expected_mask_recurrent_depth size mismatch"
                    )
                if (not args.strict_classic) and any(v != 0 for v in strict_recurrent_depth):
                    print(
                        "[info] normalizing expected_mask_recurrent_depth for classic-eagle "
                        "to non-strict (all zeros)"
                    )
                    strict_recurrent_depth = [0] * tree_depth

                expected_io_tree_width = _get_ints(
                    e2e, "expected_io_tree_width", required=True
                )
                expected_io_verify_num = _get_ints(
                    e2e, "expected_io_verify_num", required=True
                )
                expected_io_cumu_count = _get_ints(
                    e2e, "expected_io_cumu_count", required=True
                )
                expected_executed_depths = _get_ints(
                    e2e, "expected_executed_depths", required=True
                )
                expected_stopped_early = _get_ints(
                    e2e, "expected_stopped_early", required=True
                )
                if (
                    len(expected_io_tree_width) != 1
                    or len(expected_io_verify_num) != 1
                    or len(expected_io_cumu_count) != 1
                    or len(expected_executed_depths) != 1
                    or len(expected_stopped_early) != 1
                ):
                    raise ValueError("classic-eagle expected scalar fields size mismatch")

                expected_mask_fields = _get_ints(
                    e2e, "expected_mask_fields", required=False
                )
                if not expected_mask_fields:
                    expected_mask_fields = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
                if len(expected_mask_fields) != 10:
                    raise ValueError("classic-eagle expected_mask_fields size mismatch")
                if not args.strict_classic:
                    # Legacy classic mode keeps strictness only on top-level IO/control fields.
                    canonical_mask = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
                    if expected_mask_fields != canonical_mask:
                        print(
                            "[info] normalizing expected_mask_fields for classic-eagle "
                            f"from {expected_mask_fields} to {canonical_mask}"
                        )
                        expected_mask_fields = canonical_mask
                else:
                    if any(v == 0 for v in expected_mask_fields):
                        raise ValueError(
                            "strict-classic requires expected_mask_fields to be all-ones; "
                            f"got={expected_mask_fields}"
                        )
                    if tree_depth > 1:
                        tail = strict_recurrent_depth[1:]
                        if any(v == 0 for v in tail):
                            raise ValueError(
                                "strict-classic requires expected_mask_recurrent_depth[1:] "
                                f"to be all-ones; got={strict_recurrent_depth}"
                            )

                step_input_tokens_init = [0] * tree_n
                step_last_layer_scores_init = [0.0] * tree_n
                step_topk_indexs_prev_init = [0] * tree_n
                step_input_hidden_states_init = [0.0] * hidden_n
                step_input_prev_embed_init: Optional[List[float]] = None

                active_width = min(init_tree_width, max_tree_width, node_top_k)
                for t in range(active_width):
                    step_input_tokens_init[t] = initial_topk_tokens[t]
                    step_last_layer_scores_init[t] = initial_topk_probas[t]
                    step_topk_indexs_prev_init[t] = t

                # When full per-token E2E step data is available, use it directly
                # instead of replicating single-token initial values.
                if e2e_step_hidden and len(e2e_step_hidden) >= hidden_n:
                    step_input_hidden_states_init = e2e_step_hidden[:hidden_n]
                    print("[info] using full per-token step_input_hidden_states_init from E2E capture")
                else:
                    for t in range(active_width):
                        dst_base = t * hidden_size
                        step_input_hidden_states_init[
                            dst_base : dst_base + hidden_size
                        ] = initial_hidden_states[:hidden_size]

                if e2e_step_embed and len(e2e_step_embed) >= hidden_n:
                    step_input_prev_embed_init = e2e_step_embed[:hidden_n]
                    print("[info] using full per-token step_input_prev_embed_init from E2E capture")
                else:
                    embed_tokens_path = (
                        golden_tensor_root
                        / "hls_4bit"
                        / "weights_all_4bit"
                        / "embed_tokens.fp16.bin"
                    )
                    if embed_tokens_path.exists():
                        embed_tokens_raw = _load_fp16_bin(embed_tokens_path)
                        if embed_tokens_raw.size % hidden_size == 0:
                            vocab_rows = int(embed_tokens_raw.size // hidden_size)
                            step_input_prev_embed_init = [0.0] * hidden_n
                            copied = 0
                            for t in range(active_width):
                                token_id = int(step_input_tokens_init[t])
                                dst_base = t * hidden_size
                                if 0 <= token_id < vocab_rows:
                                    src_base = token_id * hidden_size
                                    step_input_prev_embed_init[
                                        dst_base : dst_base + hidden_size
                                    ] = embed_tokens_raw[
                                        src_base : src_base + hidden_size
                                    ].tolist()
                                    copied += 1
                                else:
                                    step_input_prev_embed_init[
                                        dst_base : dst_base + hidden_size
                                    ] = step_input_hidden_states_init[
                                        dst_base : dst_base + hidden_size
                                    ]
                            print(
                                "[info] rebuilt step_input_prev_embed_init from embed_tokens "
                                f"({copied}/{active_width} token rows copied)"
                            )
                        else:
                            print(
                                "[warn] embed_tokens size is not divisible by hidden_size; "
                                "falling back to hidden-as-embed in TB"
                            )
                    else:
                        print(
                            "[warn] embed_tokens.fp16.bin missing; "
                            "falling back to hidden-as-embed in TB"
                        )

                    if (
                        step_input_prev_embed_init is None
                        and initial_embed_states is not None
                        and len(initial_embed_states) >= hidden_size
                    ):
                        # Last-resort deterministic fallback if embed table is unavailable.
                        step_input_prev_embed_init = [0.0] * hidden_n
                        for t in range(active_width):
                            dst_base = t * hidden_size
                            step_input_prev_embed_init[
                                dst_base : dst_base + hidden_size
                            ] = initial_embed_states[:hidden_size]
                        print(
                            "[warn] using replicated initial_embed_states fallback for "
                            "step_input_prev_embed_init"
                        )

                if step_input_prev_embed_init is None:
                    print(
                        "[warn] step_input_prev_embed_init is unavailable; "
                        "TB will use step_input_hidden_states_init as embed input"
                    )

                if step_input_prev_embed_init is not None and all(
                    abs(v) == 0.0 for v in step_input_prev_embed_init[: hidden_size * min(2, active_width)]
                ):
                    print(
                        "[warn] step_input_prev_embed_init appears zero-initialized for first tokens; "
                        "check embed table capture."
                    )

                if step_input_prev_embed_init is not None:
                    for t in range(active_width):
                        dst_base = t * hidden_size
                        if step_input_prev_embed_init[dst_base : dst_base + hidden_size] == [0.0] * hidden_size:
                            step_input_prev_embed_init[
                                dst_base : dst_base + hidden_size
                            ] = step_input_hidden_states_init[dst_base : dst_base + hidden_size]

                if args.strict_classic:
                    # Strict classic replay must start from the exact runtime state.
                    init_legacy_cumu_tokens = _get_i64s(
                        e2e, "init_legacy_cumu_tokens", required=True
                    )
                    init_legacy_cumu_scores = _get_floats(
                        e2e, "init_legacy_cumu_scores", required=True
                    )
                    init_legacy_cumu_deltas = _get_i64s(
                        e2e, "init_legacy_cumu_deltas", required=True
                    )
                    init_legacy_prev_indexs = _get_i64s(
                        e2e, "init_legacy_prev_indexs", required=True
                    )
                    init_legacy_next_indexs = _get_i64s(
                        e2e, "init_legacy_next_indexs", required=True
                    )
                    init_legacy_side_indexs = _get_i64s(
                        e2e, "init_legacy_side_indexs", required=True
                    )
                    init_legacy_output_scores = _get_floats(
                        e2e, "init_legacy_output_scores", required=True
                    )
                    init_legacy_output_tokens = _get_i64s(
                        e2e, "init_legacy_output_tokens", required=True
                    )
                    init_legacy_work_scores = _get_floats(
                        e2e, "init_legacy_work_scores", required=True
                    )
                    init_legacy_sort_scores = _get_floats(
                        e2e, "init_legacy_sort_scores", required=True
                    )

                    expected_cumu_tokens = _get_i64s(
                        e2e, "expected_cumu_tokens", required=True
                    )
                    expected_cumu_scores = _get_floats(
                        e2e, "expected_cumu_scores", required=True
                    )
                    expected_cumu_deltas = _get_i64s(
                        e2e, "expected_cumu_deltas", required=True
                    )
                    expected_output_scores = _get_floats(
                        e2e, "expected_output_scores", required=True
                    )
                    expected_output_tokens = _get_i64s(
                        e2e, "expected_output_tokens", required=True
                    )
                    if (
                        len(init_legacy_cumu_tokens) != node_n
                        or len(init_legacy_cumu_scores) != node_n
                        or len(init_legacy_cumu_deltas) != node_n
                        or len(init_legacy_prev_indexs) != node_n
                        or len(init_legacy_next_indexs) != node_n
                        or len(init_legacy_side_indexs) != node_n
                        or len(init_legacy_output_scores) != out_n
                        or len(init_legacy_output_tokens) != out_n
                        or len(init_legacy_work_scores) != work_n
                        or len(init_legacy_sort_scores) != sort_n
                        or len(expected_cumu_tokens) != node_n
                        or len(expected_cumu_scores) != node_n
                        or len(expected_cumu_deltas) != node_n
                        or len(expected_output_scores) != out_n
                        or len(expected_output_tokens) != out_n
                    ):
                        raise ValueError(
                            "classic-eagle strict init/expected array size mismatch"
                        )
                else:
                    init_legacy_cumu_tokens = [-777] * node_n
                    init_legacy_cumu_scores = [-3.0] * node_n
                    init_legacy_cumu_deltas = [-1] * node_n
                    init_legacy_prev_indexs = [-1] * node_n
                    init_legacy_next_indexs = [-1] * node_n
                    init_legacy_side_indexs = [-1] * node_n
                    init_legacy_output_scores = [-4.0] * out_n
                    init_legacy_output_tokens = [-1] * out_n
                    init_legacy_work_scores = [-6.0] * work_n
                    init_legacy_sort_scores = [-2.0] * sort_n
                    expected_cumu_tokens = [-1] * node_n
                    expected_cumu_scores = [0.0] * node_n
                    expected_cumu_deltas = [-1] * node_n
                    expected_output_scores = [0.0] * out_n
                    expected_output_tokens = [-1] * out_n
                full_path = _build_full_path_fixtures(
                    batch_size=batch_size,
                    hidden_size=hidden_size,
                    prefix_len=prefix_len,
                    max_node_count=max_node_count,
                    max_verify_num=max_verify_num,
                    initial_hidden_states=initial_hidden_states,
                )
                full_path = _merge_full_path_fixtures_from_e2e(
                    e2e=e2e,
                    full_path=full_path,
                    batch_size=batch_size,
                    hidden_size=hidden_size,
                    max_node_count=max_node_count,
                )

                out_path = args.output.resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                prefix_k_dst = out_path.parent / "cost_draft_tree_multilayer_orchestrator_prefix_k_layer0.fp16.bin"
                prefix_v_dst = out_path.parent / "cost_draft_tree_multilayer_orchestrator_prefix_v_layer0.fp16.bin"
                if prefix_k_src.resolve() != prefix_k_dst.resolve():
                    shutil.copyfile(prefix_k_src, prefix_k_dst)
                if prefix_v_src.resolve() != prefix_v_dst.resolve():
                    shutil.copyfile(prefix_v_src, prefix_v_dst)

                with out_path.open("w", encoding="utf-8") as f:
                    f.write("# cost_draft_tree multilayer orchestrator case v2 eagle4-classic\n")
                    _write_line(
                        f,
                        "meta",
                        [
                            batch_size,
                            node_top_k,
                            hidden_size,
                            tree_depth,
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
                            seed,
                        ],
                    )
                    _write_float_line(f, "eps_abs", [eps_abs])
                    _write_float_line(f, "eps_rel", [eps_rel])
                    if args.strict_classic:
                        gt_mode_vals = _get_strings(e2e, "gt_mode", required=False)
                        gt_mode_out = gt_mode_vals[0] if gt_mode_vals else "strict"
                        _write_line(f, "gt_mode", [gt_mode_out])
                    else:
                        _write_line(f, "gt_mode", ["mixed"])
                    _write_line(f, "policy_mode", ["constant"])
                    _write_line(f, "capture_backend", ["eagle4_classic"])
                    _write_line(f, "golden_tensor_root", [str(golden_tensor_root)])
                    _write_line(f, "packed_dir", [str(packed_dir)])
                    _write_line(f, "prefix_hbm_dtype", ["fp16"])
                    _write_line(f, "prefix_hbm_k_file", [prefix_k_dst.name])
                    _write_line(f, "prefix_hbm_v_file", [prefix_v_dst.name])
                    _write_line(f, "prefix_hbm_token_count", [prefix_len])
                    _write_line(
                        f,
                        "prefix_hbm_elems_per_token",
                        [prefix_hbm_elems_per_token[0]],
                    )
                    _write_line(f, "step_input_tokens_init", step_input_tokens_init)
                    _write_float_line(
                        f, "step_input_hidden_states_init", step_input_hidden_states_init
                    )
                    if step_input_prev_embed_init is not None:
                        _write_float_line(
                            f, "step_input_prev_embed_init", step_input_prev_embed_init
                        )
                    _write_float_line(
                        f, "step_last_layer_scores_init", step_last_layer_scores_init
                    )
                    _write_line(
                        f, "step_topk_indexs_prev_init", step_topk_indexs_prev_init
                    )
                    _write_line(f, "hot_token_id", hot_token_id)
                    _write_float_line(f, "initial_hidden_states", initial_hidden_states)
                    _write_float_line(f, "initial_topk_probas", initial_topk_probas)
                    _write_line(f, "initial_topk_tokens", initial_topk_tokens)
                    _write_line(f, "enable_prefill_stage", full_path["enable_prefill_stage"])
                    _write_float_line(
                        f,
                        "prefill_input_hidden_states_3h",
                        full_path["prefill_input_hidden_states_3h"],
                    )
                    _write_float_line(
                        f,
                        "prefill_input_embed_states",
                        full_path["prefill_input_embed_states"],
                    )
                    _write_line(f, "prefill_fixture_mode", full_path["prefill_fixture_mode"])
                    _write_line(
                        f,
                        "enable_accepted_kv_compact",
                        full_path["enable_accepted_kv_compact"],
                    )
                    _write_line(
                        f,
                        "accepted_draft_node_ids",
                        full_path["accepted_draft_node_ids"],
                    )
                    _write_line(
                        f,
                        "node_to_hbm_slot_init",
                        full_path["node_to_hbm_slot_init"],
                    )
                    _write_line(f, "compact_fixture_mode", full_path["compact_fixture_mode"])
                    _write_line(f, "policy_next_tree_width", policy_next_tree_width)
                    _write_line(f, "policy_next_verify_num", policy_next_verify_num)
                    _write_line(f, "policy_stop_signal", policy_stop_signal)
                    _write_float_line(f, "recurrent_topk_probas", recurrent_topk_probas)
                    _write_line(f, "recurrent_topk_tokens", recurrent_topk_tokens)
                    _write_line(
                        f, "expected_mask_recurrent_depth", strict_recurrent_depth
                    )
                    # Pass through E2E SLM output diagnostics (tensor_110 etc.)
                    for diag_key in ("e2e_step0_logits_hidden", "e2e_step0_reasoning_hidden"):
                        diag_vals = _get_floats(e2e, diag_key, required=False)
                        if diag_vals:
                            _write_float_line(f, diag_key, diag_vals)
                            print(f"[info] wrote {diag_key} ({len(diag_vals)} floats)")
                    _write_line(f, "init_legacy_cumu_tokens", init_legacy_cumu_tokens)
                    _write_float_line(f, "init_legacy_cumu_scores", init_legacy_cumu_scores)
                    _write_line(f, "init_legacy_cumu_deltas", init_legacy_cumu_deltas)
                    _write_line(f, "init_legacy_prev_indexs", init_legacy_prev_indexs)
                    _write_line(f, "init_legacy_next_indexs", init_legacy_next_indexs)
                    _write_line(f, "init_legacy_side_indexs", init_legacy_side_indexs)
                    _write_float_line(
                        f, "init_legacy_output_scores", init_legacy_output_scores
                    )
                    _write_line(f, "init_legacy_output_tokens", init_legacy_output_tokens)
                    _write_float_line(f, "init_legacy_work_scores", init_legacy_work_scores)
                    _write_float_line(f, "init_legacy_sort_scores", init_legacy_sort_scores)
                    _write_line(f, "expected_io_tree_width", expected_io_tree_width)
                    _write_line(f, "expected_io_verify_num", expected_io_verify_num)
                    _write_line(f, "expected_io_cumu_count", expected_io_cumu_count)
                    _write_line(f, "expected_executed_depths", expected_executed_depths)
                    _write_line(f, "expected_stopped_early", expected_stopped_early)
                    _write_line(f, "expected_cumu_tokens", expected_cumu_tokens)
                    _write_float_line(f, "expected_cumu_scores", expected_cumu_scores)
                    _write_line(f, "expected_cumu_deltas", expected_cumu_deltas)
                    _write_float_line(f, "expected_output_scores", expected_output_scores)
                    _write_line(f, "expected_output_tokens", expected_output_tokens)
                    _write_line(f, "expected_mask_fields", expected_mask_fields)

                print(f"Wrote orchestrator case from EAGLE-4 classic capture: {out_path}")
                print(f"E2E source: {e2e_case_path}")
                return

            meta = _get_ints(e2e, "meta", required=True)
            if len(meta) < 19:
                raise ValueError("e2e meta must contain at least 19 ints")

            batch_size = meta[0]
            node_top_k = meta[1]
            hidden_size = meta[2]
            tree_depth = meta[3]
            curr_depth_start = meta[4]
            prefix_len = meta[5]
            max_node_count = meta[6]
            max_verify_num = meta[7]
            max_tree_width = meta[8]
            init_tree_width = meta[9]
            init_verify_num = meta[10]
            init_cumu_count = meta[11]
            enable_initial_loop = meta[12]
            hot_vocab_size = meta[13]
            use_hot_token_id = meta[14]
            efficient_lm_rank = meta[15]
            efficient_lm_vocab_size = meta[16]
            max_seq_tokens = meta[17]
            seed = meta[18]

            if batch_size != 1:
                raise ValueError("e2e case precondition failed: batch_size must be 1")
            if tree_depth <= 1:
                raise ValueError("e2e case precondition failed: tree_depth must be > 1")
            if enable_initial_loop == 0:
                raise ValueError("e2e case precondition failed: enable_initial_loop must be 1")

            tree_n = batch_size * max_tree_width
            hidden_n = tree_n * hidden_size
            per_depth_topk = batch_size * max_tree_width * node_top_k
            recurrent_n = tree_depth * per_depth_topk
            node_n = batch_size * max_node_count
            out_n = batch_size * node_top_k
            work_n = batch_size * (max_verify_num + node_top_k)
            sort_n = batch_size * max_verify_num

            eps_abs_vals = _get_floats(e2e, "eps_abs", required=False)
            eps_rel_vals = _get_floats(e2e, "eps_rel", required=False)
            eps_abs = eps_abs_vals[0] if eps_abs_vals else args.eps_abs
            eps_rel = eps_rel_vals[0] if eps_rel_vals else args.eps_rel

            gt_mode = e2e.get("gt_mode", ["strict"])
            policy_mode = e2e.get("policy_mode", ["dynamic"])

            step_input_tokens_init = _get_i64s(e2e, "step_input_tokens_init", required=True)
            if len(step_input_tokens_init) != tree_n:
                raise ValueError("step_input_tokens_init size mismatch in e2e case")
            step_input_hidden_states_init = _get_floats(
                e2e, "step_input_hidden_states_init", required=True
            )
            step_input_prev_embed_init = _get_floats(
                e2e, "step_input_prev_embed_init", required=False
            )
            if step_input_prev_embed_init and len(step_input_prev_embed_init) != hidden_n:
                print("[warn] step_input_prev_embed_init size mismatch, ignoring")
                step_input_prev_embed_init = []
            if len(step_input_hidden_states_init) != hidden_n:
                raise ValueError("step_input_hidden_states_init size mismatch in e2e case")
            step_last_layer_scores_init = _get_floats(
                e2e, "step_last_layer_scores_init", required=True
            )
            if len(step_last_layer_scores_init) != tree_n:
                raise ValueError("step_last_layer_scores_init size mismatch in e2e case")
            step_topk_indexs_prev_init = _get_i64s(e2e, "step_topk_indexs_prev_init", required=True)
            if len(step_topk_indexs_prev_init) != tree_n:
                raise ValueError("step_topk_indexs_prev_init size mismatch in e2e case")

            hot_token_id = _get_i64s(e2e, "hot_token_id", required=True)
            if len(hot_token_id) != hot_vocab_size:
                raise ValueError("hot_token_id size mismatch in e2e case")

            initial_hidden_states = _get_floats(e2e, "initial_hidden_states", required=True)
            if len(initial_hidden_states) != batch_size * hidden_size:
                raise ValueError("initial_hidden_states size mismatch in e2e case")
            initial_topk_probas = _get_floats(e2e, "initial_topk_probas", required=True)
            if len(initial_topk_probas) != out_n:
                raise ValueError("initial_topk_probas size mismatch in e2e case")
            initial_topk_tokens = _get_i64s(e2e, "initial_topk_tokens", required=True)
            if len(initial_topk_tokens) != out_n:
                raise ValueError("initial_topk_tokens size mismatch in e2e case")

            policy_next_tree_width = _get_ints(e2e, "policy_next_tree_width", required=True)
            policy_next_verify_num = _get_ints(e2e, "policy_next_verify_num", required=True)
            policy_stop_signal = _get_ints(e2e, "policy_stop_signal", required=True)
            if (
                len(policy_next_tree_width) != tree_depth
                or len(policy_next_verify_num) != tree_depth
                or len(policy_stop_signal) != tree_depth
            ):
                raise ValueError("policy_* size mismatch in e2e case")

            recurrent_topk_probas = _get_floats(e2e, "recurrent_topk_probas", required=True)
            recurrent_topk_tokens = _get_i64s(e2e, "recurrent_topk_tokens", required=True)
            if len(recurrent_topk_probas) != recurrent_n or len(recurrent_topk_tokens) != recurrent_n:
                raise ValueError("recurrent_topk_* size mismatch in e2e case")

            strict_recurrent_depth = _get_ints(
                e2e, "expected_mask_recurrent_depth", required=False
            )
            if not strict_recurrent_depth:
                strict_recurrent_depth = [0] * tree_depth
            if len(strict_recurrent_depth) != tree_depth:
                raise ValueError("expected_mask_recurrent_depth size mismatch in e2e case")

            init_legacy_cumu_tokens = _get_i64s(e2e, "init_legacy_cumu_tokens", required=True)
            init_legacy_cumu_scores = _get_floats(e2e, "init_legacy_cumu_scores", required=True)
            init_legacy_cumu_deltas = _get_i64s(e2e, "init_legacy_cumu_deltas", required=True)
            init_legacy_prev_indexs = _get_i64s(e2e, "init_legacy_prev_indexs", required=True)
            init_legacy_next_indexs = _get_i64s(e2e, "init_legacy_next_indexs", required=True)
            init_legacy_side_indexs = _get_i64s(e2e, "init_legacy_side_indexs", required=True)
            init_legacy_output_scores = _get_floats(e2e, "init_legacy_output_scores", required=True)
            init_legacy_output_tokens = _get_i64s(e2e, "init_legacy_output_tokens", required=True)
            init_legacy_work_scores = _get_floats(e2e, "init_legacy_work_scores", required=True)
            init_legacy_sort_scores = _get_floats(e2e, "init_legacy_sort_scores", required=True)
            if (
                len(init_legacy_cumu_tokens) != node_n
                or len(init_legacy_cumu_scores) != node_n
                or len(init_legacy_cumu_deltas) != node_n
                or len(init_legacy_prev_indexs) != node_n
                or len(init_legacy_next_indexs) != node_n
                or len(init_legacy_side_indexs) != node_n
                or len(init_legacy_output_scores) != out_n
                or len(init_legacy_output_tokens) != out_n
                or len(init_legacy_work_scores) != work_n
                or len(init_legacy_sort_scores) != sort_n
            ):
                raise ValueError("init_legacy_* size mismatch in e2e case")

            expected_io_tree_width = _get_ints(e2e, "expected_io_tree_width", required=True)
            expected_io_verify_num = _get_ints(e2e, "expected_io_verify_num", required=True)
            expected_io_cumu_count = _get_ints(e2e, "expected_io_cumu_count", required=True)
            expected_executed_depths = _get_ints(e2e, "expected_executed_depths", required=True)
            expected_stopped_early = _get_ints(e2e, "expected_stopped_early", required=True)
            if (
                len(expected_io_tree_width) != 1
                or len(expected_io_verify_num) != 1
                or len(expected_io_cumu_count) != 1
                or len(expected_executed_depths) != 1
                or len(expected_stopped_early) != 1
            ):
                raise ValueError("expected scalar outputs size mismatch in e2e case")

            expected_cumu_tokens = _get_i64s(e2e, "expected_cumu_tokens", required=True)
            expected_cumu_scores = _get_floats(e2e, "expected_cumu_scores", required=True)
            expected_cumu_deltas = _get_i64s(e2e, "expected_cumu_deltas", required=True)
            expected_output_scores = _get_floats(e2e, "expected_output_scores", required=True)
            expected_output_tokens = _get_i64s(e2e, "expected_output_tokens", required=True)
            if (
                len(expected_cumu_tokens) != node_n
                or len(expected_cumu_scores) != node_n
                or len(expected_cumu_deltas) != node_n
                or len(expected_output_scores) != out_n
                or len(expected_output_tokens) != out_n
            ):
                raise ValueError("expected final array size mismatch in e2e case")

            expected_mask_fields = _get_ints(e2e, "expected_mask_fields", required=False)
            if not expected_mask_fields:
                expected_mask_fields = [1] * 10
            if len(expected_mask_fields) != 10:
                raise ValueError("expected_mask_fields size mismatch in e2e case")

            full_path = _build_full_path_fixtures(
                batch_size=batch_size,
                hidden_size=hidden_size,
                prefix_len=prefix_len,
                max_node_count=max_node_count,
                max_verify_num=max_verify_num,
                initial_hidden_states=initial_hidden_states,
            )
            full_path = _merge_full_path_fixtures_from_e2e(
                e2e=e2e,
                full_path=full_path,
                batch_size=batch_size,
                hidden_size=hidden_size,
                max_node_count=max_node_count,
            )

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
                        tree_depth,
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
                        seed,
                    ],
                )
                _write_float_line(f, "eps_abs", [eps_abs])
                _write_float_line(f, "eps_rel", [eps_rel])
                _write_line(f, "gt_mode", gt_mode)
                _write_line(f, "policy_mode", policy_mode)
                _write_line(f, "source_flags", [0, 0, 0])

                _write_line(f, "step_input_tokens_init", step_input_tokens_init)
                _write_float_line(f, "step_input_hidden_states_init", step_input_hidden_states_init)
                if step_input_prev_embed_init:
                    _write_float_line(
                        f, "step_input_prev_embed_init", step_input_prev_embed_init
                    )
                _write_float_line(f, "step_last_layer_scores_init", step_last_layer_scores_init)
                _write_line(f, "step_topk_indexs_prev_init", step_topk_indexs_prev_init)

                _write_line(f, "hot_token_id", hot_token_id)

                _write_float_line(f, "initial_hidden_states", initial_hidden_states)
                _write_float_line(f, "initial_topk_probas", initial_topk_probas)
                _write_line(f, "initial_topk_tokens", initial_topk_tokens)
                _write_line(f, "enable_prefill_stage", full_path["enable_prefill_stage"])
                _write_float_line(
                    f,
                    "prefill_input_hidden_states_3h",
                    full_path["prefill_input_hidden_states_3h"],
                )
                _write_float_line(
                    f,
                    "prefill_input_embed_states",
                    full_path["prefill_input_embed_states"],
                )
                _write_line(f, "prefill_fixture_mode", full_path["prefill_fixture_mode"])
                _write_line(
                    f,
                    "enable_accepted_kv_compact",
                    full_path["enable_accepted_kv_compact"],
                )
                _write_line(
                    f,
                    "accepted_draft_node_ids",
                    full_path["accepted_draft_node_ids"],
                )
                _write_line(
                    f,
                    "node_to_hbm_slot_init",
                    full_path["node_to_hbm_slot_init"],
                )
                _write_line(f, "compact_fixture_mode", full_path["compact_fixture_mode"])

                _write_line(f, "policy_next_tree_width", policy_next_tree_width)
                _write_line(f, "policy_next_verify_num", policy_next_verify_num)
                _write_line(f, "policy_stop_signal", policy_stop_signal)

                _write_float_line(f, "recurrent_topk_probas", recurrent_topk_probas)
                _write_line(f, "recurrent_topk_tokens", recurrent_topk_tokens)
                _write_line(f, "expected_mask_recurrent_depth", strict_recurrent_depth)

                # Pass through E2E SLM output diagnostics (tensor_110 etc.)
                for diag_key in ("e2e_step0_logits_hidden", "e2e_step0_reasoning_hidden"):
                    diag_vals = _get_floats(e2e, diag_key, required=False)
                    if diag_vals:
                        _write_float_line(f, diag_key, diag_vals)
                        print(f"[info] wrote {diag_key} ({len(diag_vals)} floats)")
                    else:
                        print(f"[info] {diag_key} not found in E2E case")

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

                _write_line(f, "expected_io_tree_width", expected_io_tree_width)
                _write_line(f, "expected_io_verify_num", expected_io_verify_num)
                _write_line(f, "expected_io_cumu_count", expected_io_cumu_count)
                _write_line(f, "expected_executed_depths", expected_executed_depths)
                _write_line(f, "expected_stopped_early", expected_stopped_early)

                _write_line(f, "expected_cumu_tokens", expected_cumu_tokens)
                _write_float_line(f, "expected_cumu_scores", expected_cumu_scores)
                _write_line(f, "expected_cumu_deltas", expected_cumu_deltas)
                _write_float_line(f, "expected_output_scores", expected_output_scores)
                _write_line(f, "expected_output_tokens", expected_output_tokens)
                _write_line(f, "expected_mask_fields", expected_mask_fields)

            print(f"Wrote orchestrator case from e2e capture: {out_path}")
            print(f"E2E source: {e2e_case_path}")
            return
        except Exception as exc:
            if not args.allow_e2e_fallback:
                raise
            print(f"[warn] e2e case parse failed ({e2e_case_path}): {exc}; fallback enabled.")

    fused_case_path = _find_case(search_dirs, "cost_draft_tree_fused_wiring_case.txt")
    update_case_path = _find_case(search_dirs, "cost_draft_tree_update_case.txt")
    controller_case_path = _find_case(search_dirs, "cost_draft_tree_controller_case.txt")

    fused = _parse_key_count_file(fused_case_path) if fused_case_path is not None else None

    rng = random.Random(args.seed)

    batch_size = 1
    node_top_k = 4
    hidden_size = 4096
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
    full_path = _build_full_path_fixtures(
        batch_size=batch_size,
        hidden_size=hidden_size,
        prefix_len=prefix_len,
        max_node_count=max_node_count,
        max_verify_num=max_verify_num,
        initial_hidden_states=initial_hidden_states,
    )

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
        _write_line(f, "policy_mode", ["dynamic"])
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
        _write_line(f, "enable_prefill_stage", full_path["enable_prefill_stage"])
        _write_float_line(
            f,
            "prefill_input_hidden_states_3h",
            full_path["prefill_input_hidden_states_3h"],
        )
        _write_float_line(
            f,
            "prefill_input_embed_states",
            full_path["prefill_input_embed_states"],
        )
        _write_line(f, "prefill_fixture_mode", full_path["prefill_fixture_mode"])
        _write_line(
            f,
            "enable_accepted_kv_compact",
            full_path["enable_accepted_kv_compact"],
        )
        _write_line(
            f,
            "accepted_draft_node_ids",
            full_path["accepted_draft_node_ids"],
        )
        _write_line(
            f,
            "node_to_hbm_slot_init",
            full_path["node_to_hbm_slot_init"],
        )
        _write_line(f, "compact_fixture_mode", full_path["compact_fixture_mode"])

        _write_line(f, "policy_next_tree_width", policy_next_tree_width)
        _write_line(f, "policy_next_verify_num", policy_next_verify_num)
        _write_line(f, "policy_stop_signal", policy_stop_signal)

        _write_float_line(f, "recurrent_topk_probas", recurrent_topk_probas)
        _write_line(f, "recurrent_topk_tokens", recurrent_topk_tokens)
        _write_line(f, "expected_mask_recurrent_depth", strict_recurrent_depth)

        # Pass through E2E SLM output diagnostics (tensor_110 etc.)
        for diag_key in ("e2e_step0_logits_hidden", "e2e_step0_reasoning_hidden"):
            diag_vals = _get_floats(e2e, diag_key, required=False)
            if diag_vals:
                _write_float_line(f, diag_key, diag_vals)

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
