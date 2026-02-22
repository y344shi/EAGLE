#!/usr/bin/env python3
"""Dump deterministic controller KV visibility case for HLS TB."""

from __future__ import annotations

import argparse
from pathlib import Path


MAX_DEPTH = 64


def write_line(path_f, key: str, values) -> None:
    vals = list(values)
    if vals:
        payload = " ".join(str(v) for v in vals)
        path_f.write(f"{key} {len(vals)} {payload}\n")
    else:
        path_f.write(f"{key} 0\n")


def build_parent_visible_kv(
    frontier_node_ids,
    prefix_kv_locs,
    prefix_lens,
    node_parent_ids,
    node_cache_locs,
    batch_size,
    width,
    max_tree_width,
    max_prefix_len,
    max_input_size,
    max_node_count,
):
    kv_indices = [-1] * (batch_size * max_tree_width * max_input_size)
    kv_mask = [0] * (batch_size * max_tree_width * max_input_size)
    kv_lens = [0] * (batch_size * max_tree_width)
    ancestor = [-1] * (batch_size * max_tree_width * MAX_DEPTH)

    for b in range(batch_size):
        base = b * max_node_count
        prefix_len = prefix_lens[b]
        if prefix_len < 0:
            prefix_len = 0
        if prefix_len > max_prefix_len:
            prefix_len = max_prefix_len

        for q in range(max_tree_width):
            out_base = (b * max_tree_width + q) * max_input_size
            anc_base = (b * max_tree_width + q) * MAX_DEPTH

            if q >= width:
                continue

            nid = frontier_node_ids[b * max_tree_width + q]
            if nid < 0 or nid >= max_node_count:
                continue

            chain = []
            for _ in range(MAX_DEPTH):
                if nid < 0 or nid >= max_node_count:
                    break
                chain.append(nid)
                nid = node_parent_ids[base + nid]

            out_len = 0
            for i in range(prefix_len):
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
                anc_pos = out_len - prefix_len
                if 0 <= anc_pos < MAX_DEPTH:
                    ancestor[anc_base + anc_pos] = node_id
                out_len += 1

            kv_lens[b * max_tree_width + q] = out_len

    return kv_indices, kv_mask, kv_lens, ancestor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("cost_draft_tree_controller_case.txt"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--max-tree-width", type=int, default=4)
    parser.add_argument("--max-prefix-len", type=int, default=8)
    parser.add_argument("--max-input-size", type=int, default=16)
    parser.add_argument("--max-node-count", type=int, default=64)
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("default deterministic topology currently supports batch_size=1")
    if args.width != 4 or args.max_tree_width != 4:
        raise ValueError("default deterministic topology expects width=max_tree_width=4")

    batch_size = args.batch_size
    width = args.width
    max_tree_width = args.max_tree_width
    max_prefix_len = args.max_prefix_len
    max_input_size = args.max_input_size
    max_node_count = args.max_node_count

    frontier_node_ids = [8, 9, 10, 11]
    prefix_kv_locs = [11, 12, 13, 0, 0, 0, 0, 0]
    prefix_lens = [3]

    node_parent_ids = [-1] * (batch_size * max_node_count)
    node_cache_locs = [-1] * (batch_size * max_node_count)

    # Layer1 seeds.
    node_cache_locs[0] = 1001
    node_cache_locs[1] = 1002
    node_cache_locs[2] = 1003
    node_cache_locs[3] = 1004

    # Layer2 nodes 4..7 with parents 0,0,2,1.
    node_parent_ids[4] = 0
    node_parent_ids[5] = 0
    node_parent_ids[6] = 2
    node_parent_ids[7] = 1
    node_cache_locs[4] = 1101
    node_cache_locs[5] = 1102
    node_cache_locs[6] = 1103
    node_cache_locs[7] = 1104

    # Layer3 nodes 8..11 with parents 5,5,7,4.
    node_parent_ids[8] = 5
    node_parent_ids[9] = 5
    node_parent_ids[10] = 7
    node_parent_ids[11] = 4
    node_cache_locs[8] = 1201
    node_cache_locs[9] = 1202
    node_cache_locs[10] = 1203
    node_cache_locs[11] = 1204

    kv_indices, kv_mask, kv_lens, ancestor_ids = build_parent_visible_kv(
        frontier_node_ids,
        prefix_kv_locs,
        prefix_lens,
        node_parent_ids,
        node_cache_locs,
        batch_size,
        width,
        max_tree_width,
        max_prefix_len,
        max_input_size,
        max_node_count,
    )

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# cost_draft_tree controller deterministic case\n")
        write_line(f, "batch_size", [batch_size])
        write_line(f, "width", [width])
        write_line(f, "max_tree_width", [max_tree_width])
        write_line(f, "max_prefix_len", [max_prefix_len])
        write_line(f, "max_input_size", [max_input_size])
        write_line(f, "max_node_count", [max_node_count])

        write_line(f, "frontier_node_ids", frontier_node_ids)
        write_line(f, "prefix_kv_locs", prefix_kv_locs)
        write_line(f, "prefix_lens", prefix_lens)
        write_line(f, "node_parent_ids", node_parent_ids)
        write_line(f, "node_cache_locs", node_cache_locs)

        write_line(f, "expected_kv_indices", kv_indices)
        write_line(f, "expected_kv_mask", kv_mask)
        write_line(f, "expected_kv_lens", kv_lens)
        write_line(f, "expected_ancestor_node_ids", ancestor_ids)

    print(f"Wrote controller case: {out_path}")


if __name__ == "__main__":
    main()
