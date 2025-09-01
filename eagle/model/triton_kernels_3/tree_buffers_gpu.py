# tree_buffers_gpu.py
# GPU-native (PyTorch CUDA) generation of tree buffers:
#   - attention mask over the tree (0/1, later turned into -inf in your model)
#   - position ids per node
#   - retrieve indices for leaf-to-root paths
#   - tree_indices compatible with your original "TOPK * (depth + bias) + token + 1" scheme
#
# Input is a list of lists of ints (tree choices); each path is token indices per depth.
# Padding and sorting semantics match utils.generate_tree_buffers.

from typing import List, Dict, Tuple, Optional
import math
import torch


@torch.no_grad()
def generate_tree_buffers_gpu(
    tree_choices: List[List[int]],
    top_k: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not tree_choices:
        tree_attn_mask = torch.ones((1,1,1,1), device=device, dtype=torch.float32)
        return dict(
            tree_attn_mask=tree_attn_mask,
            tree_indices=torch.zeros(1, device=device, dtype=torch.long),
            tree_position_ids=torch.zeros(1, device=device, dtype=torch.long),
            retrieve_indices=torch.zeros((1,1), device=device, dtype=torch.long),
        )

    maxd = max(len(p) for p in tree_choices)
    M = len(tree_choices)
    P = torch.full((M, maxd), -1, device=device, dtype=torch.long)
    for i, path in enumerate(tree_choices):
        P[i, :len(path)] = torch.as_tensor(path, device=device, dtype=torch.long)

    lengths = (P != -1).sum(dim=1)
    base = max(top_k + 5, 32)
    P_key = torch.where(P >= 0, P, torch.full_like(P, base - 1))
    pos_weights = torch.arange(maxd - 1, -1, -1, device=device, dtype=torch.long)
    pw = (base ** pos_weights).to(torch.long)
    lex_key = (P_key.to(torch.long) * pw).sum(dim=1)

    BIG = (base ** maxd) + 123
    sort_key = lengths.to(torch.long) * BIG + lex_key
    order = torch.argsort(sort_key)

    P_sorted = P[order]
    len_sorted = lengths[order]

    def _tuple_prefix(row: torch.Tensor, d: int) -> Tuple[int, ...]:
        return tuple(int(x) for x in row[:d].tolist())

    prefix_to_id = {(): 0}
    node_parent = torch.full((M+1,), -1, device=device, dtype=torch.long)
    node_parent[0] = 0
    node_path: List[Tuple[int,...]] = [()] * (M+1)

    for idx in range(M):
        d = int(len_sorted[idx].item())
        path = _tuple_prefix(P_sorted[idx], d)
        parent = path[:-1] if d > 0 else ()
        parent_id = prefix_to_id[parent]
        node_id = idx + 1
        prefix_to_id[path] = node_id
        node_parent[node_id] = parent_id
        node_path[node_id] = path

    T = M + 1
    tree_attn_mask = torch.zeros((T, T), device=device, dtype=torch.float32)
    tree_attn_mask.fill_(0.0)
    tree_attn_mask[:, 0] = 1.0  # root column

    # ancestor bits
    for nid in range(1, T):
        cur = node_parent[nid].item()
        while cur != 0:
            tree_attn_mask[nid, cur] = 1.0
            cur = node_parent[cur].item()

    tree_position_ids = torch.zeros(T, device=device, dtype=torch.long)
    for nid in range(1, T):
        tree_position_ids[nid] = len(node_path[nid])

    tree_indices = torch.zeros(T, device=device, dtype=torch.long)
    tree_indices[0] = 0
    bias = 0
    max_depth = int(tree_position_ids.max().item())
    for depth in range(1, max_depth + 1):
        mask_d = (tree_position_ids[1:] == depth)
        idxs = torch.nonzero(mask_d, as_tuple=False).flatten()
        if idxs.numel() == 0:
            continue
        prev_parent = None
        inlayer_bias = 0
        for j, pos in enumerate(idxs.tolist()):
            node_id = pos + 1
            path = node_path[node_id]
            parent = path[:-1]
            if j == 0:
                prev_parent = parent
                inlayer_bias = 0
            else:
                if parent != prev_parent:
                    bias += 1
                    inlayer_bias += 1
                    prev_parent = parent
            last_token = path[-1] if len(path) > 0 else 0
            tree_indices[node_id] = last_token + top_k * (depth + bias) + 1

    is_leaf = torch.ones(M+1, device=device, dtype=torch.bool)
    is_leaf[0] = False
    for a in range(1, T):
        pa = node_path[a]
        for b in range(1, T):
            if a == b:
                continue
            pb = node_path[b]
            if len(pb) > len(pa) and pb[:len(pa)] == pa:
                is_leaf[a] = False
                break

    leaf_ids = torch.nonzero(is_leaf, as_tuple=False).flatten().tolist()
    if not leaf_ids:
        retrieve_indices = torch.zeros((1, 1), device=device, dtype=torch.long)
    else:
        paths = []
        max_path_len = 1
        for leaf in leaf_ids:
            nid = leaf
            ids = []
            while nid != 0:
                ids.append(nid)
                nid = node_parent[nid].item()
            ids.reverse()
            row = torch.tensor(ids, device=device, dtype=torch.long)
            paths.append(row)
            max_path_len = max(max_path_len, 1 + len(row))
        rows = []
        for ids in paths:
            row = torch.cat([torch.zeros(1, device=device, dtype=torch.long), ids], dim=0)
            if row.numel() < max_path_len:
                pad = torch.full((max_path_len - row.numel(),), -1, device=device, dtype=torch.long)
                row = torch.cat([row, pad], dim=0)
            rows.append(row)
        retrieve_indices = torch.stack(rows, dim=0)
        L, Dcol = retrieve_indices.shape
        MAXITEM = T + 5
        key_rows = torch.where(retrieve_indices >= 0,
                               retrieve_indices,
                               torch.full_like(retrieve_indices, MAXITEM))
        base_r = MAXITEM + 5
        posw_r = (base_r ** torch.arange(Dcol - 1, -1, -1, device=device)).to(key_rows.dtype)
        row_keys = (key_rows * posw_r).sum(dim=1)
        sort_rows = torch.argsort(row_keys)
        retrieve_indices = retrieve_indices[sort_rows]

    # --- force final invariants on the mask ---
    ar = torch.arange(T, device=tree_attn_mask.device)
    tree_attn_mask[ar, ar] = 1.0    # self-attend
    tree_attn_mask[:, 0] = 1.0      # root column
    tree_attn_mask = (tree_attn_mask > 0).to(torch.float32)

    mask_4d = tree_attn_mask[None, None, :, :]

    return dict(
        tree_attn_mask=mask_4d,
        tree_indices=tree_indices,
        tree_position_ids=tree_position_ids,
        retrieve_indices=retrieve_indices,
    )


# ---------------------- simple sanity harness ----------------------

def _print_shapes(d: Dict[str, torch.Tensor]):
    ti = d["tree_indices"]; tp = d["tree_position_ids"]
    print("tree_attn_mask:", tuple(d["tree_attn_mask"].shape), d["tree_attn_mask"].dtype)
    print("tree_indices:  ", tuple(ti.shape), ti.dtype, "head=", ti[: min(10, ti.numel())].tolist())
    print("tree_pos_ids:  ", tuple(tp.shape), tp.dtype, "head=", tp[: min(10, tp.numel())].tolist())
    print("retrieve_idx:  ", tuple(d["retrieve_indices"].shape), d["retrieve_indices"].dtype)


def _sanity_example():
    # a tiny tree: depth up to 3
    choices = [
        [2],
        [2, 7],
        [2, 7, 1],
        [2, 3],
        [5],
        [5, 0],
    ]
    out = generate_tree_buffers_gpu(choices, top_k=10, device=torch.device("cuda"))
    _print_shapes(out)

    # quick checks
    mask = out["tree_attn_mask"][0, 0]  # [T,T]
    T = mask.shape[0]
    diag = mask[torch.arange(mask.size(0)), torch.arange(mask.size(0))]
    print("diag unique:", torch.unique(diag))
    bad = (diag != 1)
    if bad.any():
        print("bad diag idx:", torch.nonzero(bad).flatten())
        print("row of bad idx:", mask[bad.nonzero().item()])

    # diagonals are 1
    assert torch.all(mask.diag() == 1)
    # col 0 (root) is 1
    assert torch.all(mask[:, 0] == 1)
    # retrieve indices first column is 0 (root)
    assert torch.all(out["retrieve_indices"][:, 0] == 0)
    print("[sanity] ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    _sanity_example()
