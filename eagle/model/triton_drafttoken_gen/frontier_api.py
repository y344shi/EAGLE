
"""
frontier_api.py — Single-file API for EAGLE draft token generation ("tree frontier").

Goal
----
Expose a small, dependency-light entrypoint that takes the *three feature taps* (Low/Mid/High)
or a pre-concatenated feature tensor, and returns the 4 tree buffers the rest of the
pipeline expects:

    draft_tokens:       Long[1, T]     (packed node order; 0 is root token id)
    retrieve_indices:   Long[R, C]     (leaf ancestor chains; left-aligned, -1 padded)
    tree_mask:          Float[1, 1, T, T]  (ancestor visibility mask, 0/1)
    tree_position_ids:  Long[T]        (depth per node; 0 = root)

Backends
--------
- "ea_layer": call your existing EA layer's topK_genrate(...). This requires you to pass
  an initialized EA layer instance (e.g., model.ea_layer) and any inputs it needs (e.g., input_ids).
- "stub": produces shape-correct outputs without heavy model weights, useful for end-to-end
  plumbing tests and debugging/tracing.

You can swap this file in place of the calls within utils.initialize_tree(...) to make
the generation path modular and testable.

Example
-------
from eagle.model.frontier_api import FrontierConfig, frontier_generate

cfg = FrontierConfig(total_token=60, depth=5, top_k=10, vocab_size=128256, hidden_size=12288)

# If you have the concatenated low/mid/high taps:
front = frontier_generate(cfg, features_concat=Xcat, backend="stub")

# If you want to call the existing EA layer:
front = frontier_generate(cfg, features_concat=Xcat, backend="ea_layer",
                          ea_layer=model.ea_layer, input_ids=cur_input_ids,
                          logits_processor=None)

Returns a FrontierOutput tuple.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal, NamedTuple, Dict

import torch


# ------------------------------
# Public config + output types
# ------------------------------

@dataclass
class FrontierConfig:
    total_token: int                # T_max = root + drafted nodes
    depth: int                      # max depth - 1 (per your code's definition)
    top_k: int                      # branching at each depth
    vocab_size: int                 # lm_head rows
    hidden_size: int                # EA hidden size (post-alignment)
    use_concat_taps: bool = True    # True if features come in concatenated already
    use_fc_align: bool = False      # If True, apply a Linear(3*H_tap -> hidden_size) first
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float16

class FrontierOutput(NamedTuple):
    draft_tokens: torch.Tensor       # [1, T]
    retrieve_indices: torch.Tensor   # [R, C]
    tree_mask: torch.Tensor          # [1, 1, T, T], float
    tree_position_ids: torch.Tensor  # [T]


# ------------------------------
# Utilities to build masks & retrieve_indices from parents
# ------------------------------

def _build_tree_mask_from_parents(parents: torch.Tensor, T_eff: int, dtype: torch.dtype) -> torch.Tensor:
    """
    parents: Long[T_eff], parent index for each node in packed order; parent[0] = -1 for root
    Returns: tree_mask [1,1,T_eff,T_eff] (float 0/1)
    """
    device = parents.device
    mask = torch.zeros((T_eff, T_eff), dtype=dtype, device=device)
    for i in range(T_eff):
        p = int(parents[i].item())
        q = i
        # include all ancestors up to root
        while q >= 0:
            mask[i, q] = 1.0
            q = int(parents[q].item()) if q > 0 else -1
    return mask.unsqueeze(0).unsqueeze(0)

def _build_retrieve_indices(parents: torch.Tensor, T_eff: int) -> torch.Tensor:
    """
    Build retrieve_indices by enumerating leaf nodes and their ancestor chains.
    Left-align the chain (root..leaf) and right-pad with -1 to a uniform width.
    Returns: Long[R, C]
    """
    device = parents.device
    # identify leaves
    children = torch.zeros((T_eff,), dtype=torch.int32, device=device)
    for i in range(1, T_eff):
        p = int(parents[i].item())
        if p >= 0:
            children[p] += 1
    leaves = (children == 0).nonzero(as_tuple=False).flatten().tolist()
    if 0 in leaves:  # root is not a leaf
        leaves.remove(0)

    # build paths
    paths: List[List[int]] = []
    max_depth = 1
    for leaf in leaves:
        chain = []
        q = leaf
        while q >= 0:
            chain.append(q)
            q = int(parents[q].item()) if q > 0 else -1
        chain.reverse()  # root..leaf
        paths.append(chain)
        max_depth = max(max_depth, len(chain))

    if not paths:
        R, C = 1, 1
        out = torch.full((R, C), -1, dtype=torch.long, device=device)
        out[0, 0] = 0
        return out

    R, C = len(paths), max_depth
    out = torch.full((R, C), -1, dtype=torch.long, device=device)
    for r, chain in enumerate(paths):
        out[r, :len(chain)] = torch.tensor(chain, dtype=torch.long, device=device)
    return out


# ------------------------------
# Backends
# ------------------------------

def _backend_stub(cfg: FrontierConfig, *, features_concat: Optional[torch.Tensor] = None) -> FrontierOutput:
    """
    Shape-correct stub: produces a reasonable packed tree, mask, and retrieve_indices
    without using real weights. Good for control-flow and tracing tests.
    """
    device = cfg.device or (features_concat.device if features_concat is not None else torch.device("cpu"))
    T = cfg.total_token
    draft_tokens = torch.full((1, T), -1, dtype=torch.long, device=device)
    parents = torch.full((T,), -1, dtype=torch.long, device=device)
    pos = torch.zeros((T,), dtype=torch.long, device=device)

    # root
    draft_tokens[0, 0] = 0
    parents[0] = -1
    pos[0] = 0

    # breadth-then-depth fill
    node = 1
    for d in range(1, cfg.depth + 1):
        for k in range(cfg.top_k):
            if node >= T:
                break
            # dummy token id
            draft_tokens[0, node] = (1000 + d * 37 + k) % cfg.vocab_size
            parents[node] = 0 if d == 1 else (1 + (d - 2) * cfg.top_k + (k % cfg.top_k))
            pos[node] = d
            node += 1
        if node >= T:
            break

    T_eff = node
    tree_mask = _build_tree_mask_from_parents(parents[:T_eff], T_eff, dtype=cfg.dtype)
    retrieve_indices = _build_retrieve_indices(parents[:T_eff], T_eff)
    tree_position_ids = pos[:T_eff]

    # Pad mask to full T (optional; downstream usually uses T_eff)
    if T_eff < T:
        pad = torch.zeros((1, 1, T - T_eff, T), dtype=cfg.dtype, device=device)
        tree_mask = torch.cat([tree_mask, pad], dim=2)
        pad2 = torch.zeros((1, 1, T, T - T_eff), dtype=cfg.dtype, device=device)
        tree_mask = torch.cat([tree_mask, pad2], dim=3)

    return FrontierOutput(draft_tokens, retrieve_indices, tree_mask, tree_position_ids)


def _backend_ea_layer(cfg: FrontierConfig, *, features_concat: torch.Tensor,
                      ea_layer, input_ids: torch.Tensor, logits_processor=None) -> FrontierOutput:
    """
    Thin wrapper that calls the existing EA layer's topK_genrate(...).
    Requires: ea_layer (e.g., model.ea_layer) and input_ids.
    """
    with torch.no_grad():
        # Expect EA3-style: features_concat already on ea_layer device with width == hidden_size or 3*H_tap
        # If EA layer expects to run its own fc align, set cfg.use_fc_align=True and pass the raw concat.
        out = ea_layer.topK_genrate(
            hidden_states=features_concat,
            input_ids=input_ids,
            head=None,  # usually unused in EA3; keep for API parity
            logits_processor=logits_processor,
            total_token=cfg.total_token,
            depth=cfg.depth,
            top_k=cfg.top_k,
        )
        # EA returns exactly the four buffers we want
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = out
        return FrontierOutput(draft_tokens, retrieve_indices, tree_mask, tree_position_ids)


# ------------------------------
# Public entrypoint
# ------------------------------

def frontier_generate(
    cfg: FrontierConfig,
    *,
    features_concat: Optional[torch.Tensor] = None,
    features_low: Optional[torch.Tensor] = None,
    features_mid: Optional[torch.Tensor] = None,
    features_high: Optional[torch.Tensor] = None,
    backend: Literal["stub", "ea_layer"] = "stub",
    ea_layer: Optional[object] = None,
    input_ids: Optional[torch.Tensor] = None,
    logits_processor: Optional[object] = None,
) -> FrontierOutput:
    """
    Generate the EAGLE draft frontier using either a stub backend (shape-only)
    or the real EA layer backend that calls topK_genrate(...).

    Args
    ----
    cfg : FrontierConfig
    features_concat : [1, L, hidden or 3*H_tap]  if use_concat_taps=True
    features_low/mid/high : [1, L, H_tap]        if use_concat_taps=False
    backend : "stub" | "ea_layer"
    ea_layer : required when backend="ea_layer"
    input_ids : required when backend="ea_layer"
    logits_processor : optional (sampler)

    Returns
    -------
    FrontierOutput(draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
    """
    if backend == "stub":
        return _backend_stub(cfg, features_concat=features_concat)

    if backend == "ea_layer":
        assert ea_layer is not None and input_ids is not None, "ea_layer and input_ids required for backend='ea_layer'"
        if cfg.use_concat_taps:
            assert features_concat is not None, "features_concat required when use_concat_taps=True"
            return _backend_ea_layer(cfg, features_concat=features_concat, ea_layer=ea_layer,
                                     input_ids=input_ids, logits_processor=logits_processor)
        else:
            assert all(t is not None for t in (features_low, features_mid, features_high)),                 "features_low/mid/high required when use_concat_taps=False"
            features_concat = torch.cat([features_low, features_mid, features_high], dim=-1)
            return _backend_ea_layer(cfg, features_concat=features_concat, ea_layer=ea_layer,
                                     input_ids=input_ids, logits_processor=logits_processor)

    raise ValueError(f"Unknown backend: {backend}")
