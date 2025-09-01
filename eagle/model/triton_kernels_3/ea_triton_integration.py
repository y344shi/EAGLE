# -*- coding: utf-8 -*-
"""
ea_triton_integration.py
Enable Triton fast-paths inside EaModel.ea_generate() with one call.

What it patches
---------------
1) LlamaAttention.forward -> fused Tree/FlashAttention (tree_attention_fused)
2) utils.tree_decoding    -> Triton gather (retrieve_logits_triton)
3) utils.evaluate_posterior (greedy only) -> Triton fast path
4) utils.update_inference_inputs -> Triton KV copy (kv_scatter_copy)

Usage
-----
from eagle.model.ea_triton_integration import enable_ea_triton, disable_ea_triton
enable_ea_triton(ea_model, use_fused_attn=True, use_gather=True, use_eval_post=True, use_kv_scatter=True)
# ... run your existing generation ...
disable_ea_triton(ea_model)
"""
import math
import types
from contextlib import contextmanager

import torch

# --- import your Triton kernels ---
try:
    from eagle.model.triton_kernels_3.fused_tree_attention_triton import tree_attention_fused
    from eagle.model.triton_kernels_3.retrieve_gather_triton import retrieve_logits_triton
    from eagle.model.triton_kernels_3.evaluate_posterior_triton import evaluate_posterior_triton
    from eagle.model.triton_kernels_3.kv_scatter_triton import kv_scatter_copy
except Exception:
    from fused_tree_attention_triton import tree_attention_fused
    from retrieve_gather_triton import retrieve_logits_triton
    from evaluate_posterior_triton import evaluate_posterior_triton
    from kv_scatter_triton import kv_scatter_copy
    
# --- ea_model (ea_generate resolves these by name imported into ea_model) ---
try:
    from eagle.model import ea_model as ea_mod
except Exception:
    import ea_model as ea_mod

# --- utils (ea_generate resolves these by name imported into ea_model) ---
try:
    from eagle.model import utils as utils_mod
except Exception:
    import utils as utils_mod

# --- apply_rotary_pos_emb import / fallback ---
try:
    from eagle.model.modeling_llama_kv import apply_rotary_pos_emb, LlamaRotaryEmbedding_L31, apply_rotary_pos_emb_L31
except Exception:
    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # noqa: F401
    except Exception:
        def _rotate_half(x: torch.Tensor) -> torch.Tensor:
            d = x.shape[-1] // 2
            x1, x2 = x[..., :d], x[..., d:]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(q: torch.Tensor,
                                 k: torch.Tensor,
                                 cos: torch.Tensor,
                                 sin: torch.Tensor,
                                 position_ids: torch.Tensor | None = None):
            # Simple broadcastable RoPE
            while cos.ndim < q.ndim:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            q_ = (q * cos) + (_rotate_half(q) * sin)
            k_ = (k * cos) + (_rotate_half(k) * sin)
            return q_, k_

# ---------------------
# utils monkey patches
# ---------------------
_ORIG = {
    "utils_tree_decoding": None,
    "utils_eval_post": None,
    "utils_update_inputs": None,
    "ea_tree_decoding": None,
    "ea_eval_post": None,
    "ea_update_inputs": None,
    "attn_forwards": [],
}
# ---------------------
# KV helpers
# ---------------------
def _kv_as_tensor(x) -> torch.Tensor:
    """
    Normalize different cache wrappers to a tensor view [B, n_kv, T, Dh],
    sliced to the active time length if such metadata exists.
    """
    # 1) direct tensor
    if torch.is_tensor(x):
        return x

    # 2) common wrappers: .data / .K / .key
    t = None
    if hasattr(x, "data"):
        t = x.data
    elif hasattr(x, "K"):
        t = x.K
    elif hasattr(x, "key"):  # some HF variants
        t = x.key
    else:
        raise TypeError(f"Unsupported KV holder type: {type(x)}")

    # 3) slice to current length if available
    clen = None
    for name in ("current_length", "curr_len", "seq_len", "length"):
        if hasattr(x, name):
            clen = getattr(x, name)
            if torch.is_tensor(clen):
                clen = int(clen.item())
            else:
                clen = int(clen)
            break

    if clen is not None:
        # time is the penultimate dim
        t = t[..., :clen, :]

    return t

def _is_kvcache(x) -> bool:
    return hasattr(x, "data") and hasattr(x, "current_length")

def _kv_len(kv_obj) -> int:
    cl = getattr(kv_obj, "current_length", None)
    if torch.is_tensor(cl):
        return int(cl.item())
    return int(cl)

def _kv_slice_data(kv_obj, upto_len: int):
    # data layout [B, n_kv, T_max, Dh]
    return kv_obj.data[..., :upto_len, :]

def _kv_append_inplace(kv_obj, new_chunk: torch.Tensor, start: int):
    # new_chunk shape [B, n_kv, T_new, Dh]
    T_new = new_chunk.shape[-2]
    kv_obj.data[..., start:start+T_new, :].copy_(new_chunk)
    cl = getattr(kv_obj, "current_length", None)
    if torch.is_tensor(cl):
        cl.fill_(start + T_new)
    else:
        kv_obj.current_length = start + T_new

def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: [B, n_kv, T, Dh] -> [B, n_kv*n_rep, T, Dh]
    if n_rep == 1:
        return x
    B, nk, T, Dh = x.shape
    return x[:, :, None, :, :].expand(B, nk, n_rep, T, Dh).reshape(B, nk * n_rep, T, Dh)

def _rope_cos_sin(rope_mod, x: torch.Tensor, kv_seq_len: int, position_ids: torch.Tensor | None = None):
    """
    Call rotary embedding across HF variants:
      - forward(x, seq_len=kv_seq_len)
      - forward(x, position_ids=position_ids)
      - forward(x, kv_seq_len)
      - forward(x)
    Returns (cos, sin) sliced to kv_seq_len along time dimension.
    """
    try:
        cos, sin = rope_mod(x, seq_len=kv_seq_len)
    except TypeError:
        if position_ids is not None:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            position_ids = position_ids.long()
            try:
                cos, sin = rope_mod(x, position_ids=position_ids)
            except TypeError:
                try:
                    cos, sin = rope_mod(x, position_ids)  # positional-only
                except TypeError:
                    try:
                        cos, sin = rope_mod(x, kv_seq_len)
                    except TypeError:
                        cos, sin = rope_mod(x)
        else:
            try:
                cos, sin = rope_mod(x, kv_seq_len)
            except TypeError:
                cos, sin = rope_mod(x)

    # ensure correct time length
    if cos.shape[-2] >= kv_seq_len:
        cos = cos[..., :kv_seq_len, :]
        sin = sin[..., :kv_seq_len, :]
    return cos, sin

# --------------------------
# Triton-backed utils hooks
# --------------------------
def _tree_decoding_triton(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices):
    # Compute position ids
    position_ids = (tree_position_ids + input_ids.shape[1]).long()
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    # Forward through base model (fused attn path will be used if enabled)
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    # EAGLE3 hidden-state concat if needed
    if getattr(model, "use_eagle3", False):
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    # `tree_logits` has shape [B, T_tree, V], where B=1.
    # `retrieve_indices` has shape [L, D] and is on CPU. It contains 1-based indices.
    # This block correctly prepares the indices for the Triton kernel.
    if retrieve_indices.numel() > 0:
        indices_cuda = retrieve_indices.to(device=tree_logits.device, non_blocking=True)
        
        # The triton kernel expects 0-based indices. `retrieve_indices` is 1-based.
        # We create a new tensor for the kernel, converting valid indices to 0-based
        # and ensuring invalid indices are -1, which the kernel handles as padding.
        indices_0based = torch.full_like(indices_cuda, -1)
        valid_mask = indices_cuda > 0
        indices_0based[valid_mask] = indices_cuda[valid_mask] - 1

        # Call the triton kernel. It expects [N,V] or [B,N,V] for logits.
        logits = retrieve_logits_triton(tree_logits, indices_0based)
    else:
        logits = torch.empty(
            (0, 0, tree_logits.shape[-1]), dtype=tree_logits.dtype, device=tree_logits.device
        )

    return logits, hidden_state, outputs

    
def _evaluate_posterior_mixed(logits: torch.Tensor, candidates: torch.Tensor, logits_processor):
    if logits_processor is None:
        best_cand, accept_len, sample_p = evaluate_posterior_triton(logits, candidates)
        return best_cand.long(), accept_len.to(torch.long), sample_p
    return _ORIG["utils_eval_post"](logits, candidates, logits_processor)

def _update_inference_inputs_triton(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    logits_processor,
    new_token,
    past_key_values_data_list,
    current_length_data,
    model,
    hidden_state_new,
    sample_p,
):
    prev_input_len = input_ids.shape[1]

    # positions to take from KV (absolute timeline)
    select_indices = retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len

    # --------------- helpers ---------------
    def _as_tensor(x):
        return x.data if hasattr(x, "data") and torch.is_tensor(x.data) else x

    def _entry_device(entry):
        if hasattr(entry, "K"):                      return _as_tensor(entry.K).device
        if isinstance(entry, dict):
            if "K" in entry:                        return _as_tensor(entry["K"]).device
            if "k" in entry:                        return _as_tensor(entry["k"]).device
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            return _as_tensor(entry[0]).device
        if torch.is_tensor(entry):                  return entry.device
        if hasattr(entry, "tensor"):                return _as_tensor(entry.tensor).device
        return None

    def _kv_first_device(kv_list, fallback):
        for e in kv_list:
            d = _entry_device(e)
            if d is not None:
                return d
        return fallback

    # pick KV device and build int32 sel on that device
    kv_dev = _kv_first_device(past_key_values_data_list, input_ids.device)
    sel = select_indices.to(dtype=torch.int32, device=kv_dev).contiguous()

    # extend input with accepted tokens
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)],
        dim=-1,
    )

    # --- Batched KV copy ---
    # This is the functionally correct but potentially slow version.
    # It stacks, runs the kernel, and then copies the data back.
    k_tensors_data = []
    v_tensors_data = []
    for entry in past_key_values_data_list:
        k_tensor, v_tensor = None, None
        if hasattr(entry, "K") and hasattr(entry, "V"):
            k_tensor, v_tensor = entry.K, entry.V
        elif isinstance(entry, dict) and (("K" in entry and "V" in entry) or ("k" in entry and "v" in entry)):
            k_tensor = entry.get("K", entry.get("k"))
            v_tensor = entry.get("V", entry.get("v"))
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            k_tensor, v_tensor = entry[0], entry[1]
        
        if k_tensor is not None and v_tensor is not None:
            k_tensors_data.append(_as_tensor(k_tensor))
            v_tensors_data.append(_as_tensor(v_tensor))

    if k_tensors_data and v_tensors_data:
        # 1. Stack to create temporary 4D tensors. Squeeze out the batch dim (B=1).
        k_stacked = torch.stack(k_tensors_data, dim=0).squeeze(1)
        v_stacked = torch.stack(v_tensors_data, dim=0).squeeze(1)

        # 2. Run the kernel on the stacked tensors. This modifies them in-place.
        kv_scatter_copy(k_stacked.contiguous(), sel, prev_len=prev_input_len)
        kv_scatter_copy(v_stacked.contiguous(), sel, prev_len=prev_input_len)

        # 3. Copy the results from the stacked tensors back to the original cache tensors.
        for i, original_k_data in enumerate(k_tensors_data):
            original_k_data.copy_(k_stacked[i].unsqueeze(0)) # Add batch dim back
        for i, original_v_data in enumerate(v_tensors_data):
            original_v_data.copy_(v_stacked[i].unsqueeze(0)) # Add batch dim back

    # update length (batch=1)
    current_length_data.fill_(int(prev_input_len + sel.numel()))

    # next-tree (unchanged)
    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]

    if logits_processor is not None:
        token = torch.multinomial(sample_p, 1)
        token = token[None]
    else:
        token = torch.argmax(sample_p)
        token = token[None, None]

    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
        accept_hidden_state_new,
        input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
        head=model.base_model.lm_head,
        logits_processor=logits_processor,
    )

    new_token += accept_length + 1
    return (
        input_ids,
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        new_token,
        None,
        token,
    )


# ---------------------------------
# LlamaAttention.forward patching
# ---------------------------------
def _make_fused_attn_forward(orig_unbound):
    """
    Wrap LlamaAttention.forward to swap the weight*mask*softmax*V part
    with tree_attention_fused when attention_mask is present (tree mode).
    'orig_unbound' must be the UNBOUND function: type(attn).forward
    """
    def fused_forward(self,
                      hidden_states: torch.Tensor,
                      attention_mask: torch.Tensor = None,
                      position_ids: torch.LongTensor = None,
                      past_key_value=None,
                      output_attentions: bool = False,
                      use_cache: bool = False,
                      **kwargs):
        # Gate: only use fused kernel if enabled and we have a mask
        if (not getattr(self, "_use_triton_fused_attn", False)) or (attention_mask is None):
            # call UNBOUND original with self=...
            return orig_unbound(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        # projections
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # reshape to [B, H, L, D]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # rotary embeddings (robust to HF variants)
        if LlamaRotaryEmbedding_L31 is not None and isinstance(self.rotary_emb, LlamaRotaryEmbedding_L31):
            cos, sin = self.rotary_emb(query_states, position_ids)
            q_rot, k_rot = apply_rotary_pos_emb_L31(query_states, key_states, cos, sin)
        else:
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                pk = past_key_value[0]
                kv_seq_len += _kv_as_tensor(pk).shape[-2]
            cos, sin = _rope_cos_sin(self.rotary_emb, value_states, kv_seq_len, position_ids=position_ids)
            q_rot, k_rot = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # concat past kv (robust to KVCache / tensor)
        if past_key_value is not None:
            k_prev = _kv_as_tensor(past_key_value[0]).to(k_rot.dtype).to(k_rot.device)
            v_prev = _kv_as_tensor(past_key_value[1]).to(value_states.dtype).to(value_states.device)
            k_full_prefix = k_prev
            v_full_prefix = v_prev
            k_cat = torch.cat([k_prev, k_rot], dim=2)
            v_cat = torch.cat([v_prev, value_states], dim=2)
        else:
            k_cat = k_rot
            v_cat = value_states

        present_key_value = (k_cat, v_cat) if use_cache else None

        # repeat kv to all heads (use concatenated tensors)
        k_full = _repeat_kv(k_cat, self.num_key_value_groups)
        v_full = _repeat_kv(v_cat, self.num_key_value_groups)
        # fused attention
        scale = 1.0 / math.sqrt(self.head_dim)
        if attention_mask is not None and attention_mask.device != query_states.device:
            attention_mask = attention_mask.to(query_states.device, non_blocking=True)

        attn_out = tree_attention_fused(
            q_rot, k_full, v_full, mask=attention_mask, scale=scale,
            BLOCK_M=64, BLOCK_N=64, BLOCK_D=min(64, self.head_dim),
        )  # [B,H,Lq,D]

        # merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_out = self.o_proj(attn_out)

        attn_weights = None  # not returned in fused path
        return attn_out, attn_weights, present_key_value

    return fused_forward

@contextmanager
def _patched_attention(model, enable=True):
    if not enable:
        yield
        return

    try:
        layers = list(model.base_model.model.layers)
    except Exception:
        layers = [m for m in model.modules() if hasattr(m, "self_attn")]

    _ORIG["attn_forwards"] = []
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "q_proj"):
            continue
        if getattr(attn, "_triton_patched", False):
            continue

        # capture UNBOUND original
        orig_unbound = type(attn).forward
        fused_fn = _make_fused_attn_forward(orig_unbound)

        # bind wrapper using descriptor protocol
        attn.forward = fused_fn.__get__(attn, type(attn))

        setattr(attn, "_triton_patched", True)
        setattr(attn, "_use_triton_fused_attn", True)
        _ORIG["attn_forwards"].append((attn, orig_unbound))

    try:
        yield
    finally:
        pass

# -----------------
# Public API
# -----------------
def enable_ea_triton(model,
                     *,
                     use_fused_attn: bool = True,
                     use_gather: bool = True,
                     use_eval_post: bool = True,
                     use_kv_scatter: bool = True):
    # utils
    if use_gather:
        if _ORIG["utils_tree_decoding"] is None:
            _ORIG["utils_tree_decoding"] = utils_mod.tree_decoding
        if _ORIG["ea_tree_decoding"] is None:
            _ORIG["ea_tree_decoding"] = getattr(ea_mod, "tree_decoding", None)

        utils_mod.tree_decoding = _tree_decoding_triton
        if hasattr(ea_mod, "tree_decoding"):
            ea_mod.tree_decoding = _tree_decoding_triton

    if use_eval_post:
        if _ORIG["utils_eval_post"] is None:
            _ORIG["utils_eval_post"] = utils_mod.evaluate_posterior
        if _ORIG["ea_eval_post"] is None:
            _ORIG["ea_eval_post"] = getattr(ea_mod, "evaluate_posterior", None)

        utils_mod.evaluate_posterior = _evaluate_posterior_mixed
        if hasattr(ea_mod, "evaluate_posterior"):
            ea_mod.evaluate_posterior = _evaluate_posterior_mixed

    if use_kv_scatter:
        if _ORIG["utils_update_inputs"] is None:
            _ORIG["utils_update_inputs"] = utils_mod.update_inference_inputs
        if _ORIG["ea_update_inputs"] is None:
            _ORIG["ea_update_inputs"] = getattr(ea_mod, "update_inference_inputs", None)

        utils_mod.update_inference_inputs = _update_inference_inputs_triton
        if hasattr(ea_mod, "update_inference_inputs"):
            ea_mod.update_inference_inputs = _update_inference_inputs_triton

    # attention
    if use_fused_attn:
        setattr(model.base_model.model, "use_triton_fused_attn", True)
        with _patched_attention(model, enable=True):
            pass

def disable_ea_triton(model=None):
    if _ORIG["utils_tree_decoding"] is not None:
        utils_mod.tree_decoding = _ORIG["utils_tree_decoding"]; _ORIG["utils_tree_decoding"] = None
    if _ORIG["ea_tree_decoding"] is not None and hasattr(ea_mod, "tree_decoding"):
        ea_mod.tree_decoding = _ORIG["ea_tree_decoding"]; _ORIG["ea_tree_decoding"] = None

    if _ORIG["utils_eval_post"] is not None:
        utils_mod.evaluate_posterior = _ORIG["utils_eval_post"]; _ORIG["utils_eval_post"] = None
    if _ORIG["ea_eval_post"] is not None and hasattr(ea_mod, "evaluate_posterior"):
        ea_mod.evaluate_posterior = _ORIG["ea_eval_post"]; _ORIG["ea_eval_post"] = None

    if _ORIG["utils_update_inputs"] is not None:
        utils_mod.update_inference_inputs = _ORIG["utils_update_inputs"]; _ORIG["utils_update_inputs"] = None
    if _ORIG["ea_update_inputs"] is not None and hasattr(ea_mod, "update_inference_inputs"):
        ea_mod.update_inference_inputs = _ORIG["ea_update_inputs"]; _ORIG["ea_update_inputs"] = None

    for attn, orig in _ORIG["attn_forwards"]:
        try:
            attn.forward = types.MethodType(orig, attn)
            setattr(attn, "_triton_patched", False)
            setattr(attn, "_use_triton_fused_attn", False)
        except Exception:
            pass
    _ORIG["attn_forwards"].clear()