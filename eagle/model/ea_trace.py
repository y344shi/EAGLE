# """EAGLE/toy_warmup.py
# ea_trace.py — Monkey-patch tracer for EaModel.ea_generate()

# What it does
# ------------
# - Wraps the helpers that ea_generate() calls:
#   - initialize_tree
#   - tree_decoding
#   - evaluate_posterior
#   - update_inference_inputs
# - Prints key tensor shapes & a few values for the FIRST iteration only
#   (configurable).

# How to use
# ----------
# from ea_trace import enable_ea_trace, disable_ea_trace
# enable_ea_trace(only_first_loop=True)   # install wrappers
# # ... run your warmup()/ea_generate() as usual ...
# disable_ea_trace()                      # restore originals when done
# """

# import importlib
# import builtins

# # We'll patch symbols on the module where ea_generate resolves them from.
# # In your repo, ea_model.py does: `from .utils import *`
# # so calls like initialize_tree(...) resolve on the `ea_model` module namespace.
# #
# # Thus we patch on `ea_model` (NOT on `utils`), so that ea_generate() picks them up.
# #
# # If your package name is different, adjust the import below.
# try:
#     em = importlib.import_module("ea_model")        # local module name variant
# except Exception:
#     try:
#         # If package-style import (e.g., project.ea_model) is needed
#         em = importlib.import_module(".ea_model", package="eagle.model")
#     except Exception:
#         # As a last resort, try absolute import
#         em = importlib.import_module("eagle.model.ea_model")


# _ORIG = {}
# _STATE = {
#     "enabled": False,
#     "loop_count": 0,
#     "only_first_loop": True,
#     "print_every": 1,
# }

# def _print(*args, **kwargs):
#     # Safe print that won't break if tensors aren't available in this interpreter.
#     print(*args, **kwargs, flush=True)


# def _shape(x):
#     try:
#         return tuple(x.shape)
#     except Exception:
#         try:
#             return len(x)
#         except Exception:
#             return "n/a"


# def _device_dtype(x):
#     dev = getattr(x, "device", None)
#     dt  = getattr(x, "dtype", None)
#     return f"device={dev}, dtype={dt}"


# def enable_ea_trace(print_every=1, only_first_loop=True):
#     """
#     Enable monkey-patched tracing.

#     Args:
#       print_every: print on every Nth loop (default 1 == every loop)
#       only_first_loop: if True, automatically disable after first full ea_generate() loop
#     """
#     if _STATE["enabled"]:
#         return
#     _STATE["enabled"] = True
#     _STATE["loop_count"] = 0
#     _STATE["only_first_loop"] = bool(only_first_loop)
#     _STATE["print_every"] = max(1, int(print_every))

#     # keep originals
#     _ORIG["initialize_tree"] = getattr(em, "initialize_tree")
#     _ORIG["tree_decoding"] = getattr(em, "tree_decoding")
#     _ORIG["evaluate_posterior"] = getattr(em, "evaluate_posterior")
#     _ORIG["update_inference_inputs"] = getattr(em, "update_inference_inputs")

#     # wrap
#     def initialize_tree_wrapped(input_ids, model, past_key_values, logits_processor):
#         res = _ORIG["initialize_tree"](input_ids, model, past_key_values, logits_processor)
#         try:
#             draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = res
#             _print("\n[TRACE:init] ------- initialize_tree -------")
#             _print(f"input_ids:           {_shape(input_ids)}")
#             _print(f"draft_tokens:        {_shape(draft_tokens)}")
#             _print(f"retrieve_indices:    {_shape(retrieve_indices)}")
#             _print(f"tree_mask:           {_shape(tree_mask)}")
#             _print(f"tree_position_ids:   {_shape(tree_position_ids)}")
#             _print(f"orig logits:         {_shape(logits)}")
#             _print(f"hidden_state:        {_shape(hidden_state)}")
#             _print(f"sample_token:        {_shape(sample_token)}")
#         except Exception as e:
#             _print(f"[TRACE:init] (printing failed: {e})")
#         return res

#     def tree_decoding_wrapped(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices):
#         res = _ORIG["tree_decoding"](model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices)
#         try:
#             logits, hidden_state_new, outputs = res
#             _print("\n[TRACE:loop] ------- tree_decoding -------")
#             _print(f"tree_candidates:     {_shape(tree_candidates)}  ({_device_dtype(tree_candidates)})")
#             _print(f"retrieve_indices:    {_shape(retrieve_indices)}")
#             _print(f"tree_position_ids:   {_shape(tree_position_ids)}")
#             _print(f"logits (sliced):     {_shape(logits)}")
#             _print(f"hidden_state_new:    {_shape(hidden_state_new)}")
#         except Exception as e:
#             _print(f"[TRACE:loop] (printing failed: {e})")
#         return res

#     def evaluate_posterior_wrapped(logits, candidates, logits_processor):
#         res = _ORIG["evaluate_posterior"](logits, candidates, logits_processor)
#         try:
#             best_candidate, accept_length, sample_p = res
#             _print("\n[TRACE:loop] ------- evaluate_posterior -------")
#             _print(f"logits:              {_shape(logits)}")
#             _print(f"candidates:          {_shape(candidates)}")
#             # Scalars might be 0-d tensors; make them printable
#             try:
#                 bc = int(best_candidate.item()) if hasattr(best_candidate, 'item') else int(best_candidate)
#             except Exception:
#                 bc = best_candidate
#             try:
#                 al = int(accept_length.item()) if hasattr(accept_length, 'item') else int(accept_length)
#             except Exception:
#                 al = accept_length
#             _print(f"best_candidate:      {bc}")
#             _print(f"accept_length:       {al}")
#             _print(f"sample_p:            {_shape(sample_p)}")
#         except Exception as e:
#             _print(f"[TRACE:loop] (printing failed: {e})")
#         return res

#     def update_inference_inputs_wrapped(
#         input_ids, candidates, best_candidate, accept_length, retrieve_indices,
#         logits_processor, new_token, past_key_values_data_list, current_length_data,
#         model, hidden_state_new, sample_p
#     ):
#         # Pre
#         try:
#             _STATE["loop_count"] += 1
#             do_print = (_STATE["loop_count"] % _STATE["print_every"] == 0)
#             if do_print:
#                 _print("\n[TRACE:loop] ------- update_inference_inputs (pre) -------")
#                 _print(f"input_ids (pre):     {_shape(input_ids)}")
#                 _print(f"candidates:          {_shape(candidates)}")
#         except Exception as e:
#             _print(f"[TRACE:loop] (pre failed: {e})")

#         res = _ORIG["update_inference_inputs"](
#             input_ids, candidates, best_candidate, accept_length, retrieve_indices,
#             logits_processor, new_token, past_key_values_data_list, current_length_data,
#             model, hidden_state_new, sample_p
#         )

#         try:
#             input_ids_new, draft_tokens, retrieve_indices_new, tree_mask, tree_position_ids, new_token_out, hidden_state, sample_token = res
#             _print("\n[TRACE:loop] ------- update_inference_inputs (post) -------")
#             _print(f"input_ids (post):    {_shape(input_ids_new)}")
#             _print(f"draft_tokens:        {_shape(draft_tokens)}")
#             _print(f"retrieve_indices:    {_shape(retrieve_indices_new)}")
#             _print(f"tree_mask:           {_shape(tree_mask)}")
#             _print(f"tree_position_ids:   {_shape(tree_position_ids)}")
#             _print(f"hidden_state:        {_shape(hidden_state)}")
#             _print(f"sample_token:        {_shape(sample_token)}")
#             _print(f"new_token counter:   {new_token_out}")
#         except Exception as e:
#             _print(f"[TRACE:loop] (post failed: {e})")

#         # Auto-disable after first loop if requested
#         if _STATE["only_first_loop"]:
#             disable_ea_trace()

#         return res

#     # install wrappers
#     setattr(em, "initialize_tree", initialize_tree_wrapped)
#     setattr(em, "tree_decoding", tree_decoding_wrapped)
#     setattr(em, "evaluate_posterior", evaluate_posterior_wrapped)
#     setattr(em, "update_inference_inputs", update_inference_inputs_wrapped)

#     _print("[ea-trace] enabled (only_first_loop=%s, print_every=%d)" % (_STATE["only_first_loop"], _STATE["print_every"]))


# def disable_ea_trace():
#     """Restore original functions."""
#     if not _STATE["enabled"]:
#         return
#     try:
#         for k, fn in _ORIG.items():
#             setattr(em, k, fn)
#     finally:
#         _ORIG.clear()
#         _STATE["enabled"] = False
#         _print("[ea-trace] disabled")

# ea_trace.py — enhanced
import importlib, time, math

try:
    em = importlib.import_module("ea_model")
except Exception:
    try:
        em = importlib.import_module(".ea_model", package="eagle.model")
    except Exception:
        em = importlib.import_module("eagle.model.ea_model")

_ORIG = {}
_STATE = {"enabled": False, "loop_count": 0, "only_first_loop": True, "print_every": 1}

def _print(*args, **kw):
    print(*args, **kw, flush=True)

def _shape(x):
    try: return tuple(x.shape)
    except Exception: 
        try: return len(x)
        except Exception: return "n/a"

def _stats(x):
    try:
        return {
            "shape": tuple(x.shape),
            "dtype": str(getattr(x, "dtype", None)),
            "device": str(getattr(x, "device", None)),
            "contig": bool(getattr(x, "is_contiguous", lambda: False)()),
            "stride": tuple(getattr(x, "stride", lambda: ())()),
            "ptr": getattr(x, "data_ptr", lambda: 0)(),
            "nbytes": int(x.element_size() * x.numel()),
        }
    except Exception:
        return {"shape": _shape(x)}

def _summ(x, k=3):
    # summarize a 1D/last-dim tensor: first/last k values
    try:
        import torch
        if not isinstance(x, torch.Tensor) or x.numel() == 0: return "n/a"
        v = x.flatten()
        head = v[:k].tolist()
        tail = v[-k:].tolist() if v.numel() > k else []
        return f"head={head}" + (f", tail={tail}" if tail else "")
    except Exception:
        return "n/a"

def _lp_summary(lp):
    # LogitsProcessorList pretty
    try:
        if lp is None: return "None"
        items = []
        for p in lp:
            cls = p.__class__.__name__
            params = []
            for key in ("temperature","top_p","top_k","repetition_penalty"):
                if hasattr(p, key):
                    params.append(f"{key}={getattr(p,key)}")
            items.append(cls + "(" + ",".join(params) + ")")
        return "[" + ", ".join(items) + "]"
    except Exception:
        return str(lp)

def enable_ea_trace(print_every=1, only_first_loop=True):
    disable_ea_trace()
    return

    if _STATE["enabled"]:
        return
    _STATE.update({"enabled": True, "loop_count": 0,
                   "only_first_loop": bool(only_first_loop),
                   "print_every": max(1, int(print_every))})
    # keep originals
    for name in ("initialize_tree","tree_decoding","evaluate_posterior","update_inference_inputs"):
        _ORIG[name] = getattr(em, name)

    # -------- wrappers --------
    def initialize_tree_wrapped(input_ids, model, past_key_values, logits_processor):
        t0 = time.perf_counter()
        res = _ORIG["initialize_tree"](input_ids, model, past_key_values, logits_processor)
        t1 = time.perf_counter()
        try:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = res
            cfg = getattr(model.base_model, "config", None)
            n_heads = getattr(cfg, "num_attention_heads", "n/a")
            n_kv = getattr(cfg, "num_key_value_heads", "n/a")
            hdim = (getattr(cfg, "hidden_size", 0) // (n_heads if isinstance(n_heads,int) and n_heads>0 else 1)) or "n/a"
            _print("\n[TRACE:init] ------- initialize_tree -------")
            _print(f"input_ids:           {_stats(input_ids)}")
            _print(f"draft_tokens:        {_stats(draft_tokens)}")
            _print(f"retrieve_indices:    {_stats(retrieve_indices)}  (L={retrieve_indices.shape[0]}, D={retrieve_indices.shape[1]})")
            # tree stats
            tm = _stats(tree_mask); _print(f"tree_mask:           {tm}")
            try:
                import torch
                nnz = int(tree_mask.nonzero().shape[0])
                _print(f"  tree_mask nnz:     {nnz} / {tree_mask.numel()}  ({nnz/tree_mask.numel():.4%})")
            except Exception: pass
            _print(f"tree_position_ids:   {_stats(tree_position_ids)}  (max_depth={int(tree_position_ids.max().item()) if hasattr(tree_position_ids,'max') else 'n/a'})")
            _print(f"orig logits:         {_stats(logits)}   (V={logits.shape[-1]})")
            _print(f"hidden_state:        {_stats(hidden_state)}")
            _print(f"sample_token:        {_stats(sample_token)}  {_summ(sample_token)}")
            _print(f"sampler:             {_lp_summary(logits_processor)}")
            _print(f"model heads:         n_heads={n_heads}, n_kv={n_kv}, head_dim={hdim}")
            _print(f"elapsed:             {(t1-t0)*1e3:.2f} ms")
        except Exception as e:
            _print(f"[TRACE:init] (printing failed: {e})")
        return res

    def tree_decoding_wrapped(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices):
        t0 = time.perf_counter()
        res = _ORIG["tree_decoding"](model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices)
        t1 = time.perf_counter()
        try:
            logits, hidden_state_new, outputs = res
            _print("\n[TRACE:loop] ------- tree_decoding -------")
            _print(f"tree_candidates:     {_stats(tree_candidates)}")
            _print(f"retrieve_indices:    {_stats(retrieve_indices)}")
            _print(f"tree_position_ids:   {_stats(tree_position_ids)}")
            _print(f"logits (sliced):     {_stats(logits)}")
            _print(f"hidden_state_new:    {_stats(hidden_state_new)}")
            # KV overview
            try:
                p = outputs.get("past_key_values", None) if isinstance(outputs, dict) else getattr(outputs, "past_key_values", None)
                if p is not None and len(p) > 0:
                    k0, v0 = p[0]
                    _print(f"past_kv[0].K:       {_stats(k0)}")
                    _print(f"past_kv[0].V:       {_stats(v0)}")
                    _print(f"layers:             {len(p)}")
            except Exception: pass
            _print(f"elapsed:             {(t1-t0)*1e3:.2f} ms")
        except Exception as e:
            _print(f"[TRACE:loop] (printing failed: {e})")
        return res

    def evaluate_posterior_wrapped(logits, candidates, logits_processor):
        t0 = time.perf_counter()
        res = _ORIG["evaluate_posterior"](logits, candidates, logits_processor)
        t1 = time.perf_counter()
        try:
            best_candidate, accept_length, sample_p = res
            bc = int(best_candidate.item()) if hasattr(best_candidate, 'item') else int(best_candidate)
            al = int(accept_length.item()) if hasattr(accept_length, 'item') else int(accept_length)
            _print("\n[TRACE:loop] ------- evaluate_posterior -------")
            _print(f"logits:              {_stats(logits)}")
            _print(f"candidates:          {_stats(candidates)}")
            # health check for -1 sentinel
            try:
                import torch
                bad = int((candidates < -1).sum().item())
                neg1 = int((candidates == -1).sum().item())
                _print(f"  sentinel:          count(-1)={neg1}, count(<-1)={bad}")
            except Exception: pass
            _print(f"best_candidate:      {bc}")
            _print(f"accept_length:       {al}")
            _print(f"sample_p:            {_stats(sample_p)}  {_summ(sample_p, k=5)}")
            _print(f"sampler:             {_lp_summary(logits_processor)}")
            _print(f"elapsed:             {(t1-t0)*1e3:.2f} ms")
        except Exception as e:
            _print(f"[TRACE:loop] (printing failed: {e})")
        return res

    def update_inference_inputs_wrapped(
        input_ids, candidates, best_candidate, accept_length, retrieve_indices,
        logits_processor, new_token, past_key_values_data_list, current_length_data,
        model, hidden_state_new, sample_p
    ):
        _STATE["loop_count"] += 1
        do_print = (_STATE["loop_count"] % _STATE["print_every"] == 0)
        if do_print:
            _print("\n[TRACE:loop] ------- update_inference_inputs (pre) -------")
            _print(f"input_ids (pre):     {_stats(input_ids)}")
            _print(f"candidates:          {_stats(candidates)}")
            try:
                bc = int(best_candidate.item()) if hasattr(best_candidate,'item') else int(best_candidate)
                al = int(accept_length.item()) if hasattr(accept_length,'item') else int(accept_length)
            except Exception:
                bc, al = best_candidate, accept_length
            _print(f"best_candidate:      {bc}, accept_length={al}")
            # KV-data list (pre)
            try:
                nlayer = len(past_key_values_data_list)
                k0 = past_key_values_data_list[0][0]
                _print(f"kv_data[0].K (pre): {_stats(k0)}  (layers={nlayer})")
                _print(f"current_length:      {_stats(current_length_data)}  {_summ(current_length_data)}")
            except Exception: pass

        t0 = time.perf_counter()
        res = _ORIG["update_inference_inputs"](
            input_ids, candidates, best_candidate, accept_length, retrieve_indices,
            logits_processor, new_token, past_key_values_data_list, current_length_data,
            model, hidden_state_new, sample_p
        )
        t1 = time.perf_counter()

        if do_print:
            try:
                (input_ids_new, draft_tokens, retrieve_indices_new, tree_mask, 
                 tree_position_ids, new_token_out, hidden_state, sample_token) = res
                _print("\n[TRACE:loop] ------- update_inference_inputs (post) -------")
                _print(f"input_ids (post):    {_stats(input_ids_new)}")
                _print(f"draft_tokens:        {_stats(draft_tokens)}")
                _print(f"retrieve_indices:    {_stats(retrieve_indices_new)}")
                _print(f"tree_mask:           {_stats(tree_mask)}")
                try:
                    import torch
                    nnz = int(tree_mask.nonzero().shape[0])
                    _print(f"  tree_mask nnz:     {nnz} / {tree_mask.numel()}  ({nnz/tree_mask.numel():.4%})")
                except Exception: pass
                _print(f"tree_position_ids:   {_stats(tree_position_ids)}")
                _print(f"hidden_state:        {_shape(hidden_state)}")  # often None
                _print(f"sample_token:        {_stats(sample_token)}  {_summ(sample_token)}")
                _print(f"new_token counter:   {new_token_out}")
                _print(f"elapsed:             {(t1-t0)*1e3:.2f} ms")
            except Exception as e:
                _print(f"[TRACE:loop] (post failed: {e})")

        if _STATE["only_first_loop"]:
            disable_ea_trace()
        return res

    # install wrappers
    setattr(em, "initialize_tree", initialize_tree_wrapped)
    setattr(em, "tree_decoding",   tree_decoding_wrapped)
    setattr(em, "evaluate_posterior", evaluate_posterior_wrapped)
    setattr(em, "update_inference_inputs", update_inference_inputs_wrapped)

    _print(f"[ea-trace] enabled (only_first_loop={_STATE['only_first_loop']}, print_every={_STATE['print_every']})")

def disable_ea_trace():
    if not _STATE["enabled"]: return
    try:
        for k, fn in _ORIG.items():
            setattr(em, k, fn)
    finally:
        _ORIG.clear(); _STATE["enabled"] = False
        _print("[ea-trace] disabled")

