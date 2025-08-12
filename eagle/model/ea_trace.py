"""EAGLE/toy_warmup.py
ea_trace.py — Monkey-patch tracer for EaModel.ea_generate()

What it does
------------
- Wraps the helpers that ea_generate() calls:
  - initialize_tree
  - tree_decoding
  - evaluate_posterior
  - update_inference_inputs
- Prints key tensor shapes & a few values for the FIRST iteration only
  (configurable).

How to use
----------
from ea_trace import enable_ea_trace, disable_ea_trace
enable_ea_trace(only_first_loop=True)   # install wrappers
# ... run your warmup()/ea_generate() as usual ...
disable_ea_trace()                      # restore originals when done
"""

import importlib
import builtins

# We'll patch symbols on the module where ea_generate resolves them from.
# In your repo, ea_model.py does: `from .utils import *`
# so calls like initialize_tree(...) resolve on the `ea_model` module namespace.
#
# Thus we patch on `ea_model` (NOT on `utils`), so that ea_generate() picks them up.
#
# If your package name is different, adjust the import below.
try:
    em = importlib.import_module("ea_model")        # local module name variant
except Exception:
    # If package-style import (e.g., project.ea_model) is needed, let users modify here.
    em = importlib.import_module(".ea_model")       # may fail; user can edit as needed


_ORIG = {}
_STATE = {
    "enabled": False,
    "loop_count": 0,
    "only_first_loop": True,
    "print_every": 1,
}

def _print(*args, **kwargs):
    # Safe print that won't break if tensors aren't available in this interpreter.
    print(*args, **kwargs, flush=True)


def _shape(x):
    try:
        return tuple(x.shape)
    except Exception:
        try:
            return len(x)
        except Exception:
            return "n/a"


def _device_dtype(x):
    dev = getattr(x, "device", None)
    dt  = getattr(x, "dtype", None)
    return f"device={dev}, dtype={dt}"


def enable_ea_trace(print_every=1, only_first_loop=True):
    """
    Enable monkey-patched tracing.

    Args:
      print_every: print on every Nth loop (default 1 == every loop)
      only_first_loop: if True, automatically disable after first full ea_generate() loop
    """
    if _STATE["enabled"]:
        return
    _STATE["enabled"] = True
    _STATE["loop_count"] = 0
    _STATE["only_first_loop"] = bool(only_first_loop)
    _STATE["print_every"] = max(1, int(print_every))

    # keep originals
    _ORIG["initialize_tree"] = getattr(em, "initialize_tree")
    _ORIG["tree_decoding"] = getattr(em, "tree_decoding")
    _ORIG["evaluate_posterior"] = getattr(em, "evaluate_posterior")
    _ORIG["update_inference_inputs"] = getattr(em, "update_inference_inputs")

    # wrap
    def initialize_tree_wrapped(input_ids, model, past_key_values, logits_processor):
        res = _ORIG["initialize_tree"](input_ids, model, past_key_values, logits_processor)
        try:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = res
            _print("\n[TRACE:init] ------- initialize_tree -------")
            _print(f"input_ids:           {_shape(input_ids)}")
            _print(f"draft_tokens:        {_shape(draft_tokens)}")
            _print(f"retrieve_indices:    {_shape(retrieve_indices)}")
            _print(f"tree_mask:           {_shape(tree_mask)}")
            _print(f"tree_position_ids:   {_shape(tree_position_ids)}")
            _print(f"orig logits:         {_shape(logits)}")
            _print(f"hidden_state:        {_shape(hidden_state)}")
            _print(f"sample_token:        {_shape(sample_token)}")
        except Exception as e:
            _print(f"[TRACE:init] (printing failed: {e})")
        return res

    def tree_decoding_wrapped(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices):
        res = _ORIG["tree_decoding"](model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices)
        try:
            logits, hidden_state_new, outputs = res
            _print("\n[TRACE:loop] ------- tree_decoding -------")
            _print(f"tree_candidates:     {_shape(tree_candidates)}  ({_device_dtype(tree_candidates)})")
            _print(f"retrieve_indices:    {_shape(retrieve_indices)}")
            _print(f"tree_position_ids:   {_shape(tree_position_ids)}")
            _print(f"logits (sliced):     {_shape(logits)}")
            _print(f"hidden_state_new:    {_shape(hidden_state_new)}")
        except Exception as e:
            _print(f"[TRACE:loop] (printing failed: {e})")
        return res

    def evaluate_posterior_wrapped(logits, candidates, logits_processor):
        res = _ORIG["evaluate_posterior"](logits, candidates, logits_processor)
        try:
            best_candidate, accept_length, sample_p = res
            _print("\n[TRACE:loop] ------- evaluate_posterior -------")
            _print(f"logits:              {_shape(logits)}")
            _print(f"candidates:          {_shape(candidates)}")
            # Scalars might be 0-d tensors; make them printable
            try:
                bc = int(best_candidate.item()) if hasattr(best_candidate, 'item') else int(best_candidate)
            except Exception:
                bc = best_candidate
            try:
                al = int(accept_length.item()) if hasattr(accept_length, 'item') else int(accept_length)
            except Exception:
                al = accept_length
            _print(f"best_candidate:      {bc}")
            _print(f"accept_length:       {al}")
            _print(f"sample_p:            {_shape(sample_p)}")
        except Exception as e:
            _print(f"[TRACE:loop] (printing failed: {e})")
        return res

    def update_inference_inputs_wrapped(
        input_ids, candidates, best_candidate, accept_length, retrieve_indices,
        logits_processor, new_token, past_key_values_data_list, current_length_data,
        model, hidden_state_new, sample_p
    ):
        # Pre
        try:
            _STATE["loop_count"] += 1
            do_print = (_STATE["loop_count"] % _STATE["print_every"] == 0)
            if do_print:
                _print("\n[TRACE:loop] ------- update_inference_inputs (pre) -------")
                _print(f"input_ids (pre):     {_shape(input_ids)}")
                _print(f"candidates:          {_shape(candidates)}")
        except Exception as e:
            _print(f"[TRACE:loop] (pre failed: {e})")

        res = _ORIG["update_inference_inputs"](
            input_ids, candidates, best_candidate, accept_length, retrieve_indices,
            logits_processor, new_token, past_key_values_data_list, current_length_data,
            model, hidden_state_new, sample_p
        )

        try:
            input_ids_new, draft_tokens, retrieve_indices_new, tree_mask, tree_position_ids, new_token_out, hidden_state, sample_token = res
            _print("\n[TRACE:loop] ------- update_inference_inputs (post) -------")
            _print(f"input_ids (post):    {_shape(input_ids_new)}")
            _print(f"draft_tokens:        {_shape(draft_tokens)}")
            _print(f"retrieve_indices:    {_shape(retrieve_indices_new)}")
            _print(f"tree_mask:           {_shape(tree_mask)}")
            _print(f"tree_position_ids:   {_shape(tree_position_ids)}")
            _print(f"hidden_state:        {_shape(hidden_state)}")
            _print(f"sample_token:        {_shape(sample_token)}")
            _print(f"new_token counter:   {new_token_out}")
        except Exception as e:
            _print(f"[TRACE:loop] (post failed: {e})")

        # Auto-disable after first loop if requested
        if _STATE["only_first_loop"]:
            disable_ea_trace()

        return res

    # install wrappers
    setattr(em, "initialize_tree", initialize_tree_wrapped)
    setattr(em, "tree_decoding", tree_decoding_wrapped)
    setattr(em, "evaluate_posterior", evaluate_posterior_wrapped)
    setattr(em, "update_inference_inputs", update_inference_inputs_wrapped)

    _print("[ea-trace] enabled (only_first_loop=%s, print_every=%d)" % (_STATE["only_first_loop"], _STATE["print_every"]))


def disable_ea_trace():
    """Restore original functions."""
    if not _STATE["enabled"]:
        return
    try:
        for k, fn in _ORIG.items():
            setattr(em, k, fn)
    finally:
        _ORIG.clear()
        _STATE["enabled"] = False
        _print("[ea-trace] disabled")
