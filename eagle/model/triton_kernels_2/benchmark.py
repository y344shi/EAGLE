import argparse
import time
from typing import Optional, Tuple

import torch

from eagle.model.ea_model import EaModel
from eagle.model.triton_kernels_2.ea_model_patch import patch_eagle_model, unpatch_eagle_model
from eagle.model.triton_kernels_2.timers import reset_timers, print_timers, get_timers, enable_timers

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None  # type: ignore


DTYPE_MAP = {
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def cuda_sync():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def load_model(
    base_model_path: str,
    ea_model_path: str,
    torch_dtype: str = "float16",
    device_map: str = "auto",
) -> EaModel:
    dtype = DTYPE_MAP.get(torch_dtype.lower(), torch.float16)
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        total_token=-1,
    )
    return model


def get_tokenizer(base_model_path: str, model: EaModel):
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        return model.tokenizer
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required to build a tokenizer but is not available")
    tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    return tok


def prepare_inputs(prompt: str, tokenizer, device) -> torch.Tensor:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    return input_ids


def _collect_output_text_from_iterable(item, tokenizer) -> Optional[str]:
    # Try to find a tensor of ids within common containers
    ids = None
    if isinstance(item, torch.Tensor):
        ids = item
    elif isinstance(item, dict):
        for key in ("input_ids", "sequences", "tokens", "output_ids"):
            v = item.get(key)
            if isinstance(v, torch.Tensor):
                ids = v
                break
    elif isinstance(item, (tuple, list)):
        for v in item:
            if isinstance(v, torch.Tensor):
                ids = v
                break
    if ids is None or not isinstance(ids, torch.Tensor):
        return None
    ids = ids.detach().to("cpu")
    if ids.dim() == 2:
        return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)
    elif ids.dim() == 1:
        return tokenizer.decode(ids.tolist(), skip_special_tokens=True)
    return None


def generate_text(model: EaModel, tokenizer, input_ids: torch.Tensor, max_new_tokens: int) -> Optional[str]:
    """
    Runs eagenerate once and tries to decode the final output text.
    Works whether eagenerate returns a Tensor or an iterable of steps.
    This call is NOT timed; use measured functions for performance timing.
    """
    try:
        out = model.eagenerate(input_ids, max_new_tokens=max_new_tokens)
        # If it's an iterable (generator), iterate and pick last item
        if hasattr(out, "__iter__") and not isinstance(out, torch.Tensor):
            last = None
            for step in out:
                last = step
            if last is None:
                return None
            return _collect_output_text_from_iterable(last, tokenizer)
        # If it's directly a tensor of ids
        if isinstance(out, torch.Tensor):
            ids = out.detach().to("cpu")
            if ids.dim() == 2:
                return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)
            elif ids.dim() == 1:
                return tokenizer.decode(ids.tolist(), skip_special_tokens=True)
        # Unknown structure
        return None
    except Exception:
        return None


def run_generate(model: EaModel, input_ids: torch.Tensor, max_new_tokens: int) -> Tuple[float, Optional[dict]]:
    # End-to-end timing (timers OFF to avoid sync overhead)
    enable_timers(False)
    cuda_sync()
    t0 = time.perf_counter()
    _ = model.eagenerate(input_ids, max_new_tokens=max_new_tokens)
    cuda_sync()
    dt = time.perf_counter() - t0
    enable_timers(False)
    return dt, None


def run_generate_with_eagle_timers(model: EaModel, input_ids: torch.Tensor, max_new_tokens: int) -> Tuple[float, dict]:
    # Reset EAGLE-only timers and run (timers ON only for this call)
    reset_timers()
    enable_timers(True)
    cuda_sync()
    t0 = time.perf_counter()
    _ = model.eagenerate(input_ids, max_new_tokens=max_new_tokens)
    cuda_sync()
    dt = time.perf_counter() - t0
    timers = get_timers()
    enable_timers(False)
    return dt, timers


def main():
    parser = argparse.ArgumentParser("EAGLE Triton benchmark (supports large token counts)")
    parser.add_argument("--base_model_path", type=str, required=True, help="Base model repo or path")
    parser.add_argument("--ea_model_path", type=str, required=True, help="EAGLE model repo or path")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="New tokens to generate (large supported)")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=list(DTYPE_MAP.keys()), help="Torch dtype")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map, e.g., 'auto'")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (unmeasured)")
    parser.add_argument("--runs", type=int, default=1, help="Measured iterations to average")
    parser.add_argument("--compare", action="store_true", help="Run baseline PyTorch then Triton-patched")
    parser.add_argument("--print_eagle_timers", action="store_true", help="Print EAGLE-only timers (Triton only)")
    parser.add_argument("--print_output", action="store_true", help="Decode and print generated text for the last run")
    args = parser.parse_args()

    model = load_model(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    device = next(model.parameters()).device
    tokenizer = get_tokenizer(args.base_model_path, model)
    input_ids = prepare_inputs(args.prompt, tokenizer, device)

    def do_warmup(do_patch: bool):
        if do_patch:
            patch_eagle_model(model)
        try:
            for _ in range(max(args.warmup, 0)):
                _ = model.eagenerate(input_ids, max_new_tokens=2)
                cuda_sync()
        finally:
            if do_patch:
                unpatch_eagle_model(model)

    def measure_block(do_patch: bool, label: str):
        if do_patch:
            patch_eagle_model(model)
        try:
            total_dt = 0.0
            for _ in range(max(args.runs, 1)):
                if do_patch and args.print_eagle_timers:
                    dt, _ = run_generate_with_eagle_timers(model, input_ids, args.max_new_tokens)
                else:
                    dt, _ = run_generate(model, input_ids, args.max_new_tokens)
                total_dt += dt

            avg_dt = total_dt / max(args.runs, 1)
            tok_per_s = float(args.max_new_tokens) / max(avg_dt, 1e-9)
            print(f"[{label}] avg_time: {avg_dt:.4f}s, tokens: {args.max_new_tokens}, tokens/s: {tok_per_s:.2f}")
            if do_patch and args.print_eagle_timers:
                print_timers(f"[{label}] EAGLE-only Triton timers")

            # Optionally print generated text (not included in timing)
            if args.print_output:
                text = generate_text(model, tokenizer, input_ids, args.max_new_tokens)
                if text is not None:
                    print(f"[{label}] output:\n{text}\n")
                else:
                    print(f"[{label}] output: <unavailable>\n")

            return avg_dt, tok_per_s
        finally:
            if do_patch:
                unpatch_eagle_model(model)

    # Warmups
    print("Warmup baseline (unpatched)...")
    do_warmup(do_patch=False)

    results = {}

    if args.compare:
        print("Warmup Triton (patched)...")
        do_warmup(do_patch=True)

        print("Benchmarking baseline...")
        base_dt, base_tps = measure_block(do_patch=False, label="baseline")
        results["baseline"] = (base_dt, base_tps)

        print("Benchmarking Triton...")
        tri_dt, tri_tps = measure_block(do_patch=True, label="triton")
        results["triton"] = (tri_dt, tri_tps)

        if "baseline" in results and "triton" in results:
            base_dt, base_tps = results["baseline"]
            tri_dt, tri_tps = results["triton"]
            speedup = base_dt / max(tri_dt, 1e-9)
            diff_pct = (tri_dt - base_dt) / max(base_dt, 1e-9) * 100.0
            print(f"[compare] baseline_time: {base_dt:.4f}s, triton_time: {tri_dt:.4f}s, speedup: {speedup:.3f}x, delta: {diff_pct:.2f}%")
            print(f"[compare] baseline_tps: {base_tps:.2f}, triton_tps: {tri_tps:.2f}, delta_tps: {tri_tps - base_tps:.2f}")
    else:
        # Single mode: Triton only
        print("Warmup Triton (patched)...")
        do_warmup(do_patch=True)

        print("Benchmarking Triton...")
        _ = measure_block(do_patch=True, label="triton")


if __name__ == "__main__":
    main()