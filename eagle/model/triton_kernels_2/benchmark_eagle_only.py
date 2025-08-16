import argparse
import time
from typing import Optional, Tuple, Dict

import torch

from eagle.model.ea_model import EaModel
from eagle.model.triton_kernels_2.ea_model_patch import patch_eagle_model, unpatch_eagle_model
from eagle.model.triton_kernels_2.timer_patch import patch_eagle_timers_only, unpatch_eagle_timers_only
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


def sum_timers(timers: Dict[str, float]) -> float:
    return float(sum(timers.values()))


def main():
    parser = argparse.ArgumentParser("EAGLE-only benchmark: baseline (PyTorch) vs Triton")
    parser.add_argument("--base_model_path", type=str, required=True, help="Base model repo or path")
    parser.add_argument("--ea_model_path", type=str, required=True, help="EAGLE model repo or path")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="New tokens to generate (EAGLE-only timing)")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=list(DTYPE_MAP.keys()), help="Torch dtype")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map, e.g., 'auto'")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (unmeasured)")
    parser.add_argument("--runs", type=int, default=1, help="Measured iterations to average")
    parser.add_argument("--print_output", action="store_true", help="Decode and print generated text for runs (untimed)")
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

    # Warmup baseline (timer-only patch to measure PyTorch EAGLE portion)
    print("Warmup baseline (timer-only patch)...")
    patch_eagle_timers_only(model)
    try:
        for _ in range(max(args.warmup, 0)):
            _ = model.eagenerate(input_ids, max_new_tokens=2)
            cuda_sync()
    finally:
        unpatch_eagle_timers_only(model)

    # Warmup Triton
    print("Warmup Triton (patched)...")
    patch_eagle_model(model)
    try:
        for _ in range(max(args.warmup, 0)):
            _ = model.eagenerate(input_ids, max_new_tokens=2)
            cuda_sync()
    finally:
        unpatch_eagle_model(model)

    # Baseline measurement (PyTorch EAGLE-only)
    print("Measuring baseline (PyTorch EAGLE-only)...")
    patch_eagle_timers_only(model)
    try:
        baseline_total = 0.0
        enable_timers(True)
        for _ in range(max(args.runs, 1)):
            reset_timers()
            cuda_sync()
            _ = model.eagenerate(input_ids, max_new_tokens=args.max_new_tokens)
            cuda_sync()
            timers = get_timers()
            part = sum_timers(timers)
            baseline_total += part
        enable_timers(False)
        baseline_avg = baseline_total / max(args.runs, 1)
        print_timers("[baseline] EAGLE-only timers")
        print(f"[baseline] EAGLE-only total: {baseline_avg:.6f} s")
        if args.print_output:
            try:
                out = model.eagenerate(input_ids, max_new_tokens=args.max_new_tokens)
                if isinstance(out, torch.Tensor):
                    ids = out.detach().to("cpu")
                    txt = tokenizer.decode(ids[0].tolist() if ids.dim() == 2 else ids.tolist(), skip_special_tokens=True)
                    print(f"[baseline] output:\n{txt}\n")
            except Exception:
                pass
    finally:
        unpatch_eagle_timers_only(model)

    # Triton measurement (Triton EAGLE-only)
    print("Measuring Triton (Triton EAGLE-only)...")
    patch_eagle_model(model)
    try:
        triton_total = 0.0
        enable_timers(True)
        for _ in range(max(args.runs, 1)):
            reset_timers()
            cuda_sync()
            _ = model.eagenerate(input_ids, max_new_tokens=args.max_new_tokens)
            cuda_sync()
            timers = get_timers()
            part = sum_timers(timers)
            triton_total += part
        enable_timers(False)
        triton_avg = triton_total / max(args.runs, 1)
        print_timers("[triton] EAGLE-only timers")
        print(f"[triton] EAGLE-only total: {triton_avg:.6f} s")
        if args.print_output:
            try:
                out = model.eagenerate(input_ids, max_new_tokens=args.max_new_tokens)
                if isinstance(out, torch.Tensor):
                    ids = out.detach().to("cpu")
                    txt = tokenizer.decode(ids[0].tolist() if ids.dim() == 2 else ids.tolist(), skip_special_tokens=True)
                    print(f"[triton] output:\n{txt}\n")
            except Exception:
                pass
    finally:
        unpatch_eagle_model(model)

    # Compare EAGLE-only speedup
    speedup = (baseline_avg / max(triton_avg, 1e-12)) if "baseline_avg" in locals() and "triton_avg" in locals() else float("nan")
    delta_pct = ((triton_avg - baseline_avg) / max(baseline_avg, 1e-12) * 100.0) if "baseline_avg" in locals() else float("nan")
    print(f"[compare-eagle-only] baseline_total: {baseline_avg:.6f}s, triton_total: {triton_avg:.6f}s, speedup: {speedup:.3f}x, delta: {delta_pct:.2f}%")

if __name__ == "__main__":
    main()