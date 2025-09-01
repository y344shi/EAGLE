#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eagle_triton_launch.py
Benchmark / launch EAGLE generation with optional Triton fast-paths.
"""

import argparse
import time
from typing import Optional, Tuple
import os

import torch

from eagle.model.ea_model import EaModel
from eagle.model.triton_kernels_3.ea_triton_integration import enable_ea_triton, disable_ea_triton

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None  # type: ignore


DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
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
        tok = model.tokenizer
    else:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is required to build a tokenizer")
        tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    return tok


def prepare_inputs(prompt: str, tokenizer, device) -> torch.Tensor:
    enc = tokenizer(prompt, return_tensors="pt")
    return enc["input_ids"].to(device)


def _decode_ids(ids: torch.Tensor, tokenizer) -> str:
    ids = ids.detach().to("cpu")
    if ids.dim() == 2:
        return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)
    elif ids.dim() == 1:
        return tokenizer.decode(ids.tolist(), skip_special_tokens=True)
    return ""


def _collect_output_text_from_iterable(item, tokenizer) -> Optional[str]:
    ids = None
    if isinstance(item, torch.Tensor):
        ids = item
    elif isinstance(item, dict):
        for key in ("input_ids", "sequences", "tokens", "output_ids"):
            v = item.get(key)
            if isinstance(v, torch.Tensor):
                ids = v; break
    elif isinstance(item, (tuple, list)):
        for v in item:
            if isinstance(v, torch.Tensor):
                ids = v; break
    return _decode_ids(ids, tokenizer) if isinstance(ids, torch.Tensor) else None


def _call_ea_generate(model: EaModel, input_ids: torch.Tensor, max_new_tokens: int):
    if hasattr(model, "ea_generate"):
        return model.ea_generate(input_ids, max_new_tokens=max_new_tokens)
    if hasattr(model, "eagenerate"):
        return model.eagenerate(input_ids, max_new_tokens=max_new_tokens)
    raise AttributeError("EaModel has neither ea_generate() nor eagenerate()")


def _drain_decode(out, init_len: int) -> int:
    last_ids = None
    if isinstance(out, torch.Tensor):
        last_ids = out
    elif hasattr(out, "__iter__"):
        for step in out:
            if isinstance(step, torch.Tensor):
                last_ids = step
            elif isinstance(step, dict):
                ids = (step.get("input_ids") or step.get("sequences") or
                       step.get("tokens") or step.get("output_ids"))
                if isinstance(ids, torch.Tensor):
                    last_ids = ids
            elif isinstance(step, (list, tuple)):
                for v in step:
                    if isinstance(v, torch.Tensor):
                        last_ids = v
                        break
    if isinstance(last_ids, torch.Tensor):
        ids = last_ids.detach()
        if ids.dim() == 2:
            return max(0, int(ids.shape[1]) - int(init_len))
        if ids.dim() == 1:
            return max(0, int(ids.numel()) - int(init_len))
    return 0


def generate_text(model: EaModel, tokenizer, input_ids: torch.Tensor, max_new_tokens: int) -> Optional[str]:
    try:
        out = _call_ea_generate(model, input_ids, max_new_tokens)
        if hasattr(out, "__iter__") and not isinstance(out, torch.Tensor):
            last = None
            for step in out:
                last = step
            return _collect_output_text_from_iterable(last, tokenizer) if last is not None else None
        if isinstance(out, torch.Tensor):
            return _decode_ids(out, tokenizer)
        return None
    except Exception:
        return None


def run_once(model: EaModel, input_ids: torch.Tensor, max_new_tokens: int) -> tuple[float, int]:
    init_len = int(input_ids.shape[1])
    cuda_sync()
    t0 = time.perf_counter()
    out = _call_ea_generate(model, input_ids, max_new_tokens)
    produced = _drain_decode(out, init_len=init_len)
    cuda_sync()
    dt = time.perf_counter() - t0
    return dt, produced


def main():
    parser = argparse.ArgumentParser("EAGLE Triton launcher / benchmark")
    parser.add_argument("--base_model_path", type=str, required=True, help="Base model repo or path")
    parser.add_argument("--ea_model_path",   type=str, required=True, help="EAGLE model repo or path")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Tokens to generate")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured warmups")
    parser.add_argument("--runs", type=int, default=1, help="Measured iterations")
    parser.add_argument("--compare", action="store_true", help="Run baseline then Triton")
    parser.add_argument("--triton_only", action="store_true", help="Run Triton only (skip baseline)")
    parser.add_argument("--print_output", action="store_true", help="Decode and print generated text for the last run")
    parser.add_argument("--no_fused_attn", action="store_true")
    parser.add_argument("--no_gather", action="store_true")
    parser.add_argument("--no_eval_post", action="store_true")
    parser.add_argument("--no_kv_scatter", action="store_true")
    parser.add_argument("--no-trace", action="store_true", help="Disable verbose tracing from ea_trace.py")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler and print a summary table.")
    args = parser.parse_args()

    if args.no_trace:
        try:
            from eagle.model import ea_trace
            ea_trace.enable_ea_trace = lambda *a, **kw: None
            ea_trace.disable_ea_trace()
        except Exception:
            pass

    model = load_model(args.base_model_path, args.ea_model_path, args.torch_dtype, args.device_map)
    device = next(model.parameters()).device
    tokenizer = get_tokenizer(args.base_model_path, model)
    input_ids = prepare_inputs(args.prompt, tokenizer, device)

    def measure_block(do_triton: bool, label: str):
        if do_triton:
            enable_ea_triton(
                model,
                use_fused_attn=not args.no_fused_attn,
                use_gather=not args.no_gather,
                use_eval_post=not args.no_eval_post,
                use_kv_scatter=not args.no_kv_scatter,
            )
        else:
            disable_ea_triton(model)

        try:
            total_t = 0.0
            total_tok = 0
            for _ in range(max(args.runs, 1)):
                dt, produced = run_once(model, input_ids, args.max_new_tokens)
                total_t += dt
                total_tok += max(produced, 1)
            avg_dt = total_t / max(args.runs, 1)
            avg_tok = total_tok / max(args.runs, 1)
            tps = avg_tok / max(avg_dt, 1e-12)

            print(f"[{label}] time={avg_dt*1000:.3f} ms  produced={avg_tok:.1f}  tok/s={tps:.2f}")

            if args.print_output:
                text = generate_text(model, tokenizer, input_ids, args.max_new_tokens)
                print(f"[{label}] output:\n{text or '<unavailable>'}\n")
            return avg_dt, tps
        finally:
            if do_triton:
                disable_ea_triton(model)

    if args.profile:
        args.runs = 1
        args.warmup = 0

        def profiler_runner(do_triton: bool, label: str):
            print(f"--- PROFILING {label.upper()} (1 run) ---")
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            ) as prof:
                measure_block(do_triton=do_triton, label=label)

            print("--- Profiler Results (Table) ---")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        if not args.triton_only:
            profiler_runner(do_triton=False, label="baseline")
        profiler_runner(do_triton=True, label="triton")

    else:
        def warmup(do_triton: bool):
            if do_triton:
                enable_ea_triton(model)
            try:
                for _ in range(max(args.warmup, 0)):
                    out = _call_ea_generate(model, input_ids, max_new_tokens=2)
                    _ = _drain_decode(out, init_len=input_ids.shape[1])
                    cuda_sync()
            finally:
                if do_triton:
                    disable_ea_triton(model)

        results = {}
        if args.compare and not args.triton_only:
            ###  baseline
            print("Warmup baseline...")
            warmup(do_triton=False)
            print("Benchmarking baseline...")

            results["baseline"] = measure_block(do_triton=False, label="baseline")
            ###  Triton
            print("Warmup Triton...")
            warmup(do_triton=True)
            print("Benchmarking Triton...")
            results["triton"]   = measure_block(do_triton=True,  label="triton")

            b = results["baseline"][0]; t = results["triton"][0]
            speedup = b / max(t, 1e-9)
            print(f"[compare] baseline={b:.4f}s  triton={t:.4f}s  speedup x{speedup:.3f}")
        else:
            print("Warmup baseline...")
            warmup(do_triton=True)
            print("Benchmarking Triton...")
            measure_block(do_triton=True, label="triton")


if __name__ == "__main__":
    main()