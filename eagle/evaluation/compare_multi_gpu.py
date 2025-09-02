import argparse
import time
import torch
import numpy as np
from transformers import AutoTokenizer
import os
import psutil
import gc

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
    from ..model.multi_gpu_inference import load_model_multi_gpu
except:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import *
    from eagle.model.multi_gpu_inference import load_model_multi_gpu


def get_memory_usage():
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'cpu_percent': cpu_percent
    }

def print_memory_usage(phase=""):
    """Print current memory usage"""
    usage = get_memory_usage()
    print(f"  Memory usage {phase}: RSS={usage['rss_mb']:.1f}MB, VMS={usage['vms_mb']:.1f}MB, CPU={usage['cpu_percent']:.1f}%")

def parse_args():
    parser = argparse.ArgumentParser(description="Compare single-GPU, pipeline parallel, and tensor parallel EAGLE inference")
    parser.add_argument("--base-model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--ea-model", type=str, required=True, help="Path to the EAGLE model")
    parser.add_argument("--use-eagle3", action="store_true", help="Use EAGLE3 instead of EAGLE2")
    parser.add_argument("--run-all", action="store_true", help="Run all three experiments (single-GPU, pipeline parallel, tensor parallel)")
    parser.add_argument("--use-tensor-parallel", action="store_true", help="Use tensor parallelism to split base model across 2 GPUs, with draft model on 3rd GPU")
    parser.add_argument("--base-device", type=str, default="cuda:0", help="Device for the base model")
    parser.add_argument("--draft-device", type=str, default="cuda:1", help="Device for the draft model in multi-GPU mode")
    parser.add_argument("--tp-draft-device", type=str, default="cuda:2", help="Device for the draft model in tensor parallel mode")
    parser.add_argument("--prompt", type=str, default="Write a short story about a robot who learns to feel emotions.", help="Prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs for timing")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.0, help="Top-p for sampling")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k for sampling")
    parser.add_argument("--total-token", type=int, default=60, help="Total tokens for EAGLE")
    parser.add_argument("--depth", type=int, default=7, help="Depth for EAGLE")
    parser.add_argument("--threshold", type=float, default=1.0, help="Threshold for EAGLE")
    parser.add_argument("--is-llama3", action="store_true", help="Whether the model is LLaMA-3")
    return parser.parse_args()

def measure_generation_time(model, input_ids, args, num_runs=5):
    """Measure the generation time for a given model and return the last output."""
    times = []
    tokens = []
    last_output_ids = None
    
    # Warmup runs (same as gen_ea_answer_llama3chat.py)
    print(f"  Performing warmup runs...")
    for warmup_idx in range(3):
        print(f"    Warmup {warmup_idx + 1}/3...")
        torch.cuda.synchronize()
        with torch.no_grad():
            _, _, _ = model.eagenerate(
                input_ids=input_ids,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=min(10, args.max_new_tokens),  # Short warmup
                is_llama3=args.is_llama3,
                log=False # Disable logging for warmup
            )
        torch.cuda.synchronize()
    print(f"  Warmup completed!")
    
    print(f"  Starting {num_runs} performance measurement runs...")
    for run_idx in range(num_runs):
        print(f"    Run {run_idx + 1}/{num_runs}: Starting generation...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output_ids, new_tokens, _ = model.eagenerate(
                input_ids=input_ids,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                is_llama3=args.is_llama3,
                log=True
            )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        run_time = end_time - start_time
        times.append(run_time)
        # Convert tensor to CPU and then to Python int if needed
        if isinstance(new_tokens, torch.Tensor):
            run_tokens = new_tokens.cpu().item()
            tokens.append(run_tokens)
        else:
            run_tokens = new_tokens
            tokens.append(new_tokens)
        
        # Store the output from the last run
        last_output_ids = output_ids
        
        # Avoid division by zero
        tps = (run_tokens / run_time) if run_time > 0 else float('inf')
        print(f"    Run {run_idx + 1}/{num_runs}: Completed in {run_time:.4f}s, generated {run_tokens} tokens ({tps:.2f} tokens/sec)")
    
    print(f"  All {num_runs} runs completed!")
    mean_time = np.mean(times)
    mean_tokens = np.mean(tokens)
    
    return {
        "mean_time": mean_time,
        "std_time": np.std(times),
        "mean_tokens": mean_tokens,
        "tokens_per_second": mean_tokens / mean_time if mean_time > 0 else float('inf'),
        "last_output_ids": last_output_ids
    }

def main():
    args = parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # Set pad token for Llama-3 models to ensure optimal performance
    if "Llama-3" in args.base_model:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare input using the chat template for optimal performance
    messages = [
        {"role": "system", "content": "You are a helpful, respectful and honest assistant."},
        {"role": "user", "content": args.prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Dictionary to store results from all experiments
    all_results = {}
    
    # Determine which experiments to run
    run_single_gpu = True  # Always run single GPU as baseline
    run_pipeline_parallel = args.run_all or (not args.use_tensor_parallel and args.draft_device != args.base_device)
    run_tensor_parallel = args.run_all or args.use_tensor_parallel
    
    print(f"Formatted Prompt: {prompt}")
    
    # ===== EXPERIMENT 1: SINGLE-GPU =====
    if run_single_gpu:
        print("\n" + "="*80)
        print("EXPERIMENT 1: SINGLE-GPU (Base model and EAGLE on same device)")
        print("="*80)
        
        # Create input_ids on the base device
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(args.base_device)
        print(f"Input length: {input_ids.shape[1]} tokens")
        
        # Load single-GPU model
        print("\nLoading single-GPU model...")
        single_gpu_model = load_model_multi_gpu(
            args.base_model,
            args.ea_model,
            use_eagle3=args.use_eagle3,
            use_multi_gpu=False,
            base_device=args.base_device,
            depth=args.depth,
            threshold=args.threshold,
            total_token=args.total_token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Measure single-GPU performance
        print(f"\nMeasuring single-GPU performance ({args.num_runs} runs)...")
        single_gpu_results = measure_generation_time(single_gpu_model, input_ids, args, args.num_runs)
        
        # Use the output from the last measurement run
        single_gpu_output = tokenizer.decode(single_gpu_results["last_output_ids"][0], skip_special_tokens=True)
        
        # Print single-GPU results summary
        print("\n===== SINGLE-GPU RESULTS SUMMARY =====")
        print(f"Device: {args.base_device}")
        print(f"Mean generation time: {single_gpu_results['mean_time']:.4f}s ± {single_gpu_results['std_time']:.4f}s")
        print(f"Mean tokens generated: {single_gpu_results['mean_tokens']:.1f}")
        print(f"Tokens per second: {single_gpu_results['tokens_per_second']:.2f}")
        print(f"Output length: {len(single_gpu_output)} characters")
        
        # Save results to dictionary
        all_results["single_gpu"] = {
            "device": args.base_device,
            "mean_time": float(single_gpu_results["mean_time"]),
            "std_time": float(single_gpu_results["std_time"]),
            "mean_tokens": float(single_gpu_results["mean_tokens"]),
            "tokens_per_second": float(single_gpu_results["tokens_per_second"]),
            "output": single_gpu_output
        }
        
        # Clean up single-GPU model to free memory
        print("\nCleaning up single-GPU model...")
        del single_gpu_model
        torch.cuda.empty_cache()
        gc.collect()
    
    # ===== EXPERIMENT 2: PIPELINE PARALLEL =====
    if run_pipeline_parallel:
        print("\n" + "="*80)
        print(f"EXPERIMENT 2: PIPELINE PARALLEL (Base model on {args.base_device}, EAGLE on {args.draft_device})")
        print("="*80)
        
        # Create input_ids on the base device
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(args.base_device)
        
        print_memory_usage("before pipeline parallel model loading")
        print(f"Loading pipeline parallel model (Base: {args.base_device}, Draft: {args.draft_device})...")
        
        # Force garbage collection before loading
        gc.collect()
        
        # Load pipeline parallel model
        pipeline_model = load_model_multi_gpu(
            args.base_model,
            args.ea_model,
            use_eagle3=args.use_eagle3,
            use_multi_gpu=True,
            base_device=args.base_device,
            draft_device=args.draft_device,
            total_token=args.total_token,
            depth=args.depth,
            threshold=args.threshold,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("Pipeline parallel model loaded successfully!")
        print_memory_usage("after pipeline parallel model loading")
        
        # Warm-up run
        print("\nPerforming warm-up run (10 tokens)...")
        gc.collect()
        with torch.no_grad():
            pipeline_model.eagenerate(
                input_ids=input_ids,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=10,
                is_llama3=args.is_llama3
            )
        print("Warm-up completed successfully!")
        
        # Measure pipeline parallel performance
        print(f"\nMeasuring pipeline parallel performance ({args.num_runs} runs)...")
        gc.collect()
        pipeline_results = measure_generation_time(pipeline_model, input_ids, args, args.num_runs)
        print("Performance measurement completed!")
        
        # Use the output from the last measurement run
        pipeline_output = tokenizer.decode(pipeline_results["last_output_ids"][0], skip_special_tokens=True)
        
        # Print pipeline parallel results summary
        print("\n===== PIPELINE PARALLEL RESULTS SUMMARY =====")
        print(f"Base Device: {args.base_device}, Draft Device: {args.draft_device}")
        print(f"Mean generation time: {pipeline_results['mean_time']:.4f}s ± {pipeline_results['std_time']:.4f}s")
        print(f"Mean tokens generated: {pipeline_results['mean_tokens']:.1f}")
        print(f"Tokens per second: {pipeline_results['tokens_per_second']:.2f}")
        print(f"Output length: {len(pipeline_output)} characters")
        
        # Save results to dictionary
        all_results["pipeline_parallel"] = {
            "base_device": args.base_device,
            "draft_device": args.draft_device,
            "mean_time": float(pipeline_results["mean_time"]),
            "std_time": float(pipeline_results["std_time"]),
            "mean_tokens": float(pipeline_results["mean_tokens"]),
            "tokens_per_second": float(pipeline_results["tokens_per_second"]),
            "output": pipeline_output
        }
        
        # Clean up pipeline parallel model to free memory
        print("\nCleaning up pipeline parallel model...")
        del pipeline_model
        torch.cuda.empty_cache()
        gc.collect()
    
    # ===== EXPERIMENT 3: TENSOR PARALLEL =====
    if run_tensor_parallel:
        print("\n" + "="*80)
        print(f"EXPERIMENT 3: TENSOR PARALLEL (Base model split across cuda:0 and cuda:1, EAGLE on {args.tp_draft_device})")
        print("="*80)
        
        # Create input_ids on cuda:0 (first GPU for tensor parallel)
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda:0")
        
        print_memory_usage("before tensor parallel model loading")
        print(f"Loading tensor parallel model (Base: split across cuda:0 and cuda:1, Draft: {args.tp_draft_device})...")
        
        # Force garbage collection before loading
        gc.collect()
        
        # Load tensor parallel model
        tensor_model = load_model_multi_gpu(
            args.base_model,
            args.ea_model,
            use_eagle3=args.use_eagle3,
            use_multi_gpu=True,
            base_device="cuda:0",  # Reference device for tensor parallel
            draft_device=args.tp_draft_device,
            total_token=args.total_token,
            depth=args.depth,
            threshold=args.threshold,
            use_tensor_parallel=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("Tensor parallel model loaded successfully!")
        print_memory_usage("after tensor parallel model loading")
        
        # Warm-up run
        print("\nPerforming warm-up run (10 tokens)...")
        gc.collect()
        with torch.no_grad():
            tensor_model.eagenerate(
                input_ids=input_ids,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=10,
                is_llama3=args.is_llama3
            )
        print("Warm-up completed successfully!")
        
        # Measure tensor parallel performance
        print(f"\nMeasuring tensor parallel performance ({args.num_runs} runs)...")
        gc.collect()
        tensor_results = measure_generation_time(tensor_model, input_ids, args, args.num_runs)
        print("Performance measurement completed!")
        
        # Use the output from the last measurement run
        tensor_output = tokenizer.decode(tensor_results["last_output_ids"][0], skip_special_tokens=True)
        
        # Print tensor parallel results summary
        print("\n===== TENSOR PARALLEL RESULTS SUMMARY =====")
        print(f"Base Device: split across cuda:0 and cuda:1, Draft Device: {args.tp_draft_device}")
        print(f"Mean generation time: {tensor_results['mean_time']:.4f}s ± {tensor_results['std_time']:.4f}s")
        print(f"Mean tokens generated: {tensor_results['mean_tokens']:.1f}")
        print(f"Tokens per second: {tensor_results['tokens_per_second']:.2f}")
        print(f"Output length: {len(tensor_output)} characters")
        
        # Save results to dictionary
        all_results["tensor_parallel"] = {
            "base_device": "cuda:0,cuda:1",
            "draft_device": args.tp_draft_device,
            "mean_time": float(tensor_results["mean_time"]),
            "std_time": float(tensor_results["std_time"]),
            "mean_tokens": float(tensor_results["mean_tokens"]),
            "tokens_per_second": float(tensor_results["tokens_per_second"]),
            "output": tensor_output
        }
        
        # Clean up tensor parallel model to free memory
        print("\nCleaning up tensor parallel model...")
        del tensor_model
        torch.cuda.empty_cache()
        gc.collect()
    
    # ===== FINAL COMPARISON =====
    print("\n" + "="*80)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Print comparison table header
    print(f"{'Configuration':<25} {'Device(s)':<25} {'Time (s)':<15} {'Tokens/s':<15} {'Speedup':<10}")
    print("-" * 90)
    
    # Single-GPU (baseline for speedup calculation)
    if "single_gpu" in all_results:
        single_gpu_time = all_results["single_gpu"]["mean_time"]
        single_gpu_tokens_per_second = all_results["single_gpu"]["tokens_per_second"]
        print(f"{'Single-GPU':<25} {all_results['single_gpu']['device']:<25} "
              f"{single_gpu_time:.4f} ± {all_results['single_gpu']['std_time']:.4f} "
              f"{single_gpu_tokens_per_second:.2f} {'1.00x':<10}")
    
    # Pipeline Parallel
    if "pipeline_parallel" in all_results:
        pipeline_time = all_results["pipeline_parallel"]["mean_time"]
        pipeline_tokens_per_second = all_results["pipeline_parallel"]["tokens_per_second"]
        pipeline_speedup = single_gpu_time / pipeline_time if "single_gpu" in all_results else float('nan')
        print(f"{'Pipeline Parallel':<25} {f'{all_results['pipeline_parallel']['base_device']}, {all_results['pipeline_parallel']['draft_device']}':<25} "
              f"{pipeline_time:.4f} ± {all_results['pipeline_parallel']['std_time']:.4f} "
              f"{pipeline_tokens_per_second:.2f} {f'{pipeline_speedup:.2f}x':<10}")
    
    # Tensor Parallel
    if "tensor_parallel" in all_results:
        tensor_time = all_results["tensor_parallel"]["mean_time"]
        tensor_tokens_per_second = all_results["tensor_parallel"]["tokens_per_second"]
        tensor_speedup = single_gpu_time / tensor_time if "single_gpu" in all_results else float('nan')
        print(f"{'Tensor Parallel':<25} {f'{all_results['tensor_parallel']['base_device']}, {all_results['tensor_parallel']['draft_device']}':<25} "
              f"{tensor_time:.4f} ± {all_results['tensor_parallel']['std_time']:.4f} "
              f"{tensor_tokens_per_second:.2f} {f'{tensor_speedup:.2f}x':<10}")
    
    # Check if outputs match
    print("\n===== OUTPUT COMPARISON =====")
    
    if "single_gpu" in all_results and "pipeline_parallel" in all_results:
        single_pipeline_match = all_results["single_gpu"]["output"] == all_results["pipeline_parallel"]["output"]
        print(f"Single-GPU and Pipeline Parallel outputs match: {single_pipeline_match}")
    
    if "single_gpu" in all_results and "tensor_parallel" in all_results:
        single_tensor_match = all_results["single_gpu"]["output"] == all_results["tensor_parallel"]["output"]
        print(f"Single-GPU and Tensor Parallel outputs match: {single_tensor_match}")
    
    if "pipeline_parallel" in all_results and "tensor_parallel" in all_results:
        pipeline_tensor_match = all_results["pipeline_parallel"]["output"] == all_results["tensor_parallel"]["output"]
        print(f"Pipeline Parallel and Tensor Parallel outputs match: {pipeline_tensor_match}")
    
    # Save all results to file
    import json
    from datetime import datetime
    
    output_file = f"multi_gpu_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Create summary with metadata
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_config": {
            "base_model": args.base_model,
            "ea_model": args.ea_model,
            "use_eagle3": args.use_eagle3,
            "max_new_tokens": args.max_new_tokens,
            "num_runs": args.num_runs
        },
        "prompt": args.prompt,
        "results": all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nAll results saved to: {output_file}")

if __name__ == "__main__":
    main()