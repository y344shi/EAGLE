import argparse
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
    parser = argparse.ArgumentParser(description="Compare single-GPU and multi-GPU EAGLE inference")
    parser.add_argument("--base-model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--ea-model", type=str, required=True, help="Path to the EAGLE model")
    parser.add_argument("--use-eagle3", action="store_true", help="Use EAGLE3 instead of EAGLE2")
    parser.add_argument("--base-device", type=str, default="cuda:0", help="Device for the base model")
    parser.add_argument("--draft-device", type=str, default="cuda:1", help="Device for the draft model in multi-GPU mode")
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
    parser.add_argument("--use-tensor-parallel", action="store_true", help="Use tensor parallelism to split base model across 2 GPUs, with draft model on 3rd GPU")
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
                log=True
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
    input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(args.base_device)
    
    print(f"Formatted Prompt: {prompt}")
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
    
    # Measure single-GPU performance first
    print(f"\nMeasuring single-GPU performance ({args.num_runs} runs)...")
    single_gpu_results = measure_generation_time(single_gpu_model, input_ids, args, args.num_runs)
    
    # Use the output from the last measurement run (no need to generate again)
    single_gpu_output = tokenizer.decode(single_gpu_results["last_output_ids"][0], skip_special_tokens=True)
    
    # Print and save single-GPU results summary
    print("\n===== SINGLE-GPU RESULTS SUMMARY =====")
    print(f"Device: {args.base_device}")
    print(f"Mean generation time: {single_gpu_results['mean_time']:.4f}s ± {single_gpu_results['std_time']:.4f}s")
    print(f"Mean tokens generated: {single_gpu_results['mean_tokens']:.1f}")
    print(f"Tokens per second: {single_gpu_results['tokens_per_second']:.2f}")
    print(f"Output length: {len(single_gpu_output)} characters")
    
    # Save single-GPU results to file
    import json
    from datetime import datetime
    
    # Convert tensors to serializable format
    performance_results = {
        "mean_time": float(single_gpu_results["mean_time"]),
        "std_time": float(single_gpu_results["std_time"]),
        "mean_tokens": float(single_gpu_results["mean_tokens"]),
        "tokens_per_second": float(single_gpu_results["tokens_per_second"])
    }
    
    single_gpu_summary = {
        "timestamp": datetime.now().isoformat(),
        "model_config": {
            "base_model": args.base_model,
            "ea_model": args.ea_model,
            "use_eagle3": args.use_eagle3,
            "device": args.base_device,
            "max_new_tokens": args.max_new_tokens,
            "num_runs": args.num_runs
        },
        "performance": performance_results,
        "prompt": args.prompt,
        "output": single_gpu_output,
        "output_length_chars": len(single_gpu_output)
    }
    
    output_file = f"single_gpu_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(single_gpu_summary, f, indent=2, ensure_ascii=False)
    
    print(f"Single-GPU results saved to: {output_file}")
    
    # Clean up single-GPU model to free memory
    print("\nCleaning up single-GPU model...")
    del single_gpu_model
    torch.cuda.empty_cache()
    
    # Load multi-GPU model
    print("\n" + "="*60)
    print("PHASE 2: MULTI-GPU MODEL SETUP")
    print("="*60)
    print_memory_usage("before model loading")
    
    # Force garbage collection before loading
    gc.collect()
    
    if args.use_tensor_parallel:
        print("Using tensor parallelism: Base model split across cuda:0 and cuda:1, draft model on cuda:2")
        multi_gpu_model = load_model_multi_gpu(
            args.base_model,
            args.ea_model,
            use_eagle3=args.use_eagle3,
            use_multi_gpu=True,
            base_device="cuda:0",  # Will be used as reference for tensor parallel split
            draft_device="cuda:2",
            total_token=args.total_token,
            depth=args.depth,
            threshold=args.threshold,
            use_tensor_parallel=True
        )
    else:
        print("Using pipeline parallelism: Base model on one GPU, draft model on another")
        multi_gpu_model = load_model_multi_gpu(
            args.base_model,
            args.ea_model,
            use_eagle3=args.use_eagle3,
            use_multi_gpu=True,
            base_device=args.base_device,
            draft_device=args.draft_device,
            total_token=args.total_token,
            depth=args.depth,
            threshold=args.threshold
        )
    print("Multi-GPU model loaded successfully!")
    print_memory_usage("after model loading")
    
    # Warm-up run for multi-GPU model
    print("\n" + "="*60)
    print("PHASE 3: WARM-UP RUN")
    print("="*60)
    print_memory_usage("before warm-up")
    print("Performing warm-up run (10 tokens)...")
    
    # Force garbage collection before warm-up
    gc.collect()
    
    with torch.no_grad():
        multi_gpu_model.eagenerate(
            input_ids=input_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=10,
            is_llama3=args.is_llama3
        )
    print("Warm-up completed successfully!")
    print_memory_usage("after warm-up")
    
    # Measure multi-GPU performance
    print("\n" + "="*60)
    print("PHASE 4: PERFORMANCE MEASUREMENT")
    print("="*60)
    print_memory_usage("before performance measurement")
    print(f"Measuring multi-GPU performance ({args.num_runs} runs)...")
    
    # Force garbage collection before measurement
    gc.collect()
    
    multi_gpu_results = measure_generation_time(multi_gpu_model, input_ids, args, args.num_runs)
    print("Performance measurement completed!")
    print_memory_usage("after performance measurement")
    
    # Use the output from the last measurement run (no need to generate again)
    multi_gpu_output = tokenizer.decode(multi_gpu_results["last_output_ids"][0], skip_special_tokens=True)
    print("Final text generation completed!")
    print_memory_usage("after final generation")
    
    # Print results
    print("\n===== PERFORMANCE COMPARISON =====")
    print(f"Single-GPU ({args.base_device}):")
    print(f"  Mean time: {single_gpu_results['mean_time']:.4f}s ± {single_gpu_results['std_time']:.4f}s")
    print(f"  Mean tokens: {single_gpu_results['mean_tokens']:.1f}")
    print(f"  Tokens per second: {single_gpu_results['tokens_per_second']:.2f}")
    
    print(f"\nMulti-GPU (Base: {args.base_device}, Draft: {args.draft_device}):")
    print(f"  Mean time: {multi_gpu_results['mean_time']:.4f}s ± {multi_gpu_results['std_time']:.4f}s")
    print(f"  Mean tokens: {multi_gpu_results['mean_tokens']:.1f}")
    print(f"  Tokens per second: {multi_gpu_results['tokens_per_second']:.2f}")
    
    speedup = single_gpu_results['mean_time'] / multi_gpu_results['mean_time']
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Check if outputs match
    outputs_match = single_gpu_output == multi_gpu_output
    print(f"\nOutputs match: {outputs_match}")
    
    # Print outputs
    print("\n===== SINGLE-GPU OUTPUT =====")
    print(single_gpu_output)
    
    print("\n===== MULTI-GPU OUTPUT =====")
    print(multi_gpu_output)
    
    # If outputs don't match, show where they differ
    if not outputs_match:
        print("\n===== DIFFERENCES =====")
        min_len = min(len(single_gpu_output), len(multi_gpu_output))
        for i in range(min_len):
            if single_gpu_output[i] != multi_gpu_output[i]:
                start = max(0, i - 10)
                end_single = min(len(single_gpu_output), i + 10)
                end_multi = min(len(multi_gpu_output), i + 10)
                print(f"First difference at position {i}:")
                print(f"Single-GPU: ...{single_gpu_output[start:end_single]}...")
                print(f"Multi-GPU:  ...{multi_gpu_output[start:end_multi]}...")
                break

if __name__ == "__main__":
    main()