import argparse
import time
import torch
import numpy as np
from transformers import AutoTokenizer

from eagle.model.ea_model import EaModel
from eagle.model.multi_gpu_inference import load_model_multi_gpu

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
    return parser.parse_args()

def measure_generation_time(model, input_ids, args, num_runs=5):
    """Measure the generation time for a given model."""
    times = []
    tokens = []
    
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        
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
        
        times.append(end_time - start_time)
        tokens.append(new_tokens)
    
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "mean_tokens": np.mean(tokens),
        "tokens_per_second": np.mean(tokens) / np.mean(times)
    }

def main():
    args = parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Prepare input
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.base_device)
    
    print(f"Prompt: {args.prompt}")
    print(f"Input length: {input_ids.shape[1]} tokens")
    
    # Load single-GPU model
    print("\nLoading single-GPU model...")
    single_gpu_model = load_model_multi_gpu(
        args.base_model,
        args.ea_model,
        use_eagle3=args.use_eagle3,
        use_multi_gpu=False,
        base_device=args.base_device,
        total_token=args.total_token,
        depth=args.depth,
        threshold=args.threshold
    )
    
    # Load multi-GPU model
    print("\nLoading multi-GPU model...")
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
    
    # Warm-up run
    print("\nPerforming warm-up run...")
    with torch.no_grad():
        single_gpu_model.eagenerate(
            input_ids=input_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=10,
            is_llama3=args.is_llama3
        )
        
        multi_gpu_model.eagenerate(
            input_ids=input_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=10,
            is_llama3=args.is_llama3
        )
    
    # Measure single-GPU performance
    print(f"\nMeasuring single-GPU performance ({args.num_runs} runs)...")
    single_gpu_results = measure_generation_time(single_gpu_model, input_ids, args, args.num_runs)
    
    # Generate text with single-GPU for comparison
    with torch.no_grad():
        single_gpu_output_ids = single_gpu_model.eagenerate(
            input_ids=input_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            is_llama3=args.is_llama3
        )
    single_gpu_output = tokenizer.decode(single_gpu_output_ids[0], skip_special_tokens=True)
    
    # Measure multi-GPU performance
    print(f"\nMeasuring multi-GPU performance ({args.num_runs} runs)...")
    multi_gpu_results = measure_generation_time(multi_gpu_model, input_ids, args, args.num_runs)
    
    # Generate text with multi-GPU for comparison
    with torch.no_grad():
        multi_gpu_output_ids = multi_gpu_model.eagenerate(
            input_ids=input_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            is_llama3=args.is_llama3
        )
    multi_gpu_output = tokenizer.decode(multi_gpu_output_ids[0], skip_special_tokens=True)
    
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