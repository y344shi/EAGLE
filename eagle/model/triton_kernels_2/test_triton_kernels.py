import torch
import time
import argparse
from pathlib import Path
import sys
import os

# Add the root directory to the Python path
# Get the absolute path to the project root (3 levels up from this file)
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, project_root)

from eagle.model.ea_model import EaModel
from eagle.model.triton_kernels_2.ea_model_patch import patch_eagle_model, unpatch_eagle_model


def test_triton_kernels(base_model_path, ea_model_path, prompt, max_new_tokens=100, use_triton=True):
    """
    Test Triton kernels with a given model and prompt.
    
    Args:
        base_model_path: Path to the base model
        ea_model_path: Path to the EAGLE model
        prompt: Text prompt to generate from
        max_new_tokens: Maximum number of new tokens to generate
        use_triton: Whether to use Triton kernels
    """
    # Load model
    print(f"Loading model from {base_model_path} and {ea_model_path}...")
    model = EaModel.from_pretrained(
        base_model_path=base_model_path, # 'meta-llama/Llama-3.1-8B-Instruct',
        ea_model_path=ea_model_path, #'yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=-1
    )
    model.eval()

    # Apply Triton patches if requested
    if use_triton:
        print("Applying Triton kernel patches...")
        model = patch_eagle_model(model)
    
    # Tokenize prompt
    tokenizer = model.get_tokenizer()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.base_model.device)
    
    # Warm-up run
    print("Performing warm-up run...")
    with torch.no_grad():
        _ = model.eagenerate(input_ids, max_new_tokens=10)
    
    # Benchmark generation
    print(f"Generating with {'Triton' if use_triton else 'PyTorch'} kernels...")
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.eagenerate(input_ids, max_new_tokens=max_new_tokens)
    end_time = time.time()
    
    # Print results
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{output_text}")
    print(f"\nGeneration time: {end_time - start_time:.2f} seconds")
    
    # Remove patches if applied
    if use_triton:
        unpatch_eagle_model(model)
    
    return output_ids, end_time - start_time


def compare_implementations(base_model_path, ea_model_path, prompt, max_new_tokens=100):
    """
    Compare PyTorch and Triton implementations.
    
    Args:
        base_model_path: Path to the base model
        ea_model_path: Path to the EAGLE model
        prompt: Text prompt to generate from
        max_new_tokens: Maximum number of new tokens to generate
    """
    print("=" * 80)
    print("Testing PyTorch implementation...")
    print("=" * 80)
    _, pytorch_time = test_triton_kernels(base_model_path, ea_model_path, prompt, max_new_tokens, use_triton=False)
    
    print("\n" + "=" * 80)
    print("Testing Triton implementation...")
    print("=" * 80)
    _, triton_time = test_triton_kernels(base_model_path, ea_model_path, prompt, max_new_tokens, use_triton=True)
    
    print("\n" + "=" * 80)
    print("Performance comparison:")
    print(f"PyTorch time: {pytorch_time:.2f} seconds")
    print(f"Triton time: {triton_time:.2f} seconds")
    print(f"Speedup: {pytorch_time / triton_time:.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test EAGLE Triton kernels")
    parser.add_argument("--base_model_path", type=str, default='meta-llama/Llama-3.1-8B-Instruct', help="Path to the base model")
    parser.add_argument("--ea_model_path", type=str, default='yuhuili/EAGLE3-LLaMA3.1-Instruct-8B', help="Path to the EAGLE model")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt to generate from")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--compare", action="store_true", help="Compare PyTorch and Triton implementations")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_implementations(args.base_model_path, args.ea_model_path, args.prompt, args.max_new_tokens)
    else:
        test_triton_kernels(args.base_model_path, args.ea_model_path, args.prompt, args.max_new_tokens)