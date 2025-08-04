import torch
import time
import argparse
import numpy as np
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    from transformers import AutoTokenizer
    from eagle.model.ea_model import EaModel
    from integration import optimize_eagle_with_triton
except ImportError:
    print("Warning: Could not import required modules. Make sure transformers is installed and eagle module is in the Python path.")

def test_integration(model_name, prompt, max_length=100, num_runs=3, verbose=True):
    """
    Test the integration of Triton-optimized kernels with the EAGLE model.
    
    Parameters:
        model_name: Name or path of the EAGLE model
        prompt: Input prompt for generation
        max_length: Maximum length of generated text
        num_runs: Number of runs for averaging
        verbose: Whether to print detailed output
    
    Returns:
        success: Whether the integration test passed
    """
    try:
        # Load model and tokenizer
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EaModel.from_pretrained(model_name)
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU (tests will be slow)")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with original model
        print("\nGenerating with original model...")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            outputs_original = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                do_sample=False,
                use_cache=True
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        original_time = time.time() - start_time
        
        # Decode output
        original_text = tokenizer.decode(outputs_original[0], skip_special_tokens=True)
        if verbose:
            print(f"Generated text: {original_text}")
        
        # Optimize model with Triton kernels
        print("\nOptimizing model with Triton kernels...")
        model, original_methods = optimize_eagle_with_triton(model)
        
        # Generate with optimized model
        print("\nGenerating with Triton-optimized model...")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            outputs_triton = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                do_sample=False,
                use_cache=True
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        triton_time = time.time() - start_time
        
        # Decode output
        triton_text = tokenizer.decode(outputs_triton[0], skip_special_tokens=True)
        if verbose:
            print(f"Generated text: {triton_text}")
        
        # Compare outputs
        outputs_match = torch.all(outputs_original == outputs_triton).item()
        
        # Print results
        print("\n" + "=" * 80)
        print("INTEGRATION TEST RESULTS")
        print("=" * 80)
        
        print(f"\nOutputs match: {outputs_match}")
        print(f"Original model time: {original_time:.4f} seconds")
        print(f"Triton-optimized model time: {triton_time:.4f} seconds")
        print(f"Speedup: {original_time / triton_time:.2f}x")
        
        # Run multiple times for more accurate timing
        if num_runs > 1:
            print("\nRunning multiple times for more accurate timing...")
            
            # Original model
            original_times = []
            for i in range(num_runs):
                print(f"Original model run {i+1}/{num_runs}...")
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    _ = model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        do_sample=False,
                        use_cache=True
                    )
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                original_times.append(time.time() - start_time)
            
            # Restore original methods
            for layer in model.base_model.model.layers:
                layer.self_attn.forward = original_methods['attention']
            
            # Triton-optimized model
            model, original_methods = optimize_eagle_with_triton(model)
            triton_times = []
            for i in range(num_runs):
                print(f"Triton-optimized model run {i+1}/{num_runs}...")
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    _ = model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        do_sample=False,
                        use_cache=True
                    )
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                triton_times.append(time.time() - start_time)
            
            # Print results
            print("\n" + "=" * 80)
            print("MULTIPLE RUNS RESULTS")
            print("=" * 80)
            
            avg_original_time = np.mean(original_times)
            avg_triton_time = np.mean(triton_times)
            
            print(f"\nAverage original model time: {avg_original_time:.4f} seconds")
            print(f"Average Triton-optimized model time: {avg_triton_time:.4f} seconds")
            print(f"Average speedup: {avg_original_time / avg_triton_time:.2f}x")
            
            print("\nOriginal model times:")
            for i, t in enumerate(original_times):
                print(f"  Run {i+1}: {t:.4f} seconds")
            
            print("\nTriton-optimized model times:")
            for i, t in enumerate(triton_times):
                print(f"  Run {i+1}: {t:.4f} seconds")
        
        # Profile memory usage
        if torch.cuda.is_available():
            print("\n" + "=" * 80)
            print("MEMORY USAGE")
            print("=" * 80)
            
            # Reset model
            for layer in model.base_model.model.layers:
                layer.self_attn.forward = original_methods['attention']
            
            # Original model
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                _ = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    do_sample=False,
                    use_cache=True
                )
            
            original_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
            # Triton-optimized model
            model, original_methods = optimize_eagle_with_triton(model)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                _ = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    do_sample=False,
                    use_cache=True
                )
            
            triton_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
            print(f"\nOriginal model peak memory: {original_memory:.2f} MB")
            print(f"Triton-optimized model peak memory: {triton_memory:.2f} MB")
            print(f"Memory difference: {triton_memory - original_memory:.2f} MB")
        
        return outputs_match
    
    except Exception as e:
        print(f"Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test integration of Triton kernels with EAGLE model")
    parser.add_argument("--model", type=str, default="EAGLE-3B", help="Model name or path")
    parser.add_argument("--prompt", type=str, default="Triton is a language and compiler for writing highly efficient custom CUDA kernels.", help="Input prompt for generation")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs for averaging")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    args = parser.parse_args()
    
    # Run integration test
    success = test_integration(
        args.model,
        args.prompt,
        args.max_length,
        args.num_runs,
        args.verbose
    )
    
    # Exit with appropriate status code
    import sys
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Check if Triton is available
    try:
        import triton
        main()
    except ImportError:
        print("Triton is not installed. Please install it with:")
        print("pip install triton")
        print("For more information, visit: https://github.com/openai/triton")