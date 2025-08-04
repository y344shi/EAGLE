import torch
from transformers import AutoTokenizer
from eagle.model.ea_model import EaModel
from eagle.model.triton_kernels import optimize_eagle_with_triton

def main():
    """
    Example of how to use Triton-optimized kernels with the EAGLE model.
    """
    # Load model and tokenizer
    model_name = "EAGLE-3B"  # Replace with actual model name/path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load EAGLE model
    model = EaModel.from_pretrained(model_name)
    model.eval()
    
    # Optimize model with Triton kernels
    model, original_methods = optimize_eagle_with_triton(model)
    
    # Prepare input
    prompt = "Triton is a language and compiler for writing highly efficient custom CUDA kernels."
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text with optimized model
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=100,
            do_sample=False,
            use_cache=True
        )
    
    # Decode and print output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    
    # Performance comparison
    import time
    
    # Measure time with Triton optimization
    start_time = time.time()
    with torch.no_grad():
        outputs_triton = model.generate(
            inputs["input_ids"],
            max_length=100,
            do_sample=False,
            use_cache=True
        )
    triton_time = time.time() - start_time
    
    # Restore original methods
    # This is a simplified version - in practice, you would need to restore all methods
    for layer in model.base_model.model.layers:
        layer.self_attn.forward = original_methods['attention']
    
    # Measure time without Triton optimization
    start_time = time.time()
    with torch.no_grad():
        outputs_original = model.generate(
            inputs["input_ids"],
            max_length=100,
            do_sample=False,
            use_cache=True
        )
    original_time = time.time() - start_time
    
    # Print performance comparison
    print(f"Time with Triton optimization: {triton_time:.4f} seconds")
    print(f"Time without Triton optimization: {original_time:.4f} seconds")
    print(f"Speedup: {original_time / triton_time:.2f}x")


if __name__ == "__main__":
    # Check if Triton is available
    try:
        import triton
        main()
    except ImportError:
        print("Triton is not installed. Please install it with:")
        print("pip install triton")
        print("For more information, visit: https://github.com/openai/triton")