import torch
import os
from .ea_model import EaModel, setup_multi_gpu_inference

def run_multi_gpu_inference(
    base_model_path,
    ea_model_path,
    input_text,
    use_eagle3=True,
    total_token=60,
    depth=7,
    top_k=10,
    threshold=1.0,
    temperature=0.0,
    top_p=0.0,
    max_new_tokens=512,
    max_length=2048,
    device_map="auto"
):
    """
    Run inference on the EAGLE model using multiple GPUs.
    
    Args:
        base_model_path (str): Path to the base model
        ea_model_path (str): Path to the EAGLE model
        input_text (str): Input text to generate from
        use_eagle3 (bool): Whether to use EAGLE3 or not
        total_token (int): Total number of tokens for speculative decoding
        depth (int): Depth of the tree for speculative decoding
        top_k (int): Top-k for speculative decoding
        threshold (float): Threshold for speculative decoding
        temperature (float): Temperature for sampling
        top_p (float): Top-p for sampling
        max_new_tokens (int): Maximum number of new tokens to generate
        max_length (int): Maximum length of the sequence
        device_map (str or dict): Device mapping strategy
        
    Returns:
        str: Generated text
    """
    # Check if we have multiple GPUs
    if torch.cuda.device_count() <= 1:
        print(f"Only {torch.cuda.device_count()} GPU detected. Using single GPU inference.")
        model = EaModel.from_pretrained(
            use_eagle3=use_eagle3,
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold
        )
    else:
        print(f"Found {torch.cuda.device_count()} GPUs. Using multi-GPU inference.")
        # Initialize distributed environment if not already done
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            
        # Load model with device map
        model = setup_multi_gpu_inference(
            model_path=base_model_path,
            device_map=device_map,
            use_eagle3=use_eagle3,
            ea_model_path=ea_model_path,
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold
        )
    
    # Tokenize input
    tokenizer = model.get_tokenizer()
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Move input to the appropriate device
    if hasattr(model, "device_map") and model.device_map is not None:
        if isinstance(model.device_map, dict):
            first_device = next(iter(model.device_map.values()))
            if isinstance(first_device, list):
                first_device = first_device[0]
            input_ids = input_ids.to(first_device)
        else:
            input_ids = input_ids.to(model.device_map)
    else:
        input_ids = input_ids.to(next(model.parameters()).device)
    
    # Generate text
    output_ids = model.eagenerate(
        input_ids=input_ids,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        use_distributed=True
    )
    
    # Decode output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return output_text

def get_optimal_device_map(model_size_in_billions):
    """
    Get an optimal device map based on model size and available GPUs.
    
    Args:
        model_size_in_billions (int): Size of the model in billions of parameters
        
    Returns:
        str or dict: Device map strategy
    """
    num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        return "auto"
    
    # For smaller models that can fit on a single GPU, use the first GPU
    if model_size_in_billions < 7:
        return "cuda:0"
    
    # For medium-sized models, use a balanced strategy
    if model_size_in_billions < 30:
        return "balanced"
    
    # For large models, use a custom device map
    # This is a simplified example - in practice, you'd want to consider
    # the memory of each GPU and distribute layers accordingly
    return "balanced_low_0"

if __name__ == "__main__":
    # Example usage
    base_model_path = "meta-llama/Llama-2-7b-chat-hf"
    ea_model_path = "path/to/eagle/model"
    input_text = "Once upon a time"
    
    # Get optimal device map based on model size
    device_map = get_optimal_device_map(7)  # 7B model
    
    output = run_multi_gpu_inference(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        input_text=input_text,
        device_map=device_map
    )
    
    print(output)