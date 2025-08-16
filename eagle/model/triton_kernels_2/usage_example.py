"""
Example usage of EAGLE Triton kernels.
"""

import torch
from transformers import AutoTokenizer
from eagle.model.ea_model import EaModel
from eagle.model.triton_kernels_2.ea_model_patch import patch_eagle_model, unpatch_eagle_model


def main():
    # Load model
    model_path = "path/to/model"  # Replace with actual model path
    
    print(f"Loading model from {model_path}...")
    model = EaModel.from_pretrained(
        base_model_path=model_path,
        ea_model_path=model_path,
        device_map="auto",
    )
    
    # Apply Triton patches
    print("Applying Triton kernel patches...")
    model = patch_eagle_model(model)
    
    # Prepare input
    tokenizer = model.get_tokenizer()
    prompt = "Once upon a time, in a land far, far away, there lived a"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.base_model.device)
    
    # Generate text
    print(f"Generating text from prompt: '{prompt}'")
    with torch.no_grad():
        output_ids = model.eagenerate(input_ids, max_new_tokens=50)
    
    # Print result
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{output_text}")
    
    # Remove patches
    unpatch_eagle_model(model)


if __name__ == "__main__":
    main()