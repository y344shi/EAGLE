# Multi-GPU Inference for EAGLE

This document explains how to use the EAGLE model with multiple GPUs for faster inference.

## Overview

The EAGLE model has been enhanced to support multi-GPU inference, allowing the model to be distributed across multiple GPUs. This can significantly improve inference speed for larger models that don't fit on a single GPU or can benefit from parallel processing.

## Implementation Details

The multi-GPU support is implemented through:

1. **Device Mapping**: The model layers are distributed across available GPUs using Hugging Face's device map functionality.
2. **Distributed KV Cache**: The key-value cache is managed across multiple devices.
3. **Tensor Synchronization**: Critical tensors are synchronized across GPUs when needed.

## Usage

### Basic Usage

```python
from eagle.model.multi_gpu_inference import run_multi_gpu_inference

output = run_multi_gpu_inference(
    base_model_path="meta-llama/Llama-2-7b-chat-hf",
    ea_model_path="path/to/eagle/model",
    input_text="Once upon a time",
    device_map="balanced"  # or "auto", "balanced_low_0", or a custom dict
)

print(output)
```

### Advanced Usage

For more control over the model distribution:

```python
from eagle.model.ea_model import setup_multi_gpu_inference

# Initialize the model with multi-GPU support
model = setup_multi_gpu_inference(
    model_path="meta-llama/Llama-2-7b-chat-hf",
    device_map="balanced",
    use_eagle3=True,
    ea_model_path="path/to/eagle/model",
    total_token=60,
    depth=7,
    top_k=10,
    threshold=1.0
)

# Tokenize input
tokenizer = model.get_tokenizer()
input_ids = tokenizer.encode("Your prompt here", return_tensors="pt")

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
    temperature=0.0,
    top_p=0.0,
    max_new_tokens=512,
    max_length=2048,
    use_distributed=True
)

# Decode output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

## Device Mapping Strategies

- **"auto"**: Automatically determine the best device mapping
- **"balanced"**: Balance memory usage across GPUs
- **"balanced_low_0"**: Balance with less memory on GPU 0 (useful when GPU 0 is also used for other tasks)
- **Custom dict**: Explicitly map model components to specific devices

Example custom device map:
```python
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    "model.layers.2": 0,
    "model.layers.3": 1,
    "model.layers.4": 1,
    "model.layers.5": 1,
    "model.layers.6": 2,
    "model.layers.7": 2,
    "model.norm": 2,
    "lm_head": 2
}
```

## Performance Considerations

1. For smaller models (< 7B parameters), using a single GPU may be more efficient than distributing across multiple GPUs.
2. For medium-sized models (7B-30B parameters), a balanced distribution works well.
3. For large models (> 30B parameters), custom device mapping based on layer sizes may provide the best performance.

## Troubleshooting

If you encounter issues with multi-GPU inference:

1. Ensure NCCL is properly installed for your CUDA version
2. Check that all GPUs are visible to PyTorch with `torch.cuda.device_count()`
3. Try different device mapping strategies
4. Monitor GPU memory usage during inference to identify bottlenecks