# Fused Transformer Kernel for Draft Token Generation

This directory contains a fused transformer kernel implementation for draft token generation in the EAGLE model. The implementation uses Triton kernels for GPU acceleration when available, and falls back to PyTorch implementations when necessary.

## Components

- `fused_transformer.py`: Contains the core implementation of the fused transformer kernel.
- `fused_draft_kernel.py`: Contains the implementation of the fused draft token generation kernel.
- `gpu_assert.py`: Contains utilities for asserting that operations are performed on GPU.
- `test_draft_token.py`: Contains tests for the draft token generation components.
- `test_fused_draft.py`: Contains tests for the fused draft kernel.
- `test_integration.py`: Contains integration tests for the entire pipeline.

## Usage

To use the fused kernel for draft token generation:

```python
from eagle.model.triton_kernels.fused_draft_kernel import fused_draft_token_generation

# Load model
model = EaModel.from_pretrained(...)

# Generate tokens
input_ids = tokenizer.encode("Hello, world!", return_tensors="pt").cuda()
output_ids = fused_draft_token_generation(
    model=model,
    input_ids=input_ids,
    temperature=0.0,
    top_p=0.0,
    top_k=0.0,
    max_new_tokens=20,
    max_length=2048,
)

# Decode output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

## Implementation Details

The fused kernel implementation combines several steps of the draft token generation pipeline:

1. Forward pass through transformer layers
2. RoPE embeddings
3. Attention computation
4. KV-cache management
5. Draft token generation and verification

The implementation uses Triton kernels for GPU acceleration when available, and falls back to PyTorch implementations when necessary.

## GPU Requirements

The fused kernel implementation requires a CUDA-capable GPU. If CUDA is not available, the implementation will raise a `GPUAssertionError`.

## Testing

To run the tests:

```bash
pytest -xvs eagle/model/triton_kernels/test_draft_token.py
pytest -xvs eagle/model/triton_kernels/test_fused_draft.py
pytest -xvs eagle/model/triton_kernels/test_integration.py