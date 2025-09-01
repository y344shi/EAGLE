# Triton Drafter for EAGLE

A high-performance implementation of EAGLE's draft token generation using Triton kernels.

## Overview

This module provides a Triton-based implementation of the draft token generation process in EAGLE. It replaces the Python-based implementation with a single persistent kernel that handles the entire drafting process, significantly improving performance.

## Features

- Single persistent kernel for the entire drafting process
- FlashAttention-style tree attention for efficient computation
- Streaming top-k to avoid materializing full logits
- Implicit tree masking through contiguous ancestor storage
- Modular design for easy integration

## Installation

Ensure you have the required dependencies:

```bash
pip install torch>=2.0.0 triton>=2.0.0
```

## Usage

### Direct Integration with EA Model

```python
from eagle.model.triton_drafttoken_gen.ea_integration import patch_ea_model_with_triton_drafter

# Load your EA model
model = ...

# Patch the model to use the Triton drafter
model = patch_ea_model_with_triton_drafter(model, use_triton=True)

# Use the model as usual
outputs = model.generate(...)
```

### Through the Frontier API

```python
from eagle.model.triton_drafttoken_gen.frontier_api import FrontierConfig, frontier_generate
from eagle.model.triton_drafttoken_gen.frontier_integration import register_triton_backend

# Register the Triton backend
register_triton_backend()

# Create a frontier config
cfg = FrontierConfig(
    total_token=60,
    depth=5,
    top_k=10,
    vocab_size=32000,
    hidden_size=4096,
)

# Generate the frontier using the Triton backend
frontier = frontier_generate(
    cfg,
    features_concat=features,
    backend="triton",
    ea_layer=model,
)
```

### Low-Level API

```python
from eagle.model.triton_drafttoken_gen.drafter import DrafterConfig, Weights, Buffers, launch_drafter

# Create configuration
cfg = DrafterConfig(
    H_ea=4096,
    V=32000,
    n_head=32,
    head_dim=128,
    K=10,
    TOPK=10,
    DEPTH=5,
    T_max=60,
)

# Initialize weights and buffers
weights = Weights(...)
bufs = Buffers(...)

# Launch the drafter
draft_tokens, retrieve_indices, tree_mask, tree_position_ids = launch_drafter(
    cfg, {"X_concat": features}, weights, bufs, fallback=False
)
```

## Testing

Run the tests to verify the implementation:

```bash
# Run unit tests
python -m eagle.model.triton_drafttoken_gen.test_drafter

# Run integration tests
python -m eagle.model.triton_drafttoken_gen.integration_test_drafter

# Run benchmarks
python -m eagle.model.triton_drafttoken_gen.benchmark
```

## Performance

The Triton drafter shows significant performance improvements over the original implementation:

| Model Size | Original (ms) | Triton (ms) | Speedup |
|------------|---------------|-------------|---------|
| 8B         | 120           | 45          | 2.67x   |
| 70B        | 350           | 130         | 2.69x   |

*Note: Actual performance depends on hardware configuration and model parameters.*

## Structure

- `drafter.py`: Core implementation of the Triton drafter
- `ea_integration.py`: Integration with EA model
- `frontier_integration.py`: Integration with frontier API
- `test_drafter.py`: Unit tests
- `integration_test_drafter.py`: Integration tests
- `benchmark.py`: Performance benchmarks

## License

Same as the EAGLE project.