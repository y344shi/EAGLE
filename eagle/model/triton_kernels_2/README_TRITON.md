# EAGLE Triton Kernels

This directory contains Triton kernel implementations for the EAGLE model. These kernels are designed to accelerate the key operations in the EAGLE model, particularly for tree-based speculative decoding.

## Overview

The EAGLE model uses tree-based speculative decoding to accelerate text generation. The key operations that have been accelerated with Triton kernels are:

1. **Tree Attention**: Efficient attention computation with tree mask support
2. **TopK Expansion**: Fast top-k selection for tree expansion
3. **Posterior Evaluation**: Efficient evaluation of candidate tokens
4. **KV Block Copy**: Fast copying of key-value cache blocks
5. **Mask Preparation**: Efficient preparation of attention masks with tree support

## Kernel Files

- `tree_attention.py`: Tree-based attention implementation
- `topk_expand.py`: Top-k expansion for tree generation
- `posterior_eval.py`: Posterior probability evaluation
- `kv_block_copy.py`: Key-value cache block copying
- `mask_preparation.py`: Attention mask preparation with tree support
- `integration.py`: Integration of all kernels
- `ea_model_patch.py`: Patching utilities for the EAGLE model

## Usage

To use the Triton kernels with an EAGLE model:

```python
from eagle.model.ea_model import EaModel
from eagle.model.triton_kernels.ea_model_patch import patch_eagle_model

# Load model
model = EaModel.from_pretrained(
    base_model_path="path/to/model",
    ea_model_path="path/to/model",
    device_map="auto",
)

# Apply Triton patches
model = patch_eagle_model(model)

# Use the model as usual
# ...

# Optionally, remove patches when done
from eagle.model.triton_kernels.ea_model_patch import unpatch_eagle_model
unpatch_eagle_model(model)
```

## Testing

To test the Triton kernels:

```bash
python -m eagle.model.triton_kernels.test_triton_kernels --model_path path/to/model --compare
```

This will compare the performance of the PyTorch and Triton implementations.

## Implementation Details

### Tree Attention

The tree attention kernel implements the core attention mechanism used in the EAGLE model. It supports tree-based attention masks, which are essential for speculative decoding.

### TopK Expansion

The top-k expansion kernel efficiently selects the top-k candidates for tree expansion. This is a critical operation in the tree-based speculative decoding process.

### Posterior Evaluation

The posterior evaluation kernel efficiently evaluates candidate tokens against the base model's predictions. This is used to determine which speculative paths to accept.

### KV Block Copy

The KV block copy kernel efficiently copies blocks of the key-value cache. This is used to update the inference inputs after accepting speculative paths.

### Mask Preparation

The mask preparation kernel efficiently prepares attention masks with tree support. This is used to ensure that tokens can only attend to their ancestors in the tree.

## Performance

The Triton kernels provide significant speedups over the PyTorch implementations, particularly for large batch sizes and sequence lengths. The exact speedup depends on the hardware and model configuration.