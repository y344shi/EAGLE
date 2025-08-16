# EAGLE Triton Kernels

This directory contains Triton kernel implementations for the EAGLE model to accelerate speculative decoding. These kernels replace the PyTorch implementations in the original codebase with optimized Triton versions.

## Key Components

1. **Tree Attention Kernel** (`tree_attention.py`): Implements efficient attention computation with tree mask support.
2. **TopK Expansion Kernel** (`topk_expand.py`): Accelerates the top-k selection for tree expansion.
3. **Posterior Evaluation Kernel** (`posterior_eval.py`): Optimizes the evaluation of candidate tokens.
4. **KV Block Copy Kernel** (`kv_block_copy.py`): Provides efficient copying of key-value cache blocks.
5. **Mask Preparation Kernel** (`mask_preparation.py`): Accelerates the preparation of attention masks with tree support.
6. **Integration Module** (`integration.py`): Brings all kernels together into a unified interface.
7. **Model Patching Utilities** (`ea_model_patch.py`): Provides functions to patch and unpatch the EAGLE model.

## Usage

To use these Triton kernels with an EAGLE model:

```python
from eagle.model.ea_model import EaModel
from eagle.model.triton_kernels_2.ea_model_patch import patch_eagle_model, unpatch_eagle_model

# Load model
model = EaModel.from_pretrained(...)

# Apply Triton patches
model = patch_eagle_model(model)

# Use the model as usual
output = model.eagenerate(...)

# Remove patches if needed
unpatch_eagle_model(model)
```

## Testing and Benchmarking

- `test_triton_kernels.py`: Test the Triton kernels with a given model and prompt.
- `benchmark.py`: Benchmark the Triton kernels against the original PyTorch implementation.
- `run_tests.py`: Run all tests for the Triton kernels.

## Compare Performance:

To benchmark the Triton kernels against the original PyTorch implementation, run:
```
python -m eagle.model.triton_kernels_2.benchmark --base_model_path meta-llama/Llama-3.1-8B-Instruct --ea_model_path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B --prompt "Hello" --max_new_tokens 2048 --runs 1 --warmup 1 --compare --print_eagle_timers --print_output
```

## Performance

The Triton kernels provide significant speedups over the original PyTorch implementation, especially for tree-based speculative decoding. The exact speedup depends on the model size, batch size, and hardware.