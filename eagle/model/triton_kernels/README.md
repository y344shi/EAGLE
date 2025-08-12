# EAGLE Triton Kernels

This package provides Triton-optimized CUDA kernels for the EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) model. These kernels can significantly accelerate inference by optimizing the most computationally intensive operations.

## Overview

EAGLE is a speculative decoding method that speeds up Large Language Model (LLM) inference while maintaining performance. The Triton kernels in this package optimize the following key operations:

1. **Attention Computation**: Optimized multi-head attention mechanism
2. **KV Cache Management**: Efficient key-value cache operations for autoregressive generation
3. **Tree-based Token Generation**: Accelerated tree exploration for speculative decoding

## Requirements

- PyTorch >= 1.12.0
- Triton >= 2.0.0
- CUDA >= 11.6

## Installation

```bash
pip install triton
```

## Running Tests

Before using the Triton kernels in your project, it's recommended to run the tests to ensure they work correctly in your environment.

### Setting Up the Environment

Run the setup script to check if your environment is properly configured:

```bash
python setup_env.py
```

This will check if Triton and PyTorch are installed and provide information about your environment.

### Running the Tests

#### On Linux/WSL:

```bash
# Make the script executable (if needed)
chmod +x run_tests.sh

# Run the tests
./run_tests.sh

# For detailed performance report
./run_tests.sh --detailed-report
```

#### On Windows:

```powershell
# Run the tests using PowerShell
.\run_tests.ps1

# For detailed performance report
.\run_tests.ps1 --detailed-report
```

### Manual Test Execution

You can also run the tests manually:

```bash
# Set up the environment
python setup_env.py

# Run the tests
python run_tests.py

# For detailed performance report
python run_tests.py --detailed-report
```

## Usage

To optimize an EAGLE model with Triton kernels:

```python
from eagle.model.ea_model import EaModel
from eagle.model.triton_kernels import optimize_eagle_with_triton

# Load EAGLE model
model = EaModel.from_pretrained("EAGLE-3B")

# Optimize model with Triton kernels
model, original_methods = optimize_eagle_with_triton(model)

# Use the optimized model as usual
outputs = model.generate(input_ids, max_length=100)
```

See `usage_example.py` for a complete example.

## Benchmarking

To benchmark the performance of the Triton kernels:

```bash
python benchmark.py --model "EAGLE-3B" --prompt "Your test prompt here" --max-length 100 --num-runs 5
```

For integration testing:

```bash
python integration_test.py --model "EAGLE-3B" --prompt "Your test prompt here" --max-length 100 --num-runs 3 --verbose
```

## Performance

The Triton-optimized kernels can provide significant speedups over the standard PyTorch implementation:

- Attention computation: Up to 2x speedup
- KV cache operations: Up to 1.5x speedup
- Tree-based token generation: Up to 1.3x speedup

Overall, these optimizations can lead to a 20-30% reduction in inference time, on top of the speedups already provided by EAGLE's speculative decoding approach.

## Implementation Details

### Attention Kernel

The attention kernel (`attention.py`) implements an optimized version of the multi-head attention mechanism. It uses Triton's block-level parallelism to efficiently compute attention scores and weighted sums.

### KV Cache Kernel

The KV cache kernel (`kv_cache.py`) provides efficient operations for managing the key-value cache during autoregressive generation. It includes functions for appending new key-value pairs to the cache and retrieving cached values based on indices.

### Tree Decoding Kernel

The tree decoding kernel (`tree_decoding.py`) optimizes the tree-based token generation process used in EAGLE's speculative decoding. It includes functions for computing top-k tokens, generating tree masks, evaluating posterior probabilities, and updating input sequences.

### Integration

The integration module (`integration.py`) provides a high-level interface for using the Triton-optimized kernels with the EAGLE model. It includes a function (`optimize_eagle_with_triton`) that replaces the standard PyTorch operations with their Triton-optimized counterparts.

## Limitations

- The current implementation focuses on optimizing the most computationally intensive operations. Some less frequent operations still use the standard PyTorch implementation.
- The kernels are optimized for NVIDIA GPUs and may not provide the same level of performance on other hardware.
- For very small batch sizes or sequence lengths, the overhead of launching CUDA kernels may outweigh the benefits of optimization.

## Future Work

- Optimize additional operations, such as MLP layers and rotary position embeddings
- Support for mixed precision (FP16/BF16) computation
- Dynamic kernel selection based on input size and hardware capabilities
- Integration with other acceleration techniques, such as FlashAttention and quantization
- Design a fused speculative decoding mega kernel combining draft generation,
  verification, KV-cache updates and input sequence updates
