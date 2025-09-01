# Multi-GPU Support for EAGLE Speculative Decoding

This document describes how to use EAGLE's speculative decoding with the draft model on a separate GPU.

## Overview

EAGLE3 uses a transformer model for speculative decoding, where:
1. The main model generates tokens
2. The draft model predicts multiple tokens ahead
3. The main model verifies these predictions

In the multi-GPU implementation, we place:
- The base model on the primary GPU (e.g., cuda:0)
- The draft model on a secondary GPU (e.g., cuda:1)

This separation can improve performance by:
- Utilizing more GPU memory across devices
- Enabling parallel computation between models
- Reducing memory pressure on the primary GPU

## Usage

### Loading a Multi-GPU Model

```python
from eagle.model.multi_gpu_inference import load_model_multi_gpu

model = load_model_multi_gpu(
    base_model_path="path/to/base/model",
    ea_model_path="path/to/eagle/model",
    use_eagle3=True,  # Set to False for EAGLE2
    use_multi_gpu=True,  # Enable multi-GPU mode
    base_device="cuda:0",  # Device for the base model
    draft_device="cuda:1"  # Device for the draft model
)
```

### Running the Comparison Script

We provide a script to compare the performance between single-GPU and multi-GPU implementations:

```bash
bash eagle/evaluation/run_multi_gpu_comparison.sh \
    --base-model path/to/base/model \
    --ea-model path/to/eagle/model \
    --use-eagle3 \
    --base-device cuda:0 \
    --draft-device cuda:1 \
    --max-new-tokens 256 \
    --num-runs 5
```

## Implementation Details

### Key Components

1. **Device Placement**: The base model and draft model are placed on separate devices.
2. **Cross-Device Communication**: Hidden states and tokens are transferred between devices as needed.
3. **Synchronization**: Operations are synchronized to ensure correct execution order.

### Potential Bottlenecks

1. **PCIe Bandwidth**: The main bottleneck is the transfer of hidden states between GPUs.
2. **Synchronization Points**: The draft model and main model operations need synchronization.

### Optimization Strategies

1. **Non-blocking Transfers**: Using `torch.cuda.Stream` for overlapping computation with communication.
2. **Minimizing Data Transfers**: Only transferring necessary data between devices.
3. **Caching**: Keeping frequently used weights on both devices to reduce transfers.

## Performance Considerations

For optimal performance:

1. Use GPUs connected via NVLink or PCIe 4.0+ for faster data transfer.
2. Balance the model sizes between GPUs based on their memory capacity.
3. Consider the trade-off between communication overhead and computation parallelism.

## Troubleshooting

If you encounter issues:

1. **Out of Memory**: Reduce batch size or model size on each GPU.
2. **Slow Performance**: Check if data transfer between GPUs is the bottleneck.
3. **Different Outputs**: Ensure the same random seeds are used for both implementations.