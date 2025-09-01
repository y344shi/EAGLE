import torch

class GPUAssertionError(RuntimeError):
    """Error raised when a GPU operation is attempted on a non-GPU device."""
    pass

def assert_cuda_available():
    """
    Assert that CUDA is available for GPU operations.
    
    Raises:
        GPUAssertionError: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise GPUAssertionError(
            "CUDA is not available. Triton kernels require a GPU to run. "
            "Use the PyTorch fallback implementation instead."
        )

def assert_tensor_on_cuda(tensor, name="Tensor"):
    """
    Assert that a tensor is on a CUDA device.
    
    Parameters:
        tensor: The tensor to check.
        name: Name of the tensor for error messages.
        
    Raises:
        GPUAssertionError: If the tensor is not on a CUDA device.
    """
    if not tensor.is_cuda:
        raise GPUAssertionError(
            f"{name} must be on a CUDA device. Triton kernels require GPU tensors. "
            "Use the PyTorch fallback implementation instead."
        )

def with_gpu_assertion(func):
    """
    Decorator to assert that CUDA is available before calling a function.
    
    Parameters:
        func: The function to decorate.
        
    Returns:
        The decorated function.
        
    Raises:
        GPUAssertionError: If CUDA is not available.
    """
    def wrapper(*args, **kwargs):
        assert_cuda_available()
        return func(*args, **kwargs)
    return wrapper