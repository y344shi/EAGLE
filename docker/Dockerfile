# EAGLE Multi-GPU Inference Docker Container
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set build arguments
ARG CUDA_VERSION=11.8
ARG PYTHON_VERSION=3.10

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    htop \
    vim \
    tmux \
    psutil \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and related packages
RUN pip3 install \
    transformers>=4.36.0 \
    accelerate \
    datasets \
    tokenizers \
    safetensors \
    huggingface-hub \
    sentencepiece \
    protobuf

# Install additional dependencies
RUN pip3 install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    jupyter \
    ipython \
    tqdm \
    psutil \
    gpustat \
    pynvml

# Install Triton for optimized kernels (optional)
RUN pip3 install triton

# Set working directory
WORKDIR /workspace

# Copy requirements if they exist
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt; fi

# Copy the EAGLE source code
COPY eagle/ ./eagle/
COPY setup.py ./
COPY README.md ./

# Install EAGLE in development mode
RUN pip3 install -e .

# Create directories for results and cache
RUN mkdir -p /workspace/results /root/.cache/huggingface

# Set up environment for HuggingFace
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# Create a startup script
RUN echo '#!/bin/bash\n\
echo "=== EAGLE Multi-GPU Inference Container ==="\n\
echo "CUDA Version: $(nvcc --version | grep release)"\n\
echo "Python Version: $(python --version)"\n\
echo "PyTorch Version: $(python -c \"import torch; print(torch.__version__)\")"\n\
echo "CUDA Available: $(python -c \"import torch; print(torch.cuda.is_available())\")"\n\
echo "GPU Count: $(python -c \"import torch; print(torch.cuda.device_count())\")"\n\
if [ $# -eq 0 ]; then\n\
    echo ""\n\
    echo "Available commands:"\n\
    echo "  bash eagle/evaluation/run_multi_gpu_comparison.sh --help"\n\
    echo "  python -m eagle.evaluation.compare_multi_gpu --help"\n\
    echo "  nvidia-smi"\n\
    echo "  htop"\n\
    echo ""\n\
    exec bash\n\
else\n\
    exec "$@"\n\
fi' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["bash"]