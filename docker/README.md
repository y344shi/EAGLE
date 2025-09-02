# Docker Setup for EAGLE Multi-GPU Inference

This guide provides instructions for running the EAGLE multi-GPU comparison system in a Docker container with CUDA support.

```bash
cd docker
./run.sh build
./run.sh run
./run.sh run meta-llama/Llama-3.1-8B-Instruct yuhuili/EAGLE3-LLaMA3.1-Instruct-8B cuda:0 cpu 256 3
```

## Prerequisites

1. **NVIDIA Docker Runtime**: Ensure you have NVIDIA Container Toolkit installed
2. **CUDA-compatible GPU**: At least one NVIDIA GPU with CUDA support
3. **Docker**: Docker Engine 19.03+ with BuildKit support
4. **Sufficient System Resources**: 
   - At least 16GB RAM (32GB recommended)
   - 50GB+ free disk space for models and container

## Quick Start

```bash
# Clone the repository
git clone <your-eagle-repo-url>
cd EAGLE

# Build the Docker image
docker build -f docker/Dockerfile -t eagle-inference .

# Run the container with GPU support
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  eagle-inference
```

## Detailed Setup

### 1. Build the Docker Image

```bash
# Build with default settings
docker build -f docker/Dockerfile -t eagle-inference .

# Build with specific CUDA version (optional)
docker build -f docker/Dockerfile \
  --build-arg CUDA_VERSION=11.8 \
  -t eagle-inference .
```

### 2. Run the Container

#### Basic Run (Interactive)
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  eagle-inference bash
```

#### Run with Specific GPU
```bash
# Use only GPU 0
docker run --gpus '"device=0"' -it --rm \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  eagle-inference

# Use multiple specific GPUs
docker run --gpus '"device=0,1"' -it --rm \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  eagle-inference
```

#### Run with Memory Limits
```bash
docker run --gpus all -it --rm \
  --memory=32g \
  --shm-size=8g \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  eagle-inference
```

### 3. Run the Multi-GPU Comparison

Once inside the container:

```bash
# Navigate to workspace
cd /workspace

# Run the comparison script
bash eagle/evaluation/run_multi_gpu_comparison.sh \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --ea-model yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --use-eagle3 \
  --base-device cuda:0 \
  --draft-device cpu \
  --max-new-tokens 128 \
  --num-runs 1
```

## Docker Compose Setup (Alternative)

For easier management, you can use Docker Compose:

```bash
# Start the service
docker-compose -f docker/docker-compose.yml up -d

# Execute commands in the running container
docker-compose -f docker/docker-compose.yml exec eagle-inference bash

# Stop the service
docker-compose -f docker/docker-compose.yml down
```

## Volume Mounts Explained

- `/workspace`: Maps your local EAGLE directory to the container
- `~/.cache/huggingface`: Caches downloaded models to avoid re-downloading
- `/tmp/eagle-results` (optional): For persistent result storage

## Environment Variables

You can customize the container behavior with environment variables:

```bash
docker run --gpus all -it --rm \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HOME=/root/.cache/huggingface \
  -v $(pwd):/workspace \
  eagle-inference
```

## Troubleshooting

### GPU Not Detected
```bash
# Check if NVIDIA runtime is available
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Out of Memory Issues
```bash
# Increase shared memory and system memory
docker run --gpus all -it --rm \
  --memory=64g \
  --shm-size=16g \
  -v $(pwd):/workspace \
  eagle-inference
```

### Permission Issues
```bash
# Run with current user ID
docker run --gpus all -it --rm \
  --user $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  eagle-inference
```

### Model Download Issues
```bash
# Pre-download models outside container
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
huggingface-cli download yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
```

## Performance Optimization

### For CPU Draft Device
```bash
# Increase CPU resources
docker run --gpus all -it --rm \
  --cpus="16" \
  --memory=32g \
  -v $(pwd):/workspace \
  eagle-inference
```

### For Multi-GPU Setup
```bash
# Use all available GPUs
docker run --gpus all -it --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  eagle-inference
```

## Monitoring Resources

Inside the container, you can monitor resource usage:

```bash
# GPU usage
nvidia-smi

# CPU and memory usage
htop

# Python process monitoring
ps aux | grep python
```

## Saving Results

Results are automatically saved to timestamped JSON files. To persist them:

```bash
# Create results directory
mkdir -p ./results

# Run with results volume
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/results:/workspace/results \
  eagle-inference

# Copy results after run
cp single_gpu_results_*.json ./results/
```

## Development Mode

For development and debugging:

```bash
# Mount source code for live editing
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/eagle:/workspace/eagle \
  --workdir /workspace \
  eagle-inference bash
```

## Security Considerations

- The container runs as root by default for GPU access
- Consider using `--user` flag for production deployments
- Limit network access if not needed: `--network none`
- Use read-only mounts where possible: `-v $(pwd):/workspace:ro`

## Support

If you encounter issues:

1. Check the container logs: `docker logs <container-id>`
2. Verify GPU access: `nvidia-smi` inside container
3. Check memory usage: `free -h` and `df -h`
4. Monitor process status: `ps aux | grep python`

For additional help, refer to the main EAGLE documentation or create an issue in the repository.