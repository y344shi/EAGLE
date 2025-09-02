#!/bin/bash

# EAGLE Docker Runner Script
# This script provides convenient commands to run EAGLE inference in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if nvidia-docker is available
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        print_warning "NVIDIA Docker runtime not available. GPU acceleration may not work."
    fi
}

# Build the Docker image
build_image() {
    print_info "Building EAGLE Docker image..."
    cd "$(dirname "$0")"
    docker-compose build eagle-inference
    print_success "Docker image built successfully!"
}

# Run the multi-GPU comparison
run_comparison() {
    local base_model=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
    local ea_model=${2:-"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"}
    local base_device=${3:-"cuda:0"}
    local draft_device=${4:-"cuda:1"}
    local max_tokens=${5:-"128"}
    local num_runs=${6:-"1"}
    
    print_info "Running EAGLE multi-GPU comparison..."
    print_info "Base model: $base_model"
    print_info "EA model: $ea_model"
    print_info "Base device: $base_device"
    print_info "Draft device: $draft_device"
    print_info "Max tokens: $max_tokens"
    print_info "Number of runs: $num_runs"
    
    cd "$(dirname "$0")"
    docker-compose run --rm eagle-inference bash -c "
        python -m eagle.evaluation.compare_multi_gpu \
            --base-model '$base_model' \
            --ea-model '$ea_model' \
            --base-device '$base_device' \
            --draft-device '$draft_device' \
            --max-new-tokens $max_tokens \
            --num-runs $num_runs \
            --prompt 'Write a short story about a robot who learns to feel emotions.' \
            --use-eagle3
    "
    
    print_success "Comparison completed!"
}

# Start interactive shell
start_shell() {
    print_info "Starting interactive shell in EAGLE container..."
    cd "$(dirname "$0")"
    docker-compose run --rm eagle-inference bash
}

# Start Jupyter notebook
start_jupyter() {
    print_info "Starting Jupyter Lab..."
    cd "$(dirname "$0")"
    docker-compose --profile jupyter up -d eagle-jupyter
    print_success "Jupyter Lab started at http://localhost:8888"
}

# Stop all services
stop_services() {
    print_info "Stopping all Docker services..."
    cd "$(dirname "$0")"
    docker-compose down
    print_success "All services stopped!"
}

# Clean up Docker resources
cleanup() {
    print_info "Cleaning up Docker resources..."
    cd "$(dirname "$0")"
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    print_success "Cleanup completed!"
}

# Show logs
show_logs() {
    cd "$(dirname "$0")"
    docker-compose logs -f eagle-inference
}

# Show usage information
show_usage() {
    echo "EAGLE Docker Runner"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build                           Build the Docker image"
    echo "  run [base_model] [ea_model]     Run multi-GPU comparison"
    echo "      [base_device] [draft_device] [max_tokens] [num_runs]"
    echo "  shell                           Start interactive shell"
    echo "  jupyter                         Start Jupyter Lab"
    echo "  logs                            Show container logs"
    echo "  stop                            Stop all services"
    echo "  cleanup                         Clean up Docker resources"
    echo "  help                            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run"
    echo "  $0 run meta-llama/Llama-3.1-8B-Instruct yuhuili/EAGLE3-LLaMA3.1-Instruct-8B cuda:0 cpu 256 3"
    echo "  $0 shell"
    echo "  $0 jupyter"
    echo ""
}

# Main script logic
main() {
    case "${1:-help}" in
        "build")
            check_dependencies
            build_image
            ;;
        "run")
            check_dependencies
            run_comparison "$2" "$3" "$4" "$5" "$6" "$7"
            ;;
        "shell")
            check_dependencies
            start_shell
            ;;
        "jupyter")
            check_dependencies
            start_jupyter
            ;;
        "logs")
            show_logs
            ;;
        "stop")
            stop_services
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# Run main function with all arguments
main "$@"