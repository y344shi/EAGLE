#!/bin/bash

# Default values
BASE_MODEL_PATH=""
EA_MODEL_PATH=""
USE_EAGLE3=false
BASE_DEVICE="cuda:0"
DRAFT_DEVICE="cuda:1"
MAX_NEW_TOKENS=256
NUM_RUNS=5
PROMPT="Write a short story about a robot who learns to feel emotions."
IS_LLAMA3=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --base-model)
      BASE_MODEL_PATH="$2"
      shift 2
      ;;
    --ea-model)
      EA_MODEL_PATH="$2"
      shift 2
      ;;
    --use-eagle3)
      USE_EAGLE3=true
      shift
      ;;
    --base-device)
      BASE_DEVICE="$2"
      shift 2
      ;;
    --draft-device)
      DRAFT_DEVICE="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --num-runs)
      NUM_RUNS="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --is-llama3)
      IS_LLAMA3=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$BASE_MODEL_PATH" ] || [ -z "$EA_MODEL_PATH" ]; then
  echo "Error: --base-model and --ea-model are required"
  echo "Usage: $0 --base-model <path> --ea-model <path> [options]"
  exit 1
fi

# Prepare command
CMD="python3 -m eagle.evaluation.compare_multi_gpu"
CMD="$CMD --base-model $BASE_MODEL_PATH"
CMD="$CMD --ea-model $EA_MODEL_PATH"
CMD="$CMD --base-device $BASE_DEVICE"
CMD="$CMD --draft-device $DRAFT_DEVICE"
CMD="$CMD --max-new-tokens $MAX_NEW_TOKENS"
CMD="$CMD --num-runs $NUM_RUNS"
CMD="$CMD --prompt \"$PROMPT\""

# Add optional flags
if [ "$USE_EAGLE3" = true ]; then
  CMD="$CMD --use-eagle3"
fi

if [ "$IS_LLAMA3" = true ]; then
  CMD="$CMD --is-llama3"
fi
if [ "$USE_TENSOR_PARALLEL" = true ]; then
  CMD="$CMD --use-tensor-parallel"
fi
# Print the command
echo "Running: $CMD"

# Execute the command
eval $CMD
