#!/bin/bash

# Set the base path for the script
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${BASE_DIR}/llama.py"

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Define an array of models to process
# Each entry is in the format "name:path"
MODELS=(
    "llama: unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
    "llama-vl:/home/exouser/akameswa/VLM-Tool-Recognition/training/models/llama-vl"
    "llama-v:/home/exouser/akameswa/VLM-Tool-Recognition/training/models/llama-v"
    "llama-l:/home/exouser/akameswa/VLM-Tool-Recognition/training/models/llama-l"
    # Add more models as needed in the format "name:path"
)

# Create a string of model specifications for the Python script
MODEL_ARGS=""
for model in "${MODELS[@]}"; do
    MODEL_ARGS="$MODEL_ARGS --models $model"
done

# Run the Python script with the models
echo "Starting evaluation with models: ${MODELS[@]}"
python "$PYTHON_SCRIPT" $MODEL_ARGS

echo "All evaluations completed"