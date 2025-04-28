#!/bin/bash

# Run the complete GRPO workflow
# This script runs the entire GRPO implementation workflow:
# 1. Setup the environment
# 2. Generate paired data
# 3. Train with GRPO
# 4. Evaluate models

# Default parameters
MODEL_PATH="akameswa/Qwen2.5-VL-7B-Instruct-bnb-4bit-finetune-vision-language"
KNOWLEDGE_BASE="tool_knowledge_base.csv"
DATASET="akameswa/tool-safety-dataset"
SPLIT="valid"
OUTPUT_DIR="grpo_outputs"
NUM_SAMPLES=20
BETA=0.1
NUM_EPOCHS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model_path)
            MODEL_PATH="$2"
            shift
            shift
            ;;
        --knowledge_base)
            KNOWLEDGE_BASE="$2"
            shift
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift
            shift
            ;;
        --split)
            SPLIT="$2"
            shift
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift
            shift
            ;;
        --beta)
            BETA="$2"
            shift
            shift
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift
            shift
            ;;
        --skip_data_generation)
            SKIP_DATA_GENERATION="--skip_data_generation"
            shift
            ;;
        --skip_training)
            SKIP_TRAINING="--skip_training"
            shift
            ;;
        --skip_evaluation)
            SKIP_EVALUATION="--skip_evaluation"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# Print configuration
echo "Running GRPO workflow with the following configuration:"
echo "- Model path: $MODEL_PATH"
echo "- Knowledge base: $KNOWLEDGE_BASE"
echo "- Dataset: $DATASET (split: $SPLIT)"
echo "- Output directory: $OUTPUT_DIR"
echo "- Number of samples: $NUM_SAMPLES"
echo "- Beta parameter: $BETA"
echo "- Number of epochs: $NUM_EPOCHS"
echo "- Skip data generation: ${SKIP_DATA_GENERATION:-false}"
echo "- Skip training: ${SKIP_TRAINING:-false}"
echo "- Skip evaluation: ${SKIP_EVALUATION:-false}"
echo

# Confirm with user
read -p "Do you want to proceed? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Workflow aborted."
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/grpo_workflow_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

# Run the workflow
{
    echo "Starting GRPO workflow at $(date)"
    echo "==============================="
    
    # Run workflow script
    python run_grpo_workflow.py \
        --model_path "$MODEL_PATH" \
        --knowledge_base "$KNOWLEDGE_BASE" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --output_dir "$OUTPUT_DIR" \
        --num_samples "$NUM_SAMPLES" \
        --beta "$BETA" \
        --num_epochs "$NUM_EPOCHS" \
        $SKIP_DATA_GENERATION $SKIP_TRAINING $SKIP_EVALUATION
    
    # Check result
    if [ $? -eq 0 ]; then
        echo "Workflow completed successfully!"
    else
        echo "Workflow failed with error code $?"
    fi
    
    echo "==============================="
    echo "Workflow finished at $(date)"
} 2>&1 | tee -a "$LOG_FILE"

echo "Log file saved to: $LOG_FILE"