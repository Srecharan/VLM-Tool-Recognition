#!/bin/bash

echo "LangChain RAG Demo for Tool Safety"
echo "=================================="

# Set default values
MODEL_PATH=${1:-"./training/final_models/vision_lora_model"}
IMAGE_PATH=${2:-"./rag-imp/test_images/sample_tools.jpg"}
OUTPUT_DIR=${3:-"./rag-imp/langchain_outputs"}

echo "Model Path: $MODEL_PATH"
echo "Image Path: $IMAGE_PATH"
echo "Output Directory: $OUTPUT_DIR"

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Warning: Image $IMAGE_PATH does not exist"
    echo "Using sample image from test_images directory..."
    
    IMAGE_PATH=$(find ./rag-imp/test_images/ -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | head -1)
    
    if [ -z "$IMAGE_PATH" ]; then
        echo "No test images found. Please provide a valid image path."
        exit 1
    fi
    
    echo "Using: $IMAGE_PATH"
fi

# Install requirements
echo "Installing LangChain requirements..."
pip install -r rag-imp/rag_requirements.txt

echo ""
echo "Starting LangChain RAG pipeline..."

# Run the simple demo (works without GPU)
python rag-imp/langchain_demo_simple.py

echo ""
echo "Demo completed!" 