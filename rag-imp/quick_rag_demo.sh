#!/bin/bash

# Make sure we have a sample image first
if [ ! -f "test_images/sample_image_0.jpg" ]; then
    echo "Getting a sample image first..."
    python run_single_rag_hf.py --use_sample_image
fi

# Run the simplified RAG implementation
echo "Running simplified RAG implementation..."
python simple_run_rag.py

echo "RAG implementation completed!"
echo "Check the rag_results directory for outputs and comparison between standard and RAG-enhanced responses."