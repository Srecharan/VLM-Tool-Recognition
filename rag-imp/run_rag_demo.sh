#!/bin/bash

# First, make sure we have all the required packages
echo "Installing necessary packages..."
pip install datasets transformers requests huggingface_hub sentence-transformers faiss-cpu torch pillow tqdm

# Run the RAG demo with a sample image from the dataset
echo "Running RAG implementation with a sample image..."
python run_single_rag_hf.py --use_sample_image

echo "Done! Check the 'rag_results' directory for the output."