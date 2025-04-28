#!/usr/bin/env python3
"""
Setup script for GRPO implementation:
1. Checks and installs required packages
2. Creates the directory structure
3. Sets up environment
"""

import os
import sys
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Setup GRPO implementation')
    parser.add_argument('--grpo_dir', type=str, 
                        default='grpo',
                        help='Directory to set up GRPO implementation')
    parser.add_argument('--copy_rag', action='store_true',
                        help='Copy necessary RAG implementation files')
    parser.add_argument('--rag_dir', type=str, 
                        default='rag-imp',
                        help='Directory containing RAG implementation')
    return parser.parse_args()

def install_requirements():
    """Install required packages"""
    print("Checking and installing required packages...")
    
    # List of required packages
    requirements = [
        "torch",
        "transformers>=4.36.0",
        "datasets",
        "unsloth>=0.4.0",
        "sentencepiece",
        "sentence-transformers",
        "faiss-cpu",  # or faiss-gpu if available
        "matplotlib",
        "seaborn",
        "pandas",
        "numpy",
        "tqdm",
        "pillow"
    ]
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Error: pip is not available. Please install pip first.")
        return False
    
    # Install requirements
    try:
        print("Installing required packages...")
        cmd = [sys.executable, "-m", "pip", "install"] + requirements
        subprocess.run(cmd, check=True)
        print("All required packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def create_directory_structure(args):
    """Create the directory structure for GRPO implementation"""
    print(f"Creating directory structure in {args.grpo_dir}...")
    
    # Create main directory
    os.makedirs(args.grpo_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "output",
        "output/paired_data",
        "output/grpo_model",
        "output/evaluation",
        "output/logs"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(args.grpo_dir, subdir), exist_ok=True)
    
    print("Directory structure created successfully!")
    return True

def copy_rag_files(args):
    """Copy necessary files from RAG implementation"""
    if not args.copy_rag:
        return True
    
    print(f"Copying necessary files from RAG implementation ({args.rag_dir})...")
    
    # Check if RAG directory exists
    if not os.path.exists(args.rag_dir):
        print(f"Error: RAG directory {args.rag_dir} not found.")
        return False
    
    # List of files to copy
    files_to_copy = [
        "rag_implementation.py",
        "tool_knowledge_base.csv"
    ]
    
    # Copy files
    for file in files_to_copy:
        src_path = os.path.join(args.rag_dir, file)
        dst_path = os.path.join(args.grpo_dir, file)
        
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found, skipping.")
            continue
        
        try:
            # Read the source file
            with open(src_path, 'r') as src_file:
                content = src_file.read()
            
            # Write to destination
            with open(dst_path, 'w') as dst_file:
                dst_file.write(content)
                
            print(f"Copied {file}")
        except Exception as e:
            print(f"Error copying {file}: {e}")
            return False
    
    print("RAG files copied successfully!")
    return True

def create_readme(args):
    """Create a README file with instructions"""
    readme_path = os.path.join(args.grpo_dir, "README.md")
    
    readme_content = """# GRPO Implementation for Tool Safety VLM

This directory contains the implementation of Generative Reinforcement from Pairwise Optimization (GRPO) 
for improving safety information generation in Vision-Language Models for mechanical tool recognition.

## Directory Structure

- `generate_grpo_pairs.py`: Script to generate paired data for GRPO training
- `train_grpo.py`: Script to train the model with GRPO
- `evaluate_grpo.py`: Script to evaluate and compare models
- `run_grpo_workflow.py`: Complete workflow script
- `rag_implementation.py`: RAG implementation used for comparison
- `tool_knowledge_base.csv`: Knowledge base for RAG

## Usage

1. Generate paired data:
```
python generate_grpo_pairs.py --model_path akameswa/Qwen2.5-VL-7B-Instruct-bnb-4bit-finetune-vision-language --knowledge_base tool_knowledge_base.csv --output_dir output/paired_data
```

2. Train with GRPO:
```
python train_grpo.py --model_path akameswa/Qwen2.5-VL-7B-Instruct-bnb-4bit-finetune-vision-language --input_data output/paired_data/grpo_pairs.pt --output_dir output/grpo_model
```

3. Evaluate models:
```
python evaluate_grpo.py --original_model akameswa/Qwen2.5-VL-7B-Instruct-bnb-4bit-finetune-vision-language --grpo_model output/grpo_model --knowledge_base tool_knowledge_base.csv --output_dir output/evaluation
```

4. Or run the complete workflow:
```
python run_grpo_workflow.py --model_path akameswa/Qwen2.5-VL-7B-Instruct-bnb-4bit-finetune-vision-language --knowledge_base tool_knowledge_base.csv --output_dir output
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Unsloth
- FAISS
- Sentence-Transformers
- Datasets
- Matplotlib, Seaborn (for visualization)

## Implementation Details

This implementation uses GRPO to teach VLMs to prioritize safety information when describing mechanical tools. 
The approach creates pairs of responses (RAG-enhanced vs. standard) and trains the model to prefer the 
RAG-enhanced ones containing complete safety information.
"""
    
    try:
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"README created at {readme_path}")
        return True
    except Exception as e:
        print(f"Error creating README: {e}")
        return False

def main():
    args = parse_args()
    
    print("Setting up GRPO implementation...")
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install them manually.")
        return
    
    # Create directory structure
    if not create_directory_structure(args):
        print("Failed to create directory structure.")
        return
    
    # Copy RAG files if requested
    if args.copy_rag:
        if not copy_rag_files(args):
            print("Failed to copy RAG files.")
            return
    
    # Create README
    if not create_readme(args):
        print("Failed to create README.")
        return
    
    print("\nSetup completed successfully!")
    print(f"GRPO implementation is ready in {args.grpo_dir}")
    print("\nNext steps:")
    print("1. Copy the Python scripts (generate_grpo_pairs.py, train_grpo.py, evaluate_grpo.py, run_grpo_workflow.py) to this directory")
    print("2. Run the workflow: python run_grpo_workflow.py")

if __name__ == "__main__":
    main()