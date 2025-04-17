# Finetuning Pipeline for Tool Recognition VLM

This script implements a finetuning pipeline for Vision-Language Models (VLM) specifically
for tool recognition and safety instruction tasks. The pipeline uses the unsloth library
for efficient training and Parameter Efficient Fine-Tuning (PEFT) techniques.

## Source 
Original implementation adapted from:
[Colab Notebook](https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing)

## Key Features
* Uses 4-bit quantization for memory efficiency
* Implements LoRA (Low-Rank Adaptation) for parameter-efficient finetuning
* Supports selective finetuning of vision and language layers
* Includes comprehensive logging and training statistics
* Supports model pushing to HuggingFace Hub

## Finetuning Process

### 1. Model Initialization
* Loads Qwen2.5-VL-7B-Instruct model with 4-bit quantization
* Configures gradient checkpointing for handling long contexts

### 2. PEFT Setup
* Implements LoRA with rank=16 and alpha=16
* Allows selective finetuning of vision/language layers
* Configures attention and MLP module finetuning

### 3. Dataset Processing
* Converts tool recognition data into conversation format
* Includes tool locations, usage instructions, PPE requirements, and hazards
* Supports 17 different tool classes with detailed annotations

### 4. Training Configuration
* Uses SFTTrainer with mixed precision (bf16 where supported)
* Batch size: 1 with gradient accumulation steps of 4
* Learning rate: 2e-4 with linear scheduler
* Uses 8-bit AdamW optimizer
* Maximum sequence length: 1024

## Usage
```bash
python finetune.py
```

> **Note**: Configure the args dictionary in `__main__` for custom training settings.