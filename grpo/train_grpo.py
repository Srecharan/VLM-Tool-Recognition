# Import unsloth first
import unsloth
from unsloth import FastVisionModel

import os
import argparse
import torch
from tqdm import tqdm
from transformers import TrainingArguments
import json

# Import TRL's GRPO Trainer
from trl import GRPOTrainer, GRPOConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train model with GRPO')
    parser.add_argument('--model_path', type=str, 
                        default='akameswa/Qwen2.5-VL-7B-Instruct-bnb-4bit-finetune-vision-language',
                        help='Path to the fine-tuned model')
    parser.add_argument('--input_data', type=str, 
                        default='grpo_data/grpo_pairs.pt',
                        help='Path to the GRPO paired data')
    parser.add_argument('--output_dir', type=str, 
                        default='grpo_model',
                        help='Output directory for GRPO model')
    parser.add_argument('--beta', type=float, 
                        default=0.1,
                        help='Beta parameter for GRPO (controls preference strength)')
    parser.add_argument('--learning_rate', type=float, 
                        default=5e-5,
                        help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, 
                        default=1,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, 
                        default=1,
                        help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, 
                        default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--save_steps', type=int, 
                        default=20,
                        help='Save checkpoint every X steps')
    parser.add_argument('--max_samples', type=int, 
                        default=0,
                        help='Maximum number of samples to use (0 for all)')
    return parser.parse_args()

def prepare_training_args(args):
    """Prepare training arguments for GRPO"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
        push_to_hub=False,
    )
    
    return training_args

def process_dataset(paired_data, tokenizer, max_samples=0):
    """Process dataset for GRPO training"""
    # Limit the dataset if needed
    if max_samples > 0 and len(paired_data) > max_samples:
        paired_data = paired_data[:max_samples]
    
    # Process each item to create the format for GRPO training
    processed_data = []
    for item in tqdm(paired_data, desc="Processing dataset"):
        # Extract data
        image = item['image']
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Add to processed data
        processed_data.append({
            'image': image,
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })
    
    print(f"Processed {len(processed_data)} samples for GRPO training")
    return processed_data

def train_with_grpo(args):
    """Train model with GRPO"""
    print(f"Loading fine-tuned model from {args.model_path}...")
    # Load model and tokenizer
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    
    # Prepare model for training
    FastVisionModel.for_training(model)
    
    # Load paired data
    print(f"Loading paired data from {args.input_data}...")
    paired_data = torch.load(args.input_data)
    
    # Process dataset
    processed_data = process_dataset(paired_data, tokenizer, args.max_samples)
    
    # Prepare training arguments
    training_args = prepare_training_args(args)
    
    # Create GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=processed_data,
        beta=args.beta,
    )
    
    # Save model info
    with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
        info = {
            "base_model": args.model_path,
            "training_type": "GRPO",
            "beta": args.beta,
            "num_samples": len(processed_data),
            "num_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
        }
        json.dump(info, f, indent=2)
    
    # Train model
    print("Starting GRPO training...")
    train_result = trainer.train()
    
    # Save model
    print(f"Saving GRPO-improved model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metrics
    with open(os.path.join(args.output_dir, "train_results.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    
    print("GRPO training completed successfully")
    return model, tokenizer

def main():
    args = parse_args()
    model, tokenizer = train_with_grpo(args)
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"Base model: {args.model_path}")
    print(f"GRPO-improved model saved to: {args.output_dir}")
    print(f"Parameter beta: {args.beta}")
    print(f"Number of epochs: {args.num_train_epochs}")

if __name__ == "__main__":
    main()