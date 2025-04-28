#!/usr/bin/env python3
"""
Complete workflow script for GRPO implementation:
1. Generate paired data
2. Train with GRPO
3. Evaluate the models
"""

import os
import argparse
import subprocess
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Run the complete GRPO workflow')
    parser.add_argument('--model_path', type=str, 
                        default='akameswa/Qwen2.5-VL-7B-Instruct-bnb-4bit-finetune-vision-language',
                        help='Path to the fine-tuned model')
    parser.add_argument('--knowledge_base', type=str, 
                        default='tool_knowledge_base.csv',
                        help='Path to the knowledge base CSV')
    parser.add_argument('--dataset', type=str, 
                        default='akameswa/tool-safety-dataset',
                        help='HuggingFace dataset name')
    parser.add_argument('--split', type=str, 
                        default='valid',
                        help='Dataset split to use')
    parser.add_argument('--output_dir', type=str, 
                        default='grpo_outputs',
                        help='Base output directory')
    parser.add_argument('--num_samples', type=int, 
                        default=20,
                        help='Number of samples to process')
    parser.add_argument('--beta', type=float, 
                        default=0.1,
                        help='Beta parameter for GRPO')
    parser.add_argument('--num_epochs', type=int, 
                        default=1,
                        help='Number of training epochs')
    parser.add_argument('--skip_data_generation', action='store_true',
                        help='Skip data generation step (use existing data)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training step (use existing model)')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation step')
    return parser.parse_args()

def run_command(cmd, description):
    """Run a command with proper error handling"""
    print(f"\n{'='*20} {description} {'='*20}")
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        print(f"Command completed successfully in {elapsed_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def run_workflow(args):
    """Run the complete GRPO workflow"""
    # Create the base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Generate paired data for GRPO
    if not args.skip_data_generation:
        data_output_dir = os.path.join(args.output_dir, "paired_data")
        os.makedirs(data_output_dir, exist_ok=True)
        
        data_cmd = [
            sys.executable,
            "generate_grpo_pairs.py",
            "--model_path", args.model_path,
            "--knowledge_base", args.knowledge_base,
            "--dataset", args.dataset,
            "--split", args.split,
            "--num_samples", str(args.num_samples),
            "--output_dir", data_output_dir
        ]
        
        if not run_command(data_cmd, "Generating Paired Data for GRPO"):
            print("Data generation failed. Exiting workflow.")
            return False
        
        # Set the path to the generated data for the next steps
        paired_data_path = os.path.join(data_output_dir, "grpo_pairs.pt")
    else:
        print("\nSkipping data generation step as requested")
        # Assume the data already exists
        data_output_dir = os.path.join(args.output_dir, "paired_data")
        paired_data_path = os.path.join(data_output_dir, "grpo_pairs.pt")
        
        if not os.path.exists(paired_data_path):
            print(f"Error: Paired data not found at {paired_data_path}")
            print("Please run the data generation step first or provide the correct path")
            return False
    
    # Step 2: Train model with GRPO
    if not args.skip_training:
        model_output_dir = os.path.join(args.output_dir, "grpo_model")
        os.makedirs(model_output_dir, exist_ok=True)
        
        train_cmd = [
            sys.executable,
            "train_grpo.py",
            "--model_path", args.model_path,
            "--input_data", paired_data_path,
            "--output_dir", model_output_dir,
            "--beta", str(args.beta),
            "--num_train_epochs", str(args.num_epochs),
            "--max_samples", str(args.num_samples)
        ]
        
        if not run_command(train_cmd, "Training with GRPO"):
            print("GRPO training failed. Exiting workflow.")
            return False
        
        # Set the path to the trained model for evaluation
        grpo_model_path = model_output_dir
    else:
        print("\nSkipping training step as requested")
        # Assume the model already exists
        grpo_model_path = os.path.join(args.output_dir, "grpo_model")
        
        if not os.path.exists(grpo_model_path):
            print(f"Error: GRPO model not found at {grpo_model_path}")
            print("Please run the training step first or provide the correct path")
            return False
    
    # Step 3: Evaluate models
    if not args.skip_evaluation:
        eval_output_dir = os.path.join(args.output_dir, "evaluation")
        os.makedirs(eval_output_dir, exist_ok=True)
        
        eval_cmd = [
            sys.executable,
            "evaluate_grpo.py",
            "--original_model", args.model_path,
            "--grpo_model", grpo_model_path,
            "--knowledge_base", args.knowledge_base,
            "--dataset", args.dataset,
            "--split", args.split,
            "--num_samples", str(min(args.num_samples, 20)),  # Limit evaluation samples
            "--output_dir", eval_output_dir
        ]
        
        if not run_command(eval_cmd, "Evaluating Models"):
            print("Model evaluation failed.")
            return False
    else:
        print("\nSkipping evaluation step as requested")
    
    print("\n" + "="*60)
    print("GRPO workflow completed successfully!")
    print(f"All outputs saved to {args.output_dir}")
    return True

def main():
    args = parse_args()
    success = run_workflow(args)
    
    if success:
        print("\nGRPO implementation completed successfully!")
        
        # Print summary of what was done
        print("\nWorkflow Summary:")
        print(f"- Base model: {args.model_path}")
        print(f"- Dataset: {args.dataset} (split: {args.split})")
        print(f"- Knowledge base: {args.knowledge_base}")
        print(f"- Number of samples: {args.num_samples}")
        print(f"- GRPO beta parameter: {args.beta}")
        print(f"- Number of epochs: {args.num_epochs}")
        print(f"- All outputs saved to: {args.output_dir}")
        
        # Provide instructions for using the trained model
        print("\nTo use the GRPO-improved model:")
        print(f"1. Load it with: model, tokenizer = FastVisionModel.from_pretrained('{os.path.join(args.output_dir, 'grpo_model')}')")
        print(f"2. Check the evaluation results in: {os.path.join(args.output_dir, 'evaluation')}")
    else:
        print("\nGRPO workflow completed with errors. Please check the logs above.")

if __name__ == "__main__":
    main()