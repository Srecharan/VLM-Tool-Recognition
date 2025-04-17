import torch
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import logging
import os
from datetime import datetime

def setup_logging(args):
    """
    Setup logging configuration to write outputs to both file and console
    Args:
        args (dict): Configuration containing run name and output directory
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(args['output_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f"{args['run_name']}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    
    logging.info(f"Starting new training run: {args['run_name']}")
    logging.info(f"Logging to: {log_filename}")
    logging.info("Configuration:")
    for key, value in args.items():
        logging.info(f"{key}: {value}")

def load_model_and_tokenizer(model_name):
    """
    Load the model and tokenizer with specified configurations
    Args:
        model_name (str): Name of the pretrained model
    Returns:
        tuple: (model, tokenizer)
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,  # Use 4bit to reduce memory use
        use_gradient_checkpointing="unsloth",  # For long context
    )
    return model, tokenizer

def setup_peft_model(model, config):
    """
    Setup the PEFT (Parameter Efficient Fine-Tuning) model
    Args:
        model: Base model
        config (dict): Configuration for PEFT
    Returns:
        PEFT model
    """
    return FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config['finetune_vision'],
        finetune_language_layers=config['finetune_language'],
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
    )

def convert_to_conversation(sample):
    """
    Convert dataset sample to conversation format
    Args:
        sample: Dataset sample
    Returns:
        dict: Conversation format
    """
    # List of tool classes
    tool_classes = [
        "Adjustable spanner", "Backsaw", "Calipers", "Cutting pliers",
        "Drill", "Gas wrench", "Gun", "Hammer", "Hand", "Handsaw",
        "Needle-nose pliers", "Pliers", "Ratchet", "Screwdriver",
        "Tape measure", "Utility knife", "Wrench"
    ]
    
    # Identify present tools
    present_tools = [
        tool for tool in tool_classes
        if f"{tool}_bboxes" in sample and sample[f"{tool}_bboxes"]
    ]
    
    instruction = ("You are an expert tool safety specialist. Identify the tools "
                  "in this image, their locations (bounding boxes), and provide "
                  "their proper usage instructions, required PPE, and primary hazards.")
    
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": generate_caption(sample, present_tools)}
                ]
            }
        ]
    }

def generate_caption(sample, present_tools):
    """
    Generate detailed caption for tools in the image
    Args:
        sample: Dataset sample
        present_tools: List of tools present in the image
    Returns:
        str: Generated caption
    """
    caption = "In this image, I can identify the following tools:\n\n"
    
    for tool in present_tools:
        caption += f"## {tool}\n"
        
        fields = {
            "bboxes": "Location (Bounding Box)",
            "main_purpose": "Main Purpose",
            "usage_instructions": "Usage Instructions",
            "required_ppe": "Required PPE",
            "primary_hazards": "Primary Hazards",
            "common_misuses": "Common Misuses to Avoid"
        }
        
        for field, title in fields.items():
            key = f"{tool}_{field}"
            if key in sample and sample[key]:
                caption += f"**{title}**: {sample[key]}\n\n"
        
        caption += "\n"
    
    return caption

def main(args):
    """
    Main training function
    Args:
        args (dict): Training configuration
    """
    # Setup logging first
    setup_logging(args)
    
    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args['model_name'])
    
    # Setup PEFT model
    logging.info("Setting up PEFT model...")
    model = setup_peft_model(model, args)
    
    # Load and prepare dataset
    logging.info("Loading and preparing dataset...")
    dataset = load_dataset(args['dataset_name'], split="train")
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    
    # Enable model for training
    FastVisionModel.for_training(model)
    
    # Setup trainer
    logging.info("Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args['run_name'],
            report_to="wandb",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=1024,
        ),
    )
    
    # Train the model
    logging.info("Starting training...")
    trainer_stats = trainer.train()
    
    # Save the model
    logging.info(f"Saving model to {args['lora_model_path']}...")
    model.save_pretrained(args['lora_model_path'])
    tokenizer.save_pretrained(args['lora_model_path'])
    
    # Push to hub if token is provided
    if args['hub_token']:
        logging.info(f"Pushing model to hub: {args['hub_model_name']}")
        model.push_to_hub(args['hub_model_name'], token=args['hub_token'])
        tokenizer.push_to_hub(args['hub_model_name'], token=args['hub_token'])
    
    # Print and log training statistics
    log_training_stats(trainer_stats)

def log_training_stats(trainer_stats):
    """
    Log training statistics to both console and file
    Args:
        trainer_stats: Statistics from training
    """
    gpu_stats = torch.cuda.get_device_properties(0)
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    stats = [
        f"\nTraining Statistics:",
        f"GPU: {gpu_stats.name}",
        f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes",
        f"Peak memory usage: {used_memory} GB / {max_memory} GB",
        f"Memory utilization: {round(used_memory / max_memory * 100, 2)}%"
    ]
    
    for stat in stats:
        logging.info(stat)

if __name__ == "__main__":
    # Configuration
    args = {
        'model_name': "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        'dataset_name': "akameswa/tool-safety-dataset",
        'finetune_vision': False,      # Configure vision layer fine-tuning
        'finetune_language': True,    # Configure language layer fine-tuning
        'output_dir': "/home/exouser/akameswa/VLM-Tool-Recognition/finetuning/outputs/qwen-l",  # Directory for training outputs
        'lora_model_path': "/home/exouser/akameswa/VLM-Tool-Recognition/finetuning/models/qwen-l",  # Local save location
        'hub_model_name': "akameswa/Qwen2.5-VL-7B-Instruct-bnb-4bit-finetune-language",
        'hub_token': '',  # Add your HuggingFace token here if needed
        'run_name': 'qwen-l'
    }
    
    main(args)