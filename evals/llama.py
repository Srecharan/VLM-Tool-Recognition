from unsloth import FastVisionModel
import torch
from datasets import load_dataset
import pandas as pd
import json
import os
import argparse
from tqdm import tqdm

def process_llama_model(model_name, model_path):
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Define output file path
    output_path = os.path.join(results_dir, f'{model_name}.csv')
    
    dataset = load_dataset("akameswa/tool-safety-dataset", split="valid")
    
    # Instruction
    instruction = """
    Analyze this image and identify ALL mechanical tools present. For each tool, provide:

    **Tool Description**
    **Main [Tool Name]**: [Name]

    **Bounding Box**
    **Main [Tool Name]**: [[x1, y1, x2, y2]]

    **Main [Tool Name] (Detailed)**
    **Primary Function**: [Description]
    **Safety Considerations**
    **Required PPE**: [Safety Equipment]
    **Primary Hazards**: [List of Hazards]
    **Common Misuses**: [List of Common Misuses]

    Please maintain this exact format for proper parsing.
    """

    print(f"\nProcessing model: {model_name}")
    
    # Create/open CSV file with headers - simplified to just filename and response
    with open(output_path, 'w', encoding='utf-8') as f:
        pd.DataFrame(columns=[
            'filename',
            'response'
        ]).to_csv(f, index=False, escapechar='\\', quoting=1)
    
    model = None
    tokenizer = None
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        FastVisionModel.for_inference(model)
        
        # Process each image
        for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing {model_name}"):
            image_name = example.get('image_name', f'image_{idx}')
            try:
                image = example['image']
                
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction}
                    ]}
                ]
                
                if not tokenizer:
                    raise ValueError("Tokenizer not loaded")
                    
                input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                
                inputs = tokenizer(
                    [image],
                    text = [input_text],
                    return_tensors="pt",
                ).to("cuda")
                
                # Generation parameters
                output = model.generate(
                    **inputs,
                    max_new_tokens=700,
                    use_cache=True,
                    temperature=1.0,
                    do_sample=True,
                    top_p=0.9
                )
                
                if not tokenizer:
                    raise ValueError("Tokenizer not loaded")
                    
                response_text = tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Store the response as a list element for CSV integrity
                responses = [response_text]
                
                # Store only filename and response list
                row_dict = {
                    'filename': image_name,
                    'response': json.dumps(responses)  # Store as JSON array
                }
                
                # Write to CSV
                pd.DataFrame([row_dict]).to_csv(
                    output_path, 
                    mode='a', 
                    header=False, 
                    index=False,
                    escapechar='\\',
                    quoting=1
                )
                
            except Exception as e:
                error_msg = str(e).replace('\n', ' ').replace('"', "'")
                print(f"\nError processing image {image_name}: {error_msg}")
                
                # Record error as list
                pd.DataFrame([{
                    'filename': image_name,
                    'response': json.dumps([f"ERROR: {error_msg}"])
                }]).to_csv(
                    output_path, 
                    mode='a', 
                    header=False, 
                    index=False,
                    escapechar='\\',
                    quoting=1
                )
                
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('"', "'")
        print(f"\nFATAL Error during setup or processing for {model_name}: {error_msg}")
        pd.DataFrame([{
            'filename': 'FATAL_ERROR',
            'response': json.dumps([f"Model loading/setup failed: {error_msg}"])
        }]).to_csv(output_path, mode='a', header=False, index=False, escapechar='\\', quoting=1)

    finally:
        # Clean up GPU memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\nProcessing finished for {model_name}. Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process vision models for tool recognition")
    parser.add_argument("--models", nargs='+', required=True, 
                        help="Model specifications in the format name:path (e.g., llama-vl:/path/to/model)")
    
    args = parser.parse_args()
    
    # Parse the model specifications
    model_paths = {}
    for model_spec in args.models:
        try:
            name, path = model_spec.split(':', 1)
            model_paths[name] = path
        except ValueError:
            print(f"Invalid model specification: {model_spec}. Use format 'name:path'")
            continue
    
    if not model_paths:
        print("No valid model specifications provided. Exiting.")
        return
    
    print(f"Processing {len(model_paths)} models: {', '.join(model_paths.keys())}")
    
    # Process each model separately
    for model_name, model_path in model_paths.items():
        process_llama_model(model_name, model_path)

    print("\nAll models processed.")

if __name__ == "__main__":
    main()
