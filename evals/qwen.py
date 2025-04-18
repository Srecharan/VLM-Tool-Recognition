from unsloth import FastVisionModel
from datasets import load_dataset
import torch
import pandas as pd
import json
import os
from tqdm import tqdm

def extract_json_from_response(response_text):
    """Extract JSON from the model's response text"""
    try:
        # Find the JSON content between ```json and ``` markers
        start_marker = "```json"
        end_marker = "```"
        
        if start_marker in response_text:
            json_start = response_text.find(start_marker) + len(start_marker)
            json_end = response_text.find(end_marker, json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            json_str = response_text

        # Parse the JSON content
        response_json = json.loads(json_str)
        return response_json, None
    except Exception as e:
        error_msg = f"Error parsing JSON: {str(e)}"
        print(error_msg)
        return None, error_msg

def process_single_model(model_name, model_path):
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Define output file path
    output_path = os.path.join(results_dir, f'{model_name}.csv')
    
    dataset = load_dataset("akameswa/tool-safety-dataset", split="valid")
    
    instruction = """
    Analyze this image and identify ALL mechanical tools present. Return ONLY a valid JSON object in the following format:

    {
        "detected_tools": [
            "tool_name_1",
            "tool_name_2",
            ...
        ],
        "bounding_boxes": [
            {
                "tool": "tool_name_1",
                "bbox": [x1, y1, x2, y2]
            },
            {
                "tool": "tool_name_2",
                "bbox": [x1, y1, x2, y2]
            }
        ],
        "detailed_analysis": {
            "tools_information": [
                {
                    "tool": "tool_name_1",
                    "primary_function": "brief description",
                    "safety_considerations": {
                        "required_ppe": "safety equipment",
                        "primary_hazards": ["hazard1", "hazard2"],
                        "common_misuses": ["misuse1", "misuse2"]
                    }
                },
                {
                    // Similar structure for tool_name_2, etc.
                }
            ]
        }
    }

    Important: Return ONLY the JSON object, no additional text or explanations.
    """

    print(f"\nProcessing model: {model_name}")
    
    # Create/open CSV file with headers
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('filename,detected_tools,bounding_boxes,full_response\n')
    
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        FastVisionModel.for_inference(model)
        
        # Process each image
        for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing {model_name}"):
            try:
                image = example['image']
                image_name = example.get('image_name', '')  # Get image_name from dataset
                if not image_name:  # If image_name is not available, try to get filename
                    image_name = example.get('filename', f'image_{idx}')
                
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction}
                    ]}
                ]
                input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                inputs = tokenizer(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to("cuda")
                
                output = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    use_cache=True,
                    temperature=0.7,
                    min_p=0.1
                )
                
                response_text = tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract and process JSON response
                json_data, error_msg = extract_json_from_response(response_text)
                
                if json_data:
                    # Extract detected tools
                    detected_tools = json_data.get('detected_tools', [])
                    tools_str = '|'.join(detected_tools) if detected_tools else ''
                    
                    # Extract bounding boxes
                    bounding_boxes = json_data.get('bounding_boxes', [])
                    bbox_str = json.dumps(bounding_boxes) if bounding_boxes else '[]'
                    
                    # Extract detailed analysis
                    detailed_analysis = json_data.get('detailed_analysis', {})
                    detailed_analysis_str = json.dumps(detailed_analysis) if detailed_analysis else ''
                else:
                    tools_str = ''
                    bbox_str = '[]'
                    # Include the original response and error message in the full response
                    detailed_analysis_str = json.dumps(response_text)
                
                # Prepare row data
                row_dict = {
                    'filename': image_name,
                    'detected_tools': tools_str,
                    'bounding_boxes': bbox_str,
                    'full_response': detailed_analysis_str
                }
                
                # Convert to DataFrame and append to CSV
                pd.DataFrame([row_dict]).to_csv(
                    output_path, 
                    mode='a', 
                    header=False, 
                    index=False,
                    escapechar='\\',
                    quoting=1  # QUOTE_ALL to handle special characters
                )
                
            except Exception as e:
                error_msg = str(e)
                print(f"\nError processing image {image_name}: {error_msg}")
                # Write error information to CSV
                pd.DataFrame([{
                    'filename': image_name,
                    'detected_tools': '',
                    'bounding_boxes': '[]',
                    'full_response': f"Error: {error_msg}"
                }]).to_csv(
                    output_path, 
                    mode='a', 
                    header=False, 
                    index=False,
                    escapechar='\\',
                    quoting=1
                )
                
    except Exception as e:
        print(f"\nError loading model {model_name}: {str(e)}")
    
    finally:
        # Clear CUDA cache after processing each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\nResults for {model_name} saved to {output_path}")

# Model paths
model_paths = {
    'qwen-v': "/home/exouser/akameswa/VLM-Tool-Recognition/training/models/qwen-v"
}

# Process each model separately
for model_name, model_path in model_paths.items():
    process_single_model(model_name, model_path)