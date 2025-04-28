import os
import argparse
import pandas as pd
import json
import torch
from tqdm import tqdm
from PIL import Image
import sys

# Import functions from rag_implementation.py
sys.path.append('.')
from rag_implementation import create_embeddings, build_faiss_index, retrieve_tool_information
from unsloth import FastVisionModel

def parse_args():
    parser = argparse.ArgumentParser(description='Run RAG inference on multiple images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--knowledge_base', type=str, required=True, help='Path to the knowledge base CSV')
    parser.add_argument('--test_csv', type=str, required=True, help='CSV file with test image paths')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='./rag_batch_results', help='Output directory')
    parser.add_argument('--max_images', type=int, default=10, help='Maximum number of images to process')
    return parser.parse_args()

def run_batch_rag(args):
    """
    Run RAG inference on multiple images
    """
    # Load knowledge base
    print(f"Loading knowledge base from {args.knowledge_base}...")
    knowledge_df = pd.read_csv(args.knowledge_base)
    
    # Create embeddings and index
    embeddings, embedding_model = create_embeddings(knowledge_df)
    index = build_faiss_index(embeddings)
    
    # Load test image paths
    print(f"Loading test image paths from {args.test_csv}...")
    test_df = pd.read_csv(args.test_csv)
    
    # Get image filename column name (might be 'filename' or 'image_name')
    filename_col = 'filename' if 'filename' in test_df.columns else 'image_name'
    
    # Limit number of images if specified
    if args.max_images > 0 and len(test_df) > args.max_images:
        print(f"Limiting to {args.max_images} images")
        test_df = test_df.head(args.max_images)
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each image
    results = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing images"):
        try:
            image_name = row[filename_col]
            image_path = os.path.join(args.image_dir, image_name)
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            # Process image with and without RAG
            image = Image.open(image_path).convert('RGB')
            
            # 1. Standard inference (without RAG)
            standard_instruction = """
            Analyze this image and identify ALL mechanical tools present. Return ONLY a valid JSON object in the following format:
            {
                "detected_tools": ["tool_name_1", ...],
                "bounding_boxes": [
                    {"tool": "tool_name_1", "bbox": [x1, y1, x2, y2]},
                    ...
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
                        ...
                    ]
                }
            }
            
            Important: Return ONLY the JSON object.
            """
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": standard_instruction}
                ]}
            ]
            
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            inputs = tokenizer(
                [image],
                text=[input_text],
                return_tensors="pt",
            ).to("cuda")
            
            standard_output = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
            )
            
            standard_response = tokenizer.decode(standard_output[0], skip_special_tokens=True)
            
            # 2. First step of RAG: Basic tool identification
            basic_instruction = "Identify the mechanical tools visible in this image."
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": basic_instruction}
                ]}
            ]
            
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            inputs = tokenizer(
                [image],
                text=[input_text],
                return_tensors="pt",
            ).to("cuda")
            
            # Initial tool identification
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
            )
            
            tool_identification = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # 3. Retrieve relevant tool information
            retrieved_info = retrieve_tool_information(
                f"Tool information for: {tool_identification}", 
                embedding_model, 
                index, 
                knowledge_df
            )
            
            # 4. Create enhanced prompt with retrieved information
            retrieved_context = "Reference information about these tools:\n"
            for info in retrieved_info:
                retrieved_context += f"Tool: {info['tool_name']}\n"
                retrieved_context += f"Function: {info['primary_function']}\n"
                
                safety_info = info['safety_considerations']
                if isinstance(safety_info, str):
                    try:
                        safety_info = json.loads(safety_info)
                    except:
                        safety_info = {"info": safety_info}
                
                retrieved_context += f"Safety: {json.dumps(safety_info)}\n\n"
            
            # 5. Final inference with RAG context
            enhanced_instruction = f"""
            {retrieved_context}
            
            Analyze this image and identify ALL mechanical tools present. Return ONLY a valid JSON object in the following format:
            {{
                "detected_tools": ["tool_name_1", ...],
                "bounding_boxes": [
                    {{"tool": "tool_name_1", "bbox": [x1, y1, x2, y2]}},
                    ...
                ],
                "detailed_analysis": {{
                    "tools_information": [
                        {{
                            "tool": "tool_name_1",
                            "primary_function": "brief description",
                            "safety_considerations": {{
                                "required_ppe": "safety equipment",
                                "primary_hazards": ["hazard1", "hazard2"],
                                "common_misuses": ["misuse1", "misuse2"]
                            }}
                        }},
                        ...
                    ]
                }}
            }}
            
            Important: Return ONLY the JSON object.
            """
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": enhanced_instruction}
                ]}
            ]
            
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            inputs = tokenizer(
                [image],
                text=[input_text],
                return_tensors="pt",
            ).to("cuda")
            
            enhanced_output = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
            )
            
            enhanced_response = tokenizer.decode(enhanced_output[0], skip_special_tokens=True)
            
            # Store results
            results.append({
                'image_name': image_name,
                'standard_response': standard_response,
                'enhanced_response': enhanced_response,
                'retrieved_info': json.dumps(retrieved_info)
            })
            
            # Save intermediate results
            if idx % 5 == 0:
                pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "intermediate_results.csv"), index=False)
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, "batch_results.csv"), index=False)
    
    # Clean up
    del model, tokenizer, embedding_model
    torch.cuda.empty_cache()
    
    print(f"Batch processing complete! Results saved to {os.path.join(args.output_dir, 'batch_results.csv')}")

def main():
    args = parse_args()
    run_batch_rag(args)

if __name__ == "__main__":
    main()