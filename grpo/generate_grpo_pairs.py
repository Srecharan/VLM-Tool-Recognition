import os
import argparse
import pandas as pd
import numpy as np
import json
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

# Import from your RAG implementation
from rag_implementation import retrieve_tool_information

# Import unsloth
from unsloth import FastVisionModel

def parse_args():
    parser = argparse.ArgumentParser(description='Generate paired data for GRPO training')
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
    parser.add_argument('--num_samples', type=int, 
                        default=50,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, 
                        default='grpo_data',
                        help='Output directory for paired data')
    return parser.parse_args()

def create_embeddings(knowledge_df):
    """Create embeddings for knowledge base entries"""
    print("Creating embeddings for knowledge base...")
    
    # Load a sentence transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Create text to embed by combining relevant fields
    texts = []
    for _, row in knowledge_df.iterrows():
        # Handle safety_considerations - might be string or dict
        if isinstance(row['safety_considerations'], dict):
            safety_info = json.dumps(row['safety_considerations'])
        else:
            safety_info = row['safety_considerations']
            
        # Create text representation
        text = f"Tool: {row['tool_name']}. "
        text += f"Function: {row['primary_function']}. "
        text += f"Instructions: {row.get('usage_instructions', '')}. "
        text += f"Safety: {safety_info}"
        
        texts.append(text)
    
    # Generate embeddings
    embeddings = model.encode(texts, batch_size=8, show_progress_bar=True)
    
    return embeddings, model

def build_faiss_index(embeddings):
    """Build a FAISS index for fast similarity search"""
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index

def generate_standard_response(image, model, tokenizer):
    """Generate standard response (without RAG)"""
    
    instruction = """
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
                        "required_ppe": "safety equipment needed",
                        "primary_hazards": ["hazard1", "hazard2"],
                        "common_misuses": ["misuse1", "misuse2"]
                    }
                },
                ...
            ]
        }
    }
    
    Important: Return ONLY the JSON object, no additional text or explanations.
    """
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    inputs = tokenizer(
        [image],
        text=[input_text],
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    output = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def generate_rag_response(image, model, tokenizer, embedding_model, faiss_index, knowledge_df):
    """Generate RAG-enhanced response"""
    
    # Step 1: Basic tool identification
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
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )
    
    tool_identification = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Step 2: Retrieve relevant information
    retrieved_info = retrieve_tool_information(
        f"Tools identified: {tool_identification}",
        embedding_model,
        faiss_index,
        knowledge_df
    )
    
    # Step 3: Create RAG context
    context = "Reference information about these tools:\n\n"
    for info in retrieved_info:
        context += f"TOOL: {info['tool_name']}\n"
        context += f"FUNCTION: {info['primary_function']}\n"
        
        safety = info['safety_considerations']
        if isinstance(safety, str):
            try:
                safety = json.loads(safety)
            except:
                pass
        
        if isinstance(safety, dict):
            context += "SAFETY INFORMATION:\n"
            for k, v in safety.items():
                context += f"- {k}: {v}\n"
        else:
            context += f"SAFETY: {safety}\n"
        
        context += "\n"
    
    # Step 4: Generate RAG-enhanced response
    rag_instruction = f"""
    {context}
    
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
                        "required_ppe": "safety equipment needed",
                        "primary_hazards": ["hazard1", "hazard2"],
                        "common_misuses": ["misuse1", "misuse2"]
                    }}
                }},
                ...
            ]
        }}
    }}
    
    Important: Return ONLY the JSON object, no additional text or explanations.
    """
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": rag_instruction}
        ]}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    inputs = tokenizer(
        [image],
        text=[input_text],
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )
    
    rag_response = tokenizer.decode(output[0], skip_special_tokens=True)
    return rag_response

def create_grpo_dataset(args):
    """Create dataset for GRPO training"""
    print(f"Creating GRPO training dataset...")
    
    # Load knowledge base
    print(f"Loading knowledge base from {args.knowledge_base}...")
    knowledge_df = pd.read_csv(args.knowledge_base)
    
    # Create embeddings and index
    embeddings, embedding_model = create_embeddings(knowledge_df)
    faiss_index = build_faiss_index(embeddings)
    
    # Load fine-tuned model
    print(f"Loading fine-tuned model from {args.model_path}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path,
        load_in_4bit=True,  # Use 4-bit quantization
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    
    # Load dataset
    print(f"Loading dataset {args.dataset} (split: {args.split})...")
    dataset = load_dataset(args.dataset, split=args.split)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Limit number of samples
    if args.num_samples > 0 and len(dataset) > args.num_samples:
        dataset = dataset.select(range(args.num_samples))
    
    # Process each image
    paired_data = []
    
    for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc="Generating pairs"):
        try:
            # Get image
            image = example['image']
            image_name = example.get('image_name', f'image_{idx}')
            
            # Generate standard response (non-RAG)
            standard_response = generate_standard_response(image, model, tokenizer)
            
            # Generate RAG-enhanced response
            rag_response = generate_rag_response(
                image, 
                model, 
                tokenizer, 
                embedding_model, 
                faiss_index, 
                knowledge_df
            )
            
            # Add to paired data
            paired_data.append({
                'image': image,
                'image_name': image_name,
                'prompt': "Analyze this image and identify all mechanical tools with safety information",
                'chosen': rag_response,  # RAG response is preferred
                'rejected': standard_response  # Standard response is rejected
            })
            
            # Save intermediate results
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} samples")
                # Save current pairs
                torch.save(paired_data, os.path.join(args.output_dir, f"grpo_pairs_intermediate.pt"))
                
                # Also save text samples for inspection
                with open(os.path.join(args.output_dir, f"sample_pairs.json"), 'w') as f:
                    # Save a sample of pairs (without images)
                    samples = []
                    for i, pair in enumerate(paired_data[:5]):
                        samples.append({
                            'image_name': pair['image_name'],
                            'prompt': pair['prompt'],
                            'chosen': pair['chosen'],
                            'rejected': pair['rejected']
                        })
                    json.dump(samples, f, indent=2)
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
    
    # Save final dataset
    print(f"Saving dataset with {len(paired_data)} pairs...")
    torch.save(paired_data, os.path.join(args.output_dir, "grpo_pairs.pt"))
    
    print(f"GRPO paired dataset created successfully in {args.output_dir}")
    return paired_data

def main():
    args = parse_args()
    paired_data = create_grpo_dataset(args)
    
    # Print summary
    print(f"\nSummary:")
    print(f"Created {len(paired_data)} paired examples for GRPO training")
    print(f"Data saved to {args.output_dir}")

if __name__ == "__main__":
    main()