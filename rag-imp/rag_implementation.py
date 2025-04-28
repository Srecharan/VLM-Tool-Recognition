import os
import pandas as pd
import numpy as np
import json
import torch
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
import importlib

# Function to check if unsloth is available
def is_unsloth_available():
    try:
        importlib.import_module('unsloth')
        return True
    except ImportError:
        return False

# Function to load model based on available libraries
def load_model(model_path, use_4bit=True):
    """
    Load the model using either unsloth or transformers based on what's available
    """
    if is_unsloth_available():
        # Use unsloth if available
        from unsloth import FastVisionModel
        print("Using Unsloth for model loading")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=use_4bit,
            use_gradient_checkpointing="unsloth",
        )
        FastVisionModel.for_inference(model)
    else:
        # Fallback to regular transformers
        print("Unsloth not available, using standard transformers")
        from transformers import AutoProcessor, AutoModelForCausalLM
        tokenizer = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if not use_4bit else torch.float32,
            device_map="auto",
        )
        
    return model, tokenizer

def create_embeddings(knowledge_df, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Create embeddings for the tool knowledge base
    """
    print("Creating embeddings for knowledge base...")
    
    # Load a lightweight sentence transformer model suitable for 8GB VRAM
    model = SentenceTransformer(model_name)
    
    # Create text to embed by combining relevant fields
    texts = []
    for _, row in knowledge_df.iterrows():
        # Convert safety_considerations to string if it's a dict
        if isinstance(row['safety_considerations'], dict):
            safety_info = json.dumps(row['safety_considerations'])
        else:
            safety_info = row['safety_considerations']
        
        text = f"Tool: {row['tool_name']}. "
        text += f"Function: {row['primary_function']}. "
        text += f"Instructions: {row['usage_instructions']}. "
        text += f"Safety: {safety_info}"
        
        texts.append(text)
    
    # Generate embeddings in batches to manage memory
    batch_size = 8  # Small batch size for 8GB VRAM
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    print(f"Created {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings, model

def build_faiss_index(embeddings):
    """
    Build a FAISS index for fast similarity search
    """
    print("Building FAISS index...")
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index

def retrieve_tool_information(query_text, embedding_model, index, knowledge_df, top_k=3):
    """
    Retrieve relevant tool information based on the query
    """
    # Encode the query
    query_embedding = embedding_model.encode([query_text])
    query_embedding = query_embedding.astype(np.float32)
    faiss.normalize_L2(query_embedding)
    
    # Search the index
    scores, indices = index.search(query_embedding, top_k)
    
    # Get the relevant tool information
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(knowledge_df):
            entry = knowledge_df.iloc[idx].to_dict()
            entry['score'] = float(scores[0][i])
            results.append(entry)
    
    return results

def run_rag_inference(image_path, model_path, knowledge_df, embeddings=None, embedding_model=None, index=None):
    """
    Run RAG-enhanced inference on a single image
    """
    # Create embeddings and index if not provided
    if embeddings is None or embedding_model is None or index is None:
        embeddings, embedding_model = create_embeddings(knowledge_df)
        index = build_faiss_index(embeddings)
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_model(model_path)
    
    # 1. First step: Basic tool identification
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    basic_instruction = "Identify the mechanical tools visible in this image."
    
    # Prepare input based on available library
    if is_unsloth_available():
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
    else:
        inputs = tokenizer(
            text=basic_instruction,
            images=image,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initial tool identification
    print("Generating initial tool identification...")
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
    )
    
    if is_unsloth_available():
        tool_identification = tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        tool_identification = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"Identified tools: {tool_identification}")
    
    # 2. Retrieve relevant tool information
    print("Retrieving information from knowledge base...")
    retrieved_info = retrieve_tool_information(
        f"Tool information for: {tool_identification}", 
        embedding_model, 
        index, 
        knowledge_df
    )
    
    # 3. Create enhanced prompt with retrieved information
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
    
    # 4. Final inference with RAG context
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
    
    # Prepare RAG-enhanced input
    if is_unsloth_available():
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
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        inputs = tokenizer(
            text=enhanced_instruction,
            images=image,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Generating enhanced response with RAG...")
    enhanced_output = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
    )
    
    enhanced_response = tokenizer.decode(enhanced_output[0], skip_special_tokens=True)
    
    # 5. Run standard inference (without RAG) for comparison
    print("Generating standard response (without RAG) for comparison...")
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
    
    # Prepare standard input
    if is_unsloth_available():
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
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        inputs = tokenizer(
            text=standard_instruction,
            images=image,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    standard_output = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
    )
    
    standard_response = tokenizer.decode(standard_output[0], skip_special_tokens=True)
    
    # Clean up GPU memory
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "standard_response": standard_response,
        "enhanced_response": enhanced_response,
        "retrieved_info": retrieved_info
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RAG inference on a single image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to your fine-tuned model')
    parser.add_argument('--knowledge_base', type=str, default='tool_knowledge_base.csv', help='Path to knowledge base CSV')
    parser.add_argument('--image_path', type=str, required=True, help='Path to test image')
    parser.add_argument('--output_dir', type=str, default='rag_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load knowledge base
    knowledge_df = pd.read_csv(args.knowledge_base)
    
    # Run RAG inference
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = run_rag_inference(args.image_path, args.model_path, knowledge_df)
    
    # Save results
    with open(os.path.join(args.output_dir, 'single_image_results.json'), 'w') as f:
        # Convert retrieved_info to a serializable format
        serializable_results = {
            "standard_response": results["standard_response"],
            "enhanced_response": results["enhanced_response"],
            "retrieved_info": json.dumps(results["retrieved_info"])
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {os.path.join(args.output_dir, 'single_image_results.json')}")