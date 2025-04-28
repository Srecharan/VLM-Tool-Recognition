import os
import argparse
import pandas as pd
import numpy as np
import json
import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# Import unsloth
from unsloth import FastVisionModel

# Import from RAG implementation
from rag_implementation import retrieve_tool_information, create_embeddings, build_faiss_index

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GRPO model')
    parser.add_argument('--original_model', type=str, 
                        default='akameswa/Qwen2.5-VL-7B-Instruct-bnb-4bit-finetune-vision-language',
                        help='Path to the original fine-tuned model')
    parser.add_argument('--grpo_model', type=str, 
                        default='grpo_model',
                        help='Path to the GRPO-improved model')
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
                        default=20,
                        help='Number of samples to evaluate')
    parser.add_argument('--output_dir', type=str, 
                        default='grpo_evaluation',
                        help='Output directory for evaluation results')
    return parser.parse_args()

def extract_json_from_response(response_text):
    """Extract JSON from model response text"""
    try:
        # Find JSON pattern in the text
        match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            data = json.loads(json_str)
            return data, None
        else:
            return None, "No JSON found in response"
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {str(e)}"
    except Exception as e:
        return None, f"Error extracting JSON: {str(e)}"

def analyze_safety_info(json_data):
    """Analyze safety information in the model response"""
    if not json_data or "detailed_analysis" not in json_data:
        return {
            "has_safety_info": False,
            "tools_with_safety": 0,
            "total_tools": 0,
            "safety_completeness": 0.0,
            "safety_fields": {
                "required_ppe": 0,
                "primary_hazards": 0,
                "common_misuses": 0
            }
        }
    
    # Check if tools_information exists
    if "tools_information" not in json_data["detailed_analysis"]:
        return {
            "has_safety_info": False,
            "tools_with_safety": 0,
            "total_tools": 0,
            "safety_completeness": 0.0,
            "safety_fields": {
                "required_ppe": 0,
                "primary_hazards": 0,
                "common_misuses": 0
            }
        }
    
    # Count tools and safety information
    tools = json_data["detailed_analysis"]["tools_information"]
    total_tools = len(tools)
    tools_with_safety = 0
    
    # Track safety fields
    safety_fields = {
        "required_ppe": 0,
        "primary_hazards": 0,
        "common_misuses": 0
    }
    
    # Analyze each tool
    for tool in tools:
        if "safety_considerations" in tool:
            tools_with_safety += 1
            safety = tool["safety_considerations"]
            
            # Check for required_ppe
            if "required_ppe" in safety and safety["required_ppe"]:
                safety_fields["required_ppe"] += 1
            
            # Check for primary_hazards
            if "primary_hazards" in safety and safety["primary_hazards"]:
                if isinstance(safety["primary_hazards"], list) and len(safety["primary_hazards"]) > 0:
                    safety_fields["primary_hazards"] += 1
                elif isinstance(safety["primary_hazards"], str) and safety["primary_hazards"].strip():
                    safety_fields["primary_hazards"] += 1
            
            # Check for common_misuses
            if "common_misuses" in safety and safety["common_misuses"]:
                if isinstance(safety["common_misuses"], list) and len(safety["common_misuses"]) > 0:
                    safety_fields["common_misuses"] += 1
                elif isinstance(safety["common_misuses"], str) and safety["common_misuses"].strip():
                    safety_fields["common_misuses"] += 1
    
    # Calculate safety completeness (average percentage of safety fields present)
    if total_tools > 0:
        safety_completeness = (
            (safety_fields["required_ppe"] + safety_fields["primary_hazards"] + safety_fields["common_misuses"]) / 
            (total_tools * 3)  # 3 safety fields per tool
        ) * 100
    else:
        safety_completeness = 0.0
    
    return {
        "has_safety_info": tools_with_safety > 0,
        "tools_with_safety": tools_with_safety,
        "total_tools": total_tools,
        "safety_completeness": safety_completeness,
        "safety_fields": safety_fields
    }

def generate_standard_response(image, model, tokenizer):
    """Generate standard response (without RAG or GRPO)"""
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
        max_new_tokens=500,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )
    
    rag_response = tokenizer.decode(output[0], skip_special_tokens=True)
    return rag_response