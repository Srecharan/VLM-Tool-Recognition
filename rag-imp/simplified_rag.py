import os
import pandas as pd
import numpy as np
import json
import torch
import gc
from PIL import Image
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import unsloth

# Import unsloth first to ensure optimizations are applied
from unsloth import FastVisionModel

# Disable torch compilation to avoid CUDA errors
os.environ["TORCH_COMPILE_MODE"] = "reduce-overhead"
os.environ["TORCH_INDUCTOR_DISABLE_CUDAGRAPHS"] = "1"

# Clear GPU memory at start
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

def create_simplified_knowledge_base():
    """Create a simplified knowledge base from the HuggingFace dataset"""
    print("Step 1: Creating knowledge base from HuggingFace dataset...")
    
    # Check if knowledge base already exists
    if os.path.exists('tool_knowledge_base.csv'):
        print("Knowledge base already exists, loading from file...")
        return pd.read_csv('tool_knowledge_base.csv')
    
    # Load the dataset
    dataset = load_dataset("akameswa/tool-safety-dataset", split="valid")
    
    # Find all tool categories
    tool_categories = []
    for column in dataset.column_names:
        if column.endswith("_bboxes"):
            tool_name = column.replace("_bboxes", "")
            tool_categories.append(tool_name)
    
    print(f"Found {len(tool_categories)} tool categories: {', '.join(tool_categories)}")
    
    # Create knowledge base
    knowledge_base = []
    
    for tool in tool_categories:
        # Find the first example with this tool
        for example in dataset:
            bbox_key = f"{tool}_bboxes"
            if bbox_key in example and example[bbox_key]:
                # Extract relevant fields
                entry = {
                    "tool_name": tool,
                    "primary_function": example.get(f"{tool}_main_purpose", ""),
                    "usage_instructions": example.get(f"{tool}_usage_instructions", ""),
                    "safety_considerations": {
                        "required_ppe": example.get(f"{tool}_required_ppe", ""),
                        "primary_hazards": example.get(f"{tool}_primary_hazards", ""),
                        "common_misuses": example.get(f"{tool}_common_misuses", "")
                    }
                }
                knowledge_base.append(entry)
                print(f"Added information for {tool}")
                break
    
    # Save to CSV
    df = pd.DataFrame(knowledge_base)
    df.to_csv('tool_knowledge_base.csv', index=False)
    print(f"Knowledge base created with {len(df)} tools and saved to tool_knowledge_base.csv")
    
    return df

def create_embeddings_and_index(knowledge_df):
    """Create embeddings and index for retrieval"""
    print("Step 2: Creating embeddings and search index...")
    
    # Load embedding model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Create texts for embedding
    texts = []
    for _, row in knowledge_df.iterrows():
        safety_info = row['safety_considerations']
        if isinstance(safety_info, str):
            try:
                safety_info = json.loads(safety_info)
            except:
                pass
                
        text = f"Tool: {row['tool_name']}. "
        text += f"Function: {row['primary_function']}. "
        text += f"Instructions: {row['usage_instructions']}. "
        text += f"Safety information: {safety_info}"
        texts.append(text)
    
    # Generate embeddings
    embeddings = model.encode(texts, batch_size=8, show_progress_bar=True)
    
    # Create FAISS index
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"Created index with {index.ntotal} vectors")
    return model, index

def retrieve_tool_info(query, embedding_model, index, knowledge_df, top_k=3):
    """Retrieve relevant tool information"""
    query_embedding = embedding_model.encode([query])
    query_embedding = query_embedding.astype(np.float32)
    faiss.normalize_L2(query_embedding)
    
    scores, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(knowledge_df):
            entry = knowledge_df.iloc[idx].to_dict()
            entry['score'] = float(scores[0][i])
            results.append(entry)
    
    return results

def run_rag_with_unsloth(model_path, image_path, embedding_model, index, knowledge_df):
    """Run RAG using Unsloth's FastVisionModel matching your evaluation code pattern"""
    print(f"Running RAG with Unsloth model {model_path} on image {image_path}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        # Load model with unsloth - using the EXACT same approach as in your evaluation scripts
        # The key is to use ONLY 4-bit quantization which is what your model was fine-tuned with
        print("Loading model with Unsloth FastVisionModel...")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=True,  # Use 4-bit (default) - NOT both 4-bit and 8-bit
            use_gradient_checkpointing="unsloth",
        )
        FastVisionModel.for_inference(model)
        
        # Load image - resize to reduce memory usage
        print("Loading and processing image...")
        image = Image.open(image_path).convert('RGB')
        resized_image = image.resize((224, 224))
        
        # Step 1: Basic tool identification
        print("Step 1: Basic tool identification")
        basic_instruction = "Identify the mechanical tools visible in this image."
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": basic_instruction}
            ]}
        ]
        
        # Making input preparation more robust with error handling
        try:
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            inputs = tokenizer(
                [resized_image],
                text=[input_text],
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Generate with minimal memory settings
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
            
            tool_identification = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Identified tools: {tool_identification}")
        except Exception as e:
            print(f"Error in tool identification step: {e}")
            # Set a default identification so we can continue with RAG
            tool_identification = "mechanical tools"
        
        # Step 2: Retrieve relevant information
        print("Step 2: Retrieving relevant information from knowledge base")
        retrieved_info = retrieve_tool_info(
            f"Tools identified: {tool_identification}",
            embedding_model,
            index,
            knowledge_df
        )
        
        print(f"Retrieved information for {len(retrieved_info)} tools")
        
        # Free memory before next generation step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Step 3: Create RAG context
        print("Step 3: Creating RAG context from retrieved information")
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
        
        # Create structured JSON output template for the instruction
        json_template = """
        {
            "detected_tools": ["tool_name_1", "tool_name_2", ...],
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
        """
        
        # Step 4: Generate RAG-enhanced response
        print("Step 4: Generating RAG-enhanced response")
        rag_instruction = f"""
        {context}
        
        Using the reference information above, analyze this image and identify all mechanical tools present. 
        Return ONLY a valid JSON object in the following format:
        {json_template}
        
        Important: Return ONLY the JSON object with no additional text.
        """
        
        try:
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": rag_instruction}
                ]}
            ]
            
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            inputs = tokenizer(
                [resized_image],
                text=[input_text],
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            rag_output = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
            
            rag_response = tokenizer.decode(rag_output[0], skip_special_tokens=True)
            print("RAG-enhanced response generated successfully")
        except Exception as e:
            print(f"Error generating RAG response: {e}")
            # Provide partial results if generation fails
            rag_response = f"Error generating full RAG response: {e}"
        
        # Free memory again
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Step 5: Generate standard response for comparison
        print("Step 5: Generating standard response (without RAG)")
        standard_instruction = f"""
        Analyze this image and identify ALL mechanical tools present. Return ONLY a valid JSON object in the following format:
        {json_template}
        
        Important: Return ONLY the JSON object.
        """
        
        try:
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": standard_instruction}
                ]}
            ]
            
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            inputs = tokenizer(
                [resized_image],
                text=[input_text],
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            standard_output = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
            
            standard_response = tokenizer.decode(standard_output[0], skip_special_tokens=True)
            print("Standard response generated successfully")
        except Exception as e:
            print(f"Error generating standard response: {e}")
            standard_response = f"Error generating standard response: {e}"
        
        # Clean up
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return {
            "standard_response": standard_response,
            "rag_response": rag_response,
            "retrieved_info": retrieved_info
        }
        
    except Exception as e:
        print(f"Error in RAG implementation: {e}")
        
        # Clean up in case of error
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        return {
            "standard_response": f"Error: {str(e)}",
            "rag_response": f"Error: {str(e)}",
            "retrieved_info": []
        }

def extract_json_from_response(response_text):
    """Extract JSON from the model's response text"""
    try:
        # Find the JSON content by looking for balanced braces
        response_text = response_text.strip()
        
        # If response starts with triple backticks (code block), remove them
        if response_text.startswith("```json"):
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            if json_end != -1:
                response_text = response_text[json_start:json_end].strip()
        
        # Find the first opening brace
        start_idx = response_text.find('{')
        if start_idx == -1:
            return None
        
        # Find the matching closing brace
        brace_count = 0
        for i in range(start_idx, len(response_text)):
            if response_text[i] == '{':
                brace_count += 1
            elif response_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count != 0:
            # Unbalanced braces
            return None
            
        json_str = response_text[start_idx:end_idx]
        return json.loads(json_str)
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None

def main():
    # Create output directory
    os.makedirs("rag_results", exist_ok=True)
    
    # Step 1: Create knowledge base
    knowledge_df = create_simplified_knowledge_base()
    
    # Step 2: Create embeddings and index
    embedding_model, index = create_embeddings_and_index(knowledge_df)
    
    # Step 3: Run RAG with fine-tuned model using unsloth
    model_path = "akameswa/Qwen2.5-VL-7B-Instruct-bnb-4bit-finetune-vision-language"
    
    # Find a test image
    test_image_dir = "test_images"
    os.makedirs(test_image_dir, exist_ok=True)
    
    # Either use an existing image or get one from the dataset
    test_images = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        print("No test images found, downloading one from the dataset...")
        dataset = load_dataset("akameswa/tool-safety-dataset", split="valid")
        
        for i, example in enumerate(dataset):
            if 'image' in example and example['image'] is not None:
                image = example['image']
                image_path = os.path.join(test_image_dir, f"sample_image_{i}.jpg")
                image.save(image_path)
                print(f"Saved sample image to {image_path}")
                break
        
        test_images = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        print("Failed to find or download test images.")
        return
    
    # Run RAG on the first test image
    test_image_path = os.path.join(test_image_dir, test_images[0])
    
    results = run_rag_with_unsloth(
        model_path=model_path,
        image_path=test_image_path,
        embedding_model=embedding_model,
        index=index,
        knowledge_df=knowledge_df
    )
    
    # Extract JSON from responses if needed
    standard_json = extract_json_from_response(results["standard_response"])
    rag_json = extract_json_from_response(results["rag_response"])
    
    # Save results
    output_path = os.path.join("rag_results", "final_rag_results.json")
    with open(output_path, 'w') as f:
        # Ensure results are serializable
        serializable_results = {
            "standard_response": results["standard_response"],
            "rag_response": results["rag_response"],
            "standard_json": standard_json,
            "rag_json": rag_json,
            "retrieved_info": [{k: v for k, v in item.items() if isinstance(v, (str, int, float, bool, list, dict))} 
                              for item in results["retrieved_info"]]
        }
        json.dump(serializable_results, f, indent=2)
    
    # Also create an analysis file with the differences
    analysis_path = os.path.join("rag_results", "rag_impact_analysis.txt")
    with open(analysis_path, 'w') as f:
        f.write("RAG IMPLEMENTATION IMPACT ANALYSIS\n")
        f.write("================================\n\n")
        
        # Compare tool detection
        f.write("TOOL DETECTION COMPARISON:\n")
        f.write("------------------------\n")
        if standard_json and "detected_tools" in standard_json:
            f.write(f"Standard detected tools: {', '.join(standard_json['detected_tools'])}\n")
        else:
            f.write("Standard response did not contain valid detected tools JSON\n")
            
        if rag_json and "detected_tools" in rag_json:
            f.write(f"RAG detected tools: {', '.join(rag_json['detected_tools'])}\n")
        else:
            f.write("RAG response did not contain valid detected tools JSON\n")
        f.write("\n")
        
        # Compare safety information
        f.write("SAFETY INFORMATION COMPARISON:\n")
        f.write("---------------------------\n")
        
        # Function to extract safety info from JSON
        def extract_safety_info(json_obj):
            safety_info = []
            if json_obj and "detailed_analysis" in json_obj and "tools_information" in json_obj["detailed_analysis"]:
                for tool_info in json_obj["detailed_analysis"]["tools_information"]:
                    if "safety_considerations" in tool_info:
                        tool_name = tool_info.get("tool", "Unknown tool")
                        safety_info.append({
                            "tool": tool_name,
                            "safety": tool_info["safety_considerations"]
                        })
            return safety_info
        
        # Extract safety info
        standard_safety = extract_safety_info(standard_json)
        rag_safety = extract_safety_info(rag_json)
        
        # Write safety comparison
        f.write("Standard response safety information:\n")
        if standard_safety:
            for info in standard_safety:
                f.write(f"- {info['tool']}:\n")
                for k, v in info['safety'].items():
                    f.write(f"  - {k}: {v}\n")
        else:
            f.write("No structured safety information found in standard response\n")
            
        f.write("\nRAG-enhanced response safety information:\n")
        if rag_safety:
            for info in rag_safety:
                f.write(f"- {info['tool']}:\n")
                for k, v in info['safety'].items():
                    f.write(f"  - {k}: {v}\n")
        else:
            f.write("No structured safety information found in RAG response\n")
        f.write("\n")
        
        # Include retrieved knowledge
        f.write("KNOWLEDGE RETRIEVED FROM RAG:\n")
        f.write("---------------------------\n")
        for i, info in enumerate(results["retrieved_info"]):
            f.write(f"Tool {i+1}: {info['tool_name']}\n")
            f.write(f"Function: {info['primary_function']}\n")
            safety = info['safety_considerations']
            if isinstance(safety, dict):
                for k, v in safety.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write(f"Safety: {safety}\n")
            f.write("\n")
        
        # Full responses for reference
        f.write("\nFULL RESPONSES:\n")
        f.write("-------------\n")
        f.write("1. STANDARD RESPONSE (WITHOUT RAG):\n")
        f.write(results["standard_response"])
        f.write("\n\n")
        f.write("2. RAG-ENHANCED RESPONSE:\n")
        f.write(results["rag_response"])
    
    print(f"Results saved to {output_path}")
    print(f"Analysis saved to {analysis_path}")
    print("\nRAG implementation completed successfully!")

if __name__ == "__main__":
    main()