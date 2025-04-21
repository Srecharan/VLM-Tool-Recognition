import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import requests
import time
import re
import argparse

# Configuration
GEMINI_API_KEY = "your-new-api-key-here"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
RESULTS_DIR = "./evaluation_results"
PLOTS_DIR = "./evaluation_plots"

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def parse_qwen_csv(file_path, max_samples=100):
    """Parse a Qwen CSV file and extract tool information, limiting to max_samples."""
    df = pd.read_csv(file_path)
    
    # Limit to max_samples
    if max_samples > 0 and len(df) > max_samples:
        df = df.head(max_samples)
    
    # Create an empty list to store parsed data
    parsed_rows = []
    
    # Process each row in the dataframe
    for idx, row in df.iterrows():
        try:
            filename = row['filename']
            full_response = row['full_response']
            
            # Find JSON content in the full_response
            try:
                # First attempt: look for the tools_information JSON
                if ']{"tools_information"' in full_response:
                    json_str = full_response.split(']', 1)[1]
                    data = json.loads(json_str)
                    
                    if "tools_information" in data:
                        tools = data["tools_information"]
                        
                        # Process each tool
                        for tool in tools:
                            tool_entry = {
                                "filename": filename,
                                "tool_name": tool.get("tool", "Unknown"),
                                "primary_function": tool.get("primary_function", ""),
                                "safety_considerations": {}
                            }
                            
                            # Extract safety considerations if available
                            safety = tool.get("safety_considerations", {})
                            if safety:
                                tool_entry["safety_considerations"] = {
                                    "required_ppe": safety.get("required_ppe", ""),
                                    "primary_hazards": safety.get("primary_hazards", []),
                                    "common_misuses": safety.get("common_misuses", [])
                                }
                            
                            parsed_rows.append(tool_entry)
                else:
                    # Look for any JSON in the response
                    match = re.search(r'(\{.*\})', full_response, re.DOTALL)
                    if match:
                        try:
                            data = json.loads(match.group(0))
                            if "tools_information" in data:
                                tools = data["tools_information"]
                                for tool in tools:
                                    tool_entry = {
                                        "filename": filename,
                                        "tool_name": tool.get("tool", "Unknown"),
                                        "primary_function": tool.get("primary_function", ""),
                                        "safety_considerations": {}
                                    }
                                    
                                    safety = tool.get("safety_considerations", {})
                                    if safety:
                                        tool_entry["safety_considerations"] = {
                                            "required_ppe": safety.get("required_ppe", ""),
                                            "primary_hazards": safety.get("primary_hazards", []),
                                            "common_misuses": safety.get("common_misuses", [])
                                        }
                                    
                                    parsed_rows.append(tool_entry)
                        except json.JSONDecodeError:
                            print(f"Could not parse JSON in row {idx}, filename: {filename}")
            except Exception as e:
                print(f"Error processing row {idx}, filename: {filename}: {e}")
                
        except Exception as e:
            print(f"General error in row {idx}: {e}")
    
    return parsed_rows

def evaluate_tool_with_gemini(tool_info, api_key):
    """Evaluate a single tool's information using the Gemini API."""
    # Extract tool information
    tool_name = tool_info.get("tool_name", "Unknown")
    primary_function = tool_info.get("primary_function", "")
    
    # Extract safety considerations
    safety = tool_info.get("safety_considerations", {})
    required_ppe = safety.get("required_ppe", "")
    primary_hazards = safety.get("primary_hazards", [])
    common_misuses = safety.get("common_misuses", [])
    
    # Prepare the prompt for Gemini
    prompt = f"""
    As an expert in mechanical tools and safety, evaluate the accuracy and completeness of the following tool information on a scale of 0-10 (where 0 is completely incorrect and 10 is perfectly accurate and complete):
    
    Tool: {tool_name}
    Primary Function: {primary_function}
    Required PPE: {required_ppe}
    Primary Hazards: {primary_hazards}
    Common Misuses: {common_misuses}
    
    Please provide individual numerical scores (0-10) for each of these aspects:
    1. Tool Identification: How accurately is the tool identified?
    2. Primary Function: How accurately and completely is the primary function described?
    3. Safety Considerations: How accurately and completely are the required PPE and hazards described?
    4. Common Misuses: How accurately and completely are the common misuses described?
    5. Overall Score: An overall assessment of all information.
    
    Return ONLY a JSON object with these five scores, using this exact format:
    {{"tool_score": X, "function_score": Y, "safety_score": Z, "misuse_score": W, "overall_score": V}}
    """
    
    # Prepare the API request
    headers = {'Content-Type': 'application/json'}
    data = {
        'contents': [{
            'parts': [{'text': prompt}]
        }]
    }
    
    # Add API key to URL
    url = f"{GEMINI_API_URL}?key={api_key}"
    
    # Make the API request
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            # Parse the response to extract the scores
            result = response.json()
            text_response = result['candidates'][0]['content']['parts'][0]['text']
            
            # Try to extract the JSON object from the response
            json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group(0))
                return {
                    'tool_score': scores.get('tool_score', 0),
                    'function_score': scores.get('function_score', 0), 
                    'safety_score': scores.get('safety_score', 0),
                    'misuse_score': scores.get('misuse_score', 0),
                    'overall_score': scores.get('overall_score', 0)
                }
            else:
                # If no JSON found, try to parse manually
                scores = {}
                for line in text_response.split('\n'):
                    if 'tool' in line.lower() and ':' in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            scores['tool_score'] = int(match.group(1))
                    elif 'function' in line.lower() and ':' in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            scores['function_score'] = int(match.group(1))
                    elif 'safety' in line.lower() and ':' in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            scores['safety_score'] = int(match.group(1))
                    elif 'misuse' in line.lower() and ':' in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            scores['misuse_score'] = int(match.group(1))
                    elif 'overall' in line.lower() and ':' in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            scores['overall_score'] = int(match.group(1))
                
                # Check if we have all scores
                if all(k in scores for k in ['tool_score', 'function_score', 'safety_score', 'misuse_score', 'overall_score']):
                    return scores
                else:
                    raise ValueError("Could not extract all scores from the response")
            
        except requests.exceptions.RequestException as e:
            print(f"API request error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("Max retries reached. Using default scores.")
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("Max retries reached. Using default scores.")
    
    # Return default scores if all attempts failed
    return {
        'tool_score': 0,
        'function_score': 0,
        'safety_score': 0,
        'misuse_score': 0,
        'overall_score': 0
    }

def evaluate_model(model_name, base_path, max_samples=100):
    """Evaluate all tools in a model's CSV file."""
    print(f"Evaluating model: {model_name}")
    file_path = os.path.join(base_path, f"{model_name}.csv")
    
    # Parse the CSV file
    parsed_tools = parse_qwen_csv(file_path, max_samples)
    
    # Evaluate each tool
    evaluation_results = []
    
    for i, tool_info in enumerate(tqdm(parsed_tools, desc=f"Evaluating {model_name}")):
        # Get the filename and tool name for tracking
        filename = tool_info.get("filename", "unknown")
        tool_name = tool_info.get("tool_name", "unknown")
        
        # Evaluate with Gemini
        scores = evaluate_tool_with_gemini(tool_info, GEMINI_API_KEY)
        
        # Add metadata to scores
        scores["filename"] = filename
        scores["tool_name"] = tool_name
        scores["model"] = model_name
        
        # Add to results
        evaluation_results.append(scores)
        
        # Add a small delay to avoid API rate limiting (10 RPM = 1 every 6 seconds)
        if i < len(parsed_tools) - 1:  # No need to sleep after the last item
            time.sleep(6)  # 6 seconds to stay well under the 10 RPM limit
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(evaluation_results)
    output_path = os.path.join(RESULTS_DIR, f"{model_name}_evaluation.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"Evaluation completed for {model_name}. Results saved to {output_path}")
    return results_df

def create_visualization(model_name, results_df):
    """Create a simple bar chart showing average scores for a single model."""
    # Calculate average scores for each metric (excluding tool_score)
    avg_scores = {
        'Function': results_df['function_score'].mean(),
        'Safety': results_df['safety_score'].mean(),
        'Misuses': results_df['misuse_score'].mean(),
        'Overall': results_df['overall_score'].mean()
    }
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    metrics = list(avg_scores.keys())
    scores = list(avg_scores.values())
    
    # Create the bar chart
    bars = plt.bar(metrics, scores, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}', ha='center', va='bottom')
    
    # Customize the plot
    plt.title(f'{model_name} Average Scores', fontsize=16)
    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Average Score (0-10)')
    plt.ylim(0, 10.5)  # Set y-axis limit to leave room for value labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, f'{model_name}_scores.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {plot_path}")
    return plot_path

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate Qwen model outputs using Gemini API')
    parser.add_argument('--base_path', type=str, default='/home/rex/VLM-Tool-Recognition/evals/results',
                        help='Base path to the CSV files')
    parser.add_argument('--model', type=str, default='qwen-v',
                        help='Model name to evaluate')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to evaluate (default: 100)')
    
    args = parser.parse_args()
    
    # Evaluate the specified model
    results_df = evaluate_model(args.model, args.base_path, args.max_samples)
    
    # Create visualization
    create_visualization(args.model, results_df)
    
    # Display summary statistics
    print("\nSummary Statistics:")
    summary = {
        'Model': args.model,
        'Samples Evaluated': len(results_df),
        'Avg Tool Score': results_df['tool_score'].mean(),
        'Avg Function Score': results_df['function_score'].mean(),
        'Avg Safety Score': results_df['safety_score'].mean(),
        'Avg Misuse Score': results_df['misuse_score'].mean(),
        'Avg Overall Score': results_df['overall_score'].mean()
    }
    
    # Print summary in a formatted way
    for key, value in summary.items():
        if key == 'Model' or key == 'Samples Evaluated':
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    main()