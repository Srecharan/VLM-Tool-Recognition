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
import ast

# Configuration
GEMINI_API_KEY = "your-new-api-key-here"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
RESULTS_DIR = "./evaluation_results"
PLOTS_DIR = "./evaluation_plots"

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def parse_llama_response(response_text):
    """Parse a LLaMA VLM response string to extract tool information."""
    # First, try to decode the string if it's in a list-like format
    try:
        # The response is typically wrapped in a list with escaped characters
        # We need to parse this to get the actual content
        parsed_list = ast.literal_eval(response_text)
        if isinstance(parsed_list, list) and len(parsed_list) > 0:
            response_text = parsed_list[0]
    except:
        # If parsing fails, use the response text as is
        pass
    
    # Clean up the response text
    response_text = response_text.replace('\\n', '\n')
    
    # Split the text to separate the user prompt and assistant response
    parts = response_text.split('assistant\\n')
    if len(parts) < 2:
        parts = response_text.split('assistant')
    
    if len(parts) < 2:
        # If we still can't find the assistant part, return None
        return None
    
    # Get the assistant's response
    assistant_response = parts[1].strip()
    
    # Extract tool information using regex patterns
    tool_info = {}
    
    # Extract tool name
    tool_match = re.search(r'\*\*Tool\*\*:\s*(.+?)(?:\n|$)', assistant_response)
    if not tool_match:
        tool_match = re.search(r'^([A-Za-z\s]+)(?:\n|$)', assistant_response)
    
    if tool_match:
        tool_info['tool_name'] = tool_match.group(1).strip()
    else:
        # If no tool name found, try other formats
        for line in assistant_response.split('\n'):
            if line.lower().startswith('tool:') or line == line.strip() and len(line.strip()) > 0 and len(line.strip()) < 30:
                tool_info['tool_name'] = line.replace('Tool:', '').strip()
                break
    
    # Extract primary function
    function_match = re.search(r'\*\*Primary Function\*\*:\s*(.+?)(?:\n|$)', assistant_response)
    if not function_match:
        function_match = re.search(r'Primary Function:\s*(.+?)(?:\n|$)', assistant_response)
    if not function_match:
        function_match = re.search(r'(?:\n|^)Primary Function[:]*\s*(.+?)(?:\n|$)', assistant_response, re.IGNORECASE)
    
    if function_match:
        tool_info['primary_function'] = function_match.group(1).strip()
    
    # Extract required PPE
    ppe_match = re.search(r'\*\*Required PPE\*\*:\s*(.+?)(?:\n|$)', assistant_response)
    if not ppe_match:
        ppe_match = re.search(r'Required PPE:\s*(.+?)(?:\n|$)', assistant_response)
    if not ppe_match:
        ppe_match = re.search(r'PPE:\s*(.+?)(?:\n|$)', assistant_response)
    if not ppe_match:
        ppe_match = re.search(r'Safety Equipment:\s*(.+?)(?:\n|$)', assistant_response)
    
    if ppe_match:
        tool_info['required_ppe'] = ppe_match.group(1).strip()
    else:
        tool_info['required_ppe'] = ""
    
    # Extract primary hazards
    hazards_match = re.search(r'\*\*Primary Hazards\*\*:\s*(.+?)(?:\n|$)', assistant_response)
    if not hazards_match:
        hazards_match = re.search(r'Primary Hazards:\s*(.+?)(?:\n|$)', assistant_response)
    if not hazards_match:
        hazards_match = re.search(r'Hazards:\s*(.+?)(?:\n|$)', assistant_response)
    
    if hazards_match:
        hazards_text = hazards_match.group(1).strip()
        # Convert to list if it contains semicolons or commas
        if ';' in hazards_text:
            tool_info['primary_hazards'] = [h.strip() for h in hazards_text.split(';')]
        elif ',' in hazards_text:
            tool_info['primary_hazards'] = [h.strip() for h in hazards_text.split(',')]
        else:
            tool_info['primary_hazards'] = [hazards_text]
    else:
        tool_info['primary_hazards'] = []
    
    # Extract common misuses
    misuses_match = re.search(r'\*\*Common Misuses\*\*:\s*(.+?)(?:\n|$)', assistant_response)
    if not misuses_match:
        misuses_match = re.search(r'Common Misuses:\s*(.+?)(?:\n|$)', assistant_response)
    if not misuses_match:
        misuses_match = re.search(r'Misuses:\s*(.+?)(?:\n|$)', assistant_response)
    
    if misuses_match:
        misuses_text = misuses_match.group(1).strip()
        # Convert to list if it contains semicolons or commas
        if ';' in misuses_text:
            tool_info['common_misuses'] = [m.strip() for m in misuses_text.split(';')]
        elif ',' in misuses_text:
            tool_info['common_misuses'] = [m.strip() for m in misuses_text.split(',')]
        else:
            tool_info['common_misuses'] = [misuses_text]
    else:
        tool_info['common_misuses'] = []
    
    # Return None if we don't have at least a tool name and one other piece of information
    if 'tool_name' not in tool_info or (
        'primary_function' not in tool_info and 
        'required_ppe' not in tool_info and 
        not tool_info['primary_hazards'] and 
        not tool_info['common_misuses']
    ):
        return None
    
    return tool_info

def parse_llama_csv(file_path, max_samples=100):
    """Parse a LLaMA CSV file and extract tool information, limiting to max_samples."""
    try:
        df = pd.read_csv(file_path)
        
        # Limit to max_samples
        if max_samples > 0 and len(df) > max_samples:
            df = df.head(max_samples)
        
        # Create an empty list to store parsed data
        parsed_tools = []
        skipped_count = 0
        
        # Process each row in the dataframe
        for idx, row in df.iterrows():
            try:
                filename = row['filename']
                response = row['response']
                
                # Parse the LLaMA response
                tool_info = parse_llama_response(response)
                
                if tool_info:
                    tool_info['filename'] = filename
                    parsed_tools.append(tool_info)
                else:
                    skipped_count += 1
                    print(f"Could not extract tool information from row {idx}, filename: {filename}")
            except Exception as e:
                skipped_count += 1
                print(f"Error processing row {idx}: {e}")
        
        print(f"Parsed {len(parsed_tools)} tools, skipped {skipped_count} rows")
        return parsed_tools
    
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return []

def evaluate_tool_with_gemini(tool_info, api_key):
    """Evaluate a single tool's information using the Gemini API."""
    # Extract tool information
    tool_name = tool_info.get("tool_name", "Unknown")
    primary_function = tool_info.get("primary_function", "")
    required_ppe = tool_info.get("required_ppe", "")
    primary_hazards = tool_info.get("primary_hazards", [])
    common_misuses = tool_info.get("common_misuses", [])
    
    # Convert lists to strings for display
    if isinstance(primary_hazards, list):
        primary_hazards = ", ".join(primary_hazards)
    if isinstance(common_misuses, list):
        common_misuses = ", ".join(common_misuses)
    
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
    parsed_tools = parse_llama_csv(file_path, max_samples)
    
    # If no tools were parsed, return an empty DataFrame
    if not parsed_tools:
        print(f"No tools could be parsed from {model_name}")
        return pd.DataFrame()
    
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
    bars = plt.bar(metrics, scores, color='salmon')
    
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
    parser = argparse.ArgumentParser(description='Evaluate LLaMA model outputs using Gemini API')
    parser.add_argument('--base_path', type=str, default='/home/rex/VLM-Tool-Recognition/evals/results',
                        help='Base path to the CSV files')
    parser.add_argument('--model', type=str, default='llama-v',
                        help='Model name to evaluate')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to evaluate (default: 100)')
    
    args = parser.parse_args()
    
    # Evaluate the specified model
    results_df = evaluate_model(args.model, args.base_path, args.max_samples)
    
    # If no results, exit
    if len(results_df) == 0:
        print(f"No results obtained for {args.model}. Exiting.")
        return
    
    # Create visualization
    create_visualization(args.model, results_df)
    
    # Display summary statistics
    print("\nSummary Statistics:")
    summary = {
        'Model': args.model,
        'Samples Evaluated': len(results_df),
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