import pandas as pd
import json
import requests
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyAELVREoOEr2TnyxdOn5dt4DDQ54pQbXJQ"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# File paths
QWEN_MODELS = ["qwen-v", "qwen-l", "qwen-vl", "qwen"]  # Zero-shot model is named 'qwen'
FILE_PATH = "/home/rex/VLM-Tool-Recognition/evals/results/"

def read_qwen_csv(model_name):
    """Read the CSV file for a specific Qwen model."""
    file_path = f"{FILE_PATH}{model_name}.csv"
    df = pd.read_csv(file_path)
    return df

def parse_full_response(full_response):
    """Parse the 'full_response' column to extract tool information."""
    try:
        # Find the JSON part in the response (after the bounding boxes)
        if ']{"tools_information"' in full_response:
            json_str = full_response.split(']', 1)[1]
        else:
            json_str = full_response
            
        # Load the JSON data
        data = json.loads(json_str)
        
        # Extract tools information
        if "tools_information" in data:
            return data["tools_information"]
        return []
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Problematic response: {full_response}")
        return []

def evaluate_with_gemini(tool_info, gemini_api_key):
    """Use Gemini API to evaluate the tool information."""
    # Prepare the prompt for Gemini
    prompt = f"""
    Evaluate the following information about a mechanical tool on a scale of 0-10, where 0 is completely incorrect and 10 is perfectly accurate:
    
    Tool: {tool_info.get('tool', 'Unknown')}
    Primary Function: {tool_info.get('primary_function', 'Unknown')}
    Required PPE: {tool_info.get('safety_considerations', {}).get('required_ppe', 'Unknown')}
    Primary Hazards: {tool_info.get('safety_considerations', {}).get('primary_hazards', [])}
    Common Misuses: {tool_info.get('safety_considerations', {}).get('common_misuses', [])}
    
    Please provide numerical scores (0-10) for each of the following aspects:
    1. Tool Identification Accuracy: How accurately is the tool identified?
    2. Primary Function Accuracy: How accurately is the primary function described?
    3. Safety Considerations Accuracy: How accurately are the PPE and hazards described?
    4. Common Misuses Accuracy: How accurately are the common misuses described?
    5. Overall Score: An overall assessment of the information's accuracy and completeness.
    
    Return your evaluation as a JSON object with these exact keys: tool_score, function_score, safety_score, misuse_score, overall_score.
    """
    
    # Prepare the API request
    headers = {'Content-Type': 'application/json'}
    data = {
        'contents': [{
            'parts': [{'text': prompt}]
        }]
    }
    
    # Add API key to URL
    url = f"{GEMINI_API_URL}?key={gemini_api_key}"
    
    # Make the API call
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        # Parse the response to extract the scores
        result = response.json()
        text_response = result['candidates'][0]['content']['parts'][0]['text']
        
        # Extract JSON from the text response
        json_str = text_response
        try:
            # Try to find JSON block in the response
            import re
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            
            scores = json.loads(json_str)
            return {
                'tool_score': scores.get('tool_score', 0),
                'function_score': scores.get('function_score', 0),
                'safety_score': scores.get('safety_score', 0),
                'misuse_score': scores.get('misuse_score', 0),
                'overall_score': scores.get('overall_score', 0)
            }
        except json.JSONDecodeError:
            # If JSON extraction fails, parse manually
            scores = {}
            for line in text_response.split('\n'):
                if 'tool_score' in line.lower():
                    scores['tool_score'] = int(re.search(r'\d+', line).group(0))
                elif 'function_score' in line.lower():
                    scores['function_score'] = int(re.search(r'\d+', line).group(0))
                elif 'safety_score' in line.lower():
                    scores['safety_score'] = int(re.search(r'\d+', line).group(0))
                elif 'misuse_score' in line.lower():
                    scores['misuse_score'] = int(re.search(r'\d+', line).group(0))
                elif 'overall_score' in line.lower():
                    scores['overall_score'] = int(re.search(r'\d+', line).group(0))
            
            return scores
            
    except Exception as e:
        print(f"API call error: {e}")
        # Return default scores on error
        return {
            'tool_score': 0,
            'function_score': 0,
            'safety_score': 0,
            'misuse_score': 0,
            'overall_score': 0
        }

def evaluate_model(model_name, max_samples=50):
    """Evaluate a specific Qwen model."""
    print(f"Evaluating {model_name}...")
    df = read_qwen_csv(model_name)
    
    # Limit samples for testing if needed
    if max_samples > 0:
        df = df.head(max_samples)
    
    all_scores = []
    
    # Process each row in the dataframe
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {model_name}"):
        try:
            full_response = row['full_response']
            tools_info = parse_full_response(full_response)
            
            # Evaluate each tool in the response
            for tool_info in tools_info:
                scores = evaluate_with_gemini(tool_info, GEMINI_API_KEY)
                scores['filename'] = row['filename']
                scores['tool'] = tool_info.get('tool', 'Unknown')
                all_scores.append(scores)
                
                # Add a short delay to avoid API rate limiting
                time.sleep(0.5)
        except Exception as e:
            print(f"Error processing row: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_scores)
    
    # Save results
    results_df.to_csv(f"{model_name}_evaluation_results.csv", index=False)
    
    return results_df

def plot_model_comparison(model_results):
    """Create visualizations comparing model performance."""
    # Prepare data for plotting
    models = list(model_results.keys())
    metrics = ['tool_score', 'function_score', 'safety_score', 'misuse_score', 'overall_score']
    metric_labels = ['Tool ID', 'Function', 'Safety', 'Misuses', 'Overall']
    
    # Calculate average scores for each model and metric
    avg_scores = {model: {} for model in models}
    for model, df in model_results.items():
        for metric in metrics:
            avg_scores[model][metric] = df[metric].mean()
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(metrics))
    width = 0.2
    multiplier = 0
    
    for model, scores in avg_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, [scores[metric] for metric in metrics], width, label=model)
        multiplier += 1
    
    # Add labels and legend
    ax.set_xlabel('Evaluation Metrics')
    ax.set_ylabel('Average Score (0-10)')
    ax.set_title('Qwen Model Performance Comparison')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(models))
    ax.set_ylim(0, 10)
    
    # Save the chart
    plt.tight_layout()
    plt.savefig('qwen_model_comparison.png')
    
    # Create radar chart for more detailed visualization
    metrics_count = len(metrics)
    angles = np.linspace(0, 2*np.pi, metrics_count, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for model, scores in avg_scores.items():
        values = [scores[metric] for metric in metrics]
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 10)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Radar Chart: Qwen Model Performance')
    
    # Save radar chart
    plt.tight_layout()
    plt.savefig('qwen_model_radar_chart.png')
    
    return fig

def main():
    """Main function to run the evaluation."""
    # Evaluate each model
    model_results = {}
    for model in QWEN_MODELS:
        results_df = evaluate_model(model, max_samples=50)  # Limit to 50 samples for testing
        model_results[model] = results_df
    
    # Generate comparison visualizations
    plot_model_comparison(model_results)
    
    print("Evaluation complete. Results saved to CSV files and visualizations.")

if __name__ == "__main__":
    main()