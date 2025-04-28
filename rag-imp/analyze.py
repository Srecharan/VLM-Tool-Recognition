import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from tqdm import tqdm

def extract_json_from_response(response):
    """
    Extract JSON from the model's response
    """
    try:
        # Try to find JSON block in the response
        json_match = re.search(r'(\{.*\})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            return None
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None

def analyze_tool_information(json_data):
    """
    Analyze the quality of tool information in the JSON response
    """
    if not json_data or not isinstance(json_data, dict):
        return {
            "tools_detected": 0,
            "has_bounding_boxes": False,
            "has_detailed_analysis": False,
            "complete_safety_info": False,
            "total_safety_fields": 0
        }
    
    # Check for detected tools
    tools_detected = 0
    if "detected_tools" in json_data and isinstance(json_data["detected_tools"], list):
        tools_detected = len(json_data["detected_tools"])
    
    # Check for bounding boxes
    has_bounding_boxes = "bounding_boxes" in json_data and isinstance(json_data["bounding_boxes"], list) and len(json_data["bounding_boxes"]) > 0
    
    # Check for detailed analysis
    has_detailed_analysis = False
    complete_safety_info = False
    total_safety_fields = 0
    
    if "detailed_analysis" in json_data and "tools_information" in json_data["detailed_analysis"]:
        has_detailed_analysis = True
        tools_info = json_data["detailed_analysis"]["tools_information"]
        
        if isinstance(tools_info, list) and len(tools_info) > 0:
            # Count tools with complete safety information
            safety_fields = 0
            for tool in tools_info:
                if "safety_considerations" in tool:
                    safety = tool["safety_considerations"]
                    if isinstance(safety, dict):
                        if "required_ppe" in safety:
                            safety_fields += 1
                        if "primary_hazards" in safety and isinstance(safety["primary_hazards"], list):
                            safety_fields += 1
                        if "common_misuses" in safety and isinstance(safety["common_misuses"], list):
                            safety_fields += 1
            
            total_safety_fields = safety_fields
            # Check if at least one tool has complete safety information
            complete_safety_info = safety_fields >= 3
    
    return {
        "tools_detected": tools_detected,
        "has_bounding_boxes": has_bounding_boxes,
        "has_detailed_analysis": has_detailed_analysis,
        "complete_safety_info": complete_safety_info,
        "total_safety_fields": total_safety_fields
    }

def analyze_batch_results(batch_results_path, output_dir='rag_analysis'):
    """
    Analyze batch processing results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load batch results
    df = pd.read_csv(batch_results_path)
    
    # Initialize results containers
    standard_analysis = []
    enhanced_analysis = []
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing results"):
        # Extract JSON from responses
        standard_json = extract_json_from_response(row['standard_response'])
        enhanced_json = extract_json_from_response(row['enhanced_response'])
        
        # Analyze tool information
        standard_info = analyze_tool_information(standard_json)
        enhanced_info = analyze_tool_information(enhanced_json)
        
        # Add image name
        standard_info['image_name'] = row['image_name']
        enhanced_info['image_name'] = row['image_name']
        
        # Add to results
        standard_analysis.append(standard_info)
        enhanced_analysis.append(enhanced_info)
    
    # Convert to DataFrames
    standard_df = pd.DataFrame(standard_analysis)
    enhanced_df = pd.DataFrame(enhanced_analysis)
    
    # Calculate improvement metrics
    improvement_metrics = []
    
    for i in range(len(standard_df)):
        improvement = {
            'image_name': standard_df.iloc[i]['image_name'],
            'tools_detected_diff': enhanced_df.iloc[i]['tools_detected'] - standard_df.iloc[i]['tools_detected'],
            'bounding_boxes_improved': enhanced_df.iloc[i]['has_bounding_boxes'] and not standard_df.iloc[i]['has_bounding_boxes'],
            'detailed_analysis_improved': enhanced_df.iloc[i]['has_detailed_analysis'] and not standard_df.iloc[i]['has_detailed_analysis'],
            'safety_info_improved': enhanced_df.iloc[i]['complete_safety_info'] and not standard_df.iloc[i]['complete_safety_info'],
            'safety_fields_diff': enhanced_df.iloc[i]['total_safety_fields'] - standard_df.iloc[i]['total_safety_fields']
        }
        improvement_metrics.append(improvement)
    
    improvement_df = pd.DataFrame(improvement_metrics)
    
    # Calculate summary statistics
    summary = {
        'standard_tools_detected_avg': standard_df['tools_detected'].mean(),
        'enhanced_tools_detected_avg': enhanced_df['tools_detected'].mean(),
        'standard_has_bounding_boxes_pct': standard_df['has_bounding_boxes'].mean() * 100,
        'enhanced_has_bounding_boxes_pct': enhanced_df['has_bounding_boxes'].mean() * 100,
        'standard_has_detailed_analysis_pct': standard_df['has_detailed_analysis'].mean() * 100,
        'enhanced_has_detailed_analysis_pct': enhanced_df['has_detailed_analysis'].mean() * 100,
        'standard_complete_safety_info_pct': standard_df['complete_safety_info'].mean() * 100,
        'enhanced_complete_safety_info_pct': enhanced_df['complete_safety_info'].mean() * 100,
        'standard_safety_fields_avg': standard_df['total_safety_fields'].mean(),
        'enhanced_safety_fields_avg': enhanced_df['total_safety_fields'].mean(),
        'tools_detected_improvement_avg': improvement_df['tools_detected_diff'].mean(),
        'safety_fields_improvement_avg': improvement_df['safety_fields_diff'].mean(),
        'bounding_boxes_improved_pct': improvement_df['bounding_boxes_improved'].mean() * 100,
        'detailed_analysis_improved_pct': improvement_df['detailed_analysis_improved'].mean() * 100,
        'safety_info_improved_pct': improvement_df['safety_info_improved'].mean() * 100
    }
    
    # Save analysis results
    standard_df.to_csv(os.path.join(output_dir, 'standard_analysis.csv'), index=False)
    enhanced_df.to_csv(os.path.join(output_dir, 'enhanced_analysis.csv'), index=False)
    improvement_df.to_csv(os.path.join(output_dir, 'improvement_analysis.csv'), index=False)
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    
    # Create visualizations
    create_visualizations(standard_df, enhanced_df, improvement_df, output_dir)
    
    return standard_df, enhanced_df, improvement_df, summary

def create_visualizations(standard_df, enhanced_df, improvement_df, output_dir):
    """
    Create visualizations comparing standard and RAG-enhanced results
    """
    # 1. Comparison bar chart for averages
    plt.figure(figsize=(12, 6))
    
    # Metrics to compare
    metrics = ['tools_detected', 'total_safety_fields']
    metric_labels = ['Tools Detected', 'Safety Fields']
    
    # Calculate means
    standard_means = [standard_df[metric].mean() for metric in metrics]
    enhanced_means = [enhanced_df[metric].mean() for metric in metrics]
    
    # Plot
    x = np.arange(len(metric_labels))
    width = 0.35
    
    plt.bar(x - width/2, standard_means, width, label='Standard', color='skyblue')
    plt.bar(x + width/2, enhanced_means, width, label='RAG-Enhanced', color='orange')
    
    plt.xlabel('Metrics')
    plt.ylabel('Average Values')
    plt.title('Comparison of Standard vs RAG-Enhanced Results')
    plt.xticks(x, metric_labels)
    plt.legend()
    
    # Add value labels on top of bars
    for i, v in enumerate(standard_means):
        plt.text(i - width/2, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
    
    for i, v in enumerate(enhanced_means):
        plt.text(i + width/2, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()
    
    # 2. Percentage improvements pie chart
    plt.figure(figsize=(10, 10))
    
    # Data
    labels = ['Bounding Boxes', 'Detailed Analysis', 'Safety Info']
    sizes = [
        improvement_df['bounding_boxes_improved'].mean() * 100,
        improvement_df['detailed_analysis_improved'].mean() * 100,
        improvement_df['safety_info_improved'].mean() * 100
    ]
    
    # Plot
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Percentage of Images with Improvements')
    
    plt.savefig(os.path.join(output_dir, 'improvements_pie.png'))
    plt.close()
    
    # 3. Safety fields improvement histogram
    plt.figure(figsize=(10, 6))
    
    plt.hist(improvement_df['safety_fields_diff'], bins=range(min(improvement_df['safety_fields_diff'])-1, max(improvement_df['safety_fields_diff'])+2), alpha=0.7, color='green')
    plt.xlabel('Improvement in Safety Fields')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Safety Fields Improvement')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'safety_fields_histogram.png'))
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze RAG batch processing results')
    parser.add_argument('--results_path', type=str, required=True, help='Path to batch results CSV file')
    parser.add_argument('--output_dir', type=str, default='rag_analysis', help='Output directory for analysis')
    
    args = parser.parse_args()
    
    # Analyze results
    standard_df, enhanced_df, improvement_df, summary = analyze_batch_results(args.results_path, args.output_dir)
    
    # Print summary
    print("\nSummary of RAG Improvements:")
    print(f"Average tools detected: Standard = {summary['standard_tools_detected_avg']:.2f}, RAG = {summary['enhanced_tools_detected_avg']:.2f}")
    print(f"Average safety fields: Standard = {summary['standard_safety_fields_avg']:.2f}, RAG = {summary['enhanced_safety_fields_avg']:.2f}")
    print(f"Percentage with bounding boxes: Standard = {summary['standard_has_bounding_boxes_pct']:.1f}%, RAG = {summary['enhanced_has_bounding_boxes_pct']:.1f}%")
    print(f"Percentage with complete safety info: Standard = {summary['standard_complete_safety_info_pct']:.1f}%, RAG = {summary['enhanced_complete_safety_info_pct']:.1f}%")
    
    print(f"\nAverage improvement in tools detected: {summary['tools_detected_improvement_avg']:.2f}")
    print(f"Average improvement in safety fields: {summary['safety_fields_improvement_avg']:.2f}")
    print(f"Percentage of images with improved bounding boxes: {summary['bounding_boxes_improved_pct']:.1f}%")
    print(f"Percentage of images with improved safety info: {summary['safety_info_improved_pct']:.1f}%")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()