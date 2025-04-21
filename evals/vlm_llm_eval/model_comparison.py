import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Directories
RESULTS_DIR = "./evaluation_results"
PLOTS_DIR = "./evaluation_plots"
COMPARISON_DIR = "./comparison_plots"

# Create comparison directory if it doesn't exist
os.makedirs(COMPARISON_DIR, exist_ok=True)

def load_all_evaluation_results():
    """Load all evaluation CSV files and merge them into one DataFrame."""
    all_results = []
    
    # Find all evaluation CSV files
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "*_evaluation.csv"))
    
    for file_path in csv_files:
        # Extract model name from file name
        model_name = os.path.basename(file_path).replace('_evaluation.csv', '')
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Add model family (qwen or llama)
        if model_name.startswith('qwen'):
            df['model_family'] = 'Qwen'
        elif model_name.startswith('llama'):
            df['model_family'] = 'LLaMA'
        
        # Add fine-tuning type
        if model_name.endswith('-v'):
            df['fine_tuning'] = 'Vision'
        elif model_name.endswith('-l'):
            df['fine_tuning'] = 'Language'
        elif model_name.endswith('-vl'):
            df['fine_tuning'] = 'Vision+Language'
        else:
            df['fine_tuning'] = 'Zero-Shot'
        
        # Append to the list
        all_results.append(df)
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def create_model_comparison_bar_chart(combined_df):
    """Create a bar chart comparing all models."""
    # Calculate average scores by model
    metrics = ['function_score', 'safety_score', 'misuse_score', 'overall_score']
    metric_labels = ['Function', 'Safety', 'Misuses', 'Overall']
    
    # Group by model and calculate mean scores
    model_scores = combined_df.groupby('model')[metrics].mean().reset_index()
    
    # Reshape for plotting
    plot_data = []
    for _, row in model_scores.iterrows():
        model = row['model']
        # Determine model family and color
        if model.startswith('qwen'):
            model_family = 'Qwen'
        else:
            model_family = 'LLaMA'
            
        # Determine fine-tuning approach
        if model.endswith('-v'):
            fine_tuning = 'Vision'
        elif model.endswith('-l'):
            fine_tuning = 'Language'
        elif model.endswith('-vl'):
            fine_tuning = 'Vision+Language'
        else:
            fine_tuning = 'Zero-Shot'
            
        # Create a better display name
        display_name = f"{model_family} ({fine_tuning})"
        
        for i, metric in enumerate(metrics):
            plot_data.append({
                'Model': display_name,
                'Model Family': model_family,
                'Fine Tuning': fine_tuning,
                'Metric': metric_labels[i],
                'Score': row[metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=plot_df)
    
    # Customize the plot
    plt.title('Comparison of VLM Models for Tool Recognition', fontsize=18)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Average Score (0-10)', fontsize=14)
    plt.ylim(0, 10.5)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metric', title_fontsize=12, fontsize=10)
    
    # Save the plot
    plt.tight_layout()
    bar_chart_path = os.path.join(COMPARISON_DIR, 'all_models_comparison_bar.png')
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()
    
    return bar_chart_path

def create_model_family_comparison(combined_df):
    """Create a bar chart comparing Qwen vs LLaMA model families."""
    # Calculate average scores by model family
    metrics = ['function_score', 'safety_score', 'misuse_score', 'overall_score']
    metric_labels = ['Function', 'Safety', 'Misuses', 'Overall']
    
    # Group by model family and calculate mean scores
    family_scores = combined_df.groupby('model_family')[metrics].mean().reset_index()
    
    # Reshape for plotting
    plot_data = []
    for _, row in family_scores.iterrows():
        for i, metric in enumerate(metrics):
            plot_data.append({
                'Model Family': row['model_family'],
                'Metric': metric_labels[i],
                'Score': row[metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Metric', y='Score', hue='Model Family', data=plot_df, palette=['skyblue', 'salmon'])
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    # Customize the plot
    plt.title('Qwen vs LLaMA Model Families Comparison', fontsize=18)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Average Score (0-10)', fontsize=14)
    plt.ylim(0, 10.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Model Family', title_fontsize=12, fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    family_chart_path = os.path.join(COMPARISON_DIR, 'model_family_comparison.png')
    plt.savefig(family_chart_path, dpi=300)
    plt.close()
    
    return family_chart_path

def create_fine_tuning_comparison(combined_df):
    """Create a bar chart comparing different fine-tuning approaches."""
    # Calculate average scores by fine-tuning approach
    metrics = ['function_score', 'safety_score', 'misuse_score', 'overall_score']
    metric_labels = ['Function', 'Safety', 'Misuses', 'Overall']
    
    # Group by fine-tuning and calculate mean scores
    tuning_scores = combined_df.groupby('fine_tuning')[metrics].mean().reset_index()
    
    # Reshape for plotting
    plot_data = []
    for _, row in tuning_scores.iterrows():
        for i, metric in enumerate(metrics):
            plot_data.append({
                'Fine Tuning': row['fine_tuning'],
                'Metric': metric_labels[i],
                'Score': row[metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Sort fine-tuning approaches in a logical order
    order = ['Vision', 'Language', 'Vision+Language', 'Zero-Shot']
    plot_df['Fine Tuning'] = pd.Categorical(plot_df['Fine Tuning'], categories=order, ordered=True)
    plot_df = plot_df.sort_values('Fine Tuning')
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Fine Tuning', y='Score', hue='Metric', data=plot_df)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    # Customize the plot
    plt.title('Comparison of Fine-Tuning Approaches', fontsize=18)
    plt.xlabel('Fine-Tuning Approach', fontsize=14)
    plt.ylabel('Average Score (0-10)', fontsize=14)
    plt.ylim(0, 10.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metric', title_fontsize=12, fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    tuning_chart_path = os.path.join(COMPARISON_DIR, 'fine_tuning_comparison.png')
    plt.savefig(tuning_chart_path, dpi=300)
    plt.close()
    
    return tuning_chart_path

def create_heatmap_comparison(combined_df):
    """Create a heatmap comparing all models across metrics."""
    # Calculate average scores for each model and metric
    metrics = ['function_score', 'safety_score', 'misuse_score', 'overall_score']
    metric_labels = ['Function', 'Safety', 'Misuses', 'Overall']
    
    # Prepare data for the heatmap
    pivot_data = combined_df.groupby('model')[metrics].mean().reset_index()
    
    # Create better model names
    model_display_names = []
    for model in pivot_data['model']:
        if model.startswith('qwen'):
            family = 'Qwen'
        else:
            family = 'LLaMA'
            
        if model.endswith('-v'):
            tuning = 'Vision'
        elif model.endswith('-l'):
            tuning = 'Language'
        elif model.endswith('-vl'):
            tuning = 'Vision+Language'
        else:
            tuning = 'Zero-Shot'
            
        model_display_names.append(f"{family} ({tuning})")
    
    pivot_data['model_display'] = model_display_names
    
    # Create the pivot table
    heatmap_data = pd.DataFrame(index=model_display_names)
    for i, metric in enumerate(metrics):
        heatmap_data[metric_labels[i]] = pivot_data[metric].values
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', vmin=0, vmax=10, fmt='.2f')
    
    # Customize the plot
    plt.title('Heatmap of VLM Model Performance', fontsize=18)
    plt.xlabel('Evaluation Metric', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    
    # Save the plot
    plt.tight_layout()
    heatmap_path = os.path.join(COMPARISON_DIR, 'model_performance_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    return heatmap_path

def create_radar_chart(combined_df):
    """Create a radar chart comparing model families and fine-tuning approaches."""
    # Calculate average scores
    metrics = ['function_score', 'safety_score', 'misuse_score', 'overall_score']
    metric_labels = ['Function', 'Safety', 'Misuses', 'Overall']
    
    # Group by model family and calculate mean scores
    family_scores = combined_df.groupby('model_family')[metrics].mean()
    
    # Group by fine-tuning and calculate mean scores
    tuning_scores = combined_df.groupby('fine_tuning')[metrics].mean()
    
    # Set up the radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), subplot_kw=dict(polar=True))
    
    # Plot model families on the first radar chart
    for model_family in family_scores.index:
        values = family_scores.loc[model_family].values.tolist()
        values += values[:1]  # Close the loop
        
        color = 'skyblue' if model_family == 'Qwen' else 'salmon'
        ax1.plot(angles, values, 'o-', linewidth=2, label=model_family, color=color)
        ax1.fill(angles, values, alpha=0.1, color=color)
    
    # Set labels for the first chart
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metric_labels)
    ax1.set_yticks([2, 4, 6, 8, 10])
    ax1.set_ylim(0, 10)
    ax1.set_title('Model Family Comparison', fontsize=16)
    ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Plot fine-tuning approaches on the second radar chart
    colors = {'Vision': '#1f77b4', 'Language': '#ff7f0e', 'Vision+Language': '#2ca02c', 'Zero-Shot': '#d62728'}
    for tuning_type in tuning_scores.index:
        values = tuning_scores.loc[tuning_type].values.tolist()
        values += values[:1]  # Close the loop
        
        ax2.plot(angles, values, 'o-', linewidth=2, label=tuning_type, color=colors.get(tuning_type, 'gray'))
        ax2.fill(angles, values, alpha=0.1, color=colors.get(tuning_type, 'gray'))
    
    # Set labels for the second chart
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_labels)
    ax2.set_yticks([2, 4, 6, 8, 10])
    ax2.set_ylim(0, 10)
    ax2.set_title('Fine-Tuning Approach Comparison', fontsize=16)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save the plot
    plt.tight_layout()
    radar_path = os.path.join(COMPARISON_DIR, 'model_radar_comparison.png')
    plt.savefig(radar_path, dpi=300)
    plt.close()
    
    return radar_path

def create_summary_table(combined_df):
    """Create a summary table with key statistics."""
    # Overall statistics by model
    model_summary = combined_df.groupby(['model', 'model_family', 'fine_tuning'])['overall_score'].agg(['mean', 'count']).reset_index()
    model_summary.columns = ['Model', 'Model Family', 'Fine Tuning', 'Average Overall Score', 'Number of Samples']
    
    # Sort by average overall score (descending)
    model_summary = model_summary.sort_values('Average Overall Score', ascending=False)
    
    # Add ranking
    model_summary['Rank'] = range(1, len(model_summary) + 1)
    
    # Reorder columns
    model_summary = model_summary[['Rank', 'Model', 'Model Family', 'Fine Tuning', 'Average Overall Score', 'Number of Samples']]
    
    # Save the summary table
    summary_path = os.path.join(COMPARISON_DIR, 'model_ranking_summary.csv')
    model_summary.to_csv(summary_path, index=False)
    
    return summary_path, model_summary

def main():
    """Main function to create all comparison visualizations."""
    print("Loading evaluation results...")
    combined_df = load_all_evaluation_results()
    
    if len(combined_df) == 0:
        print("No evaluation results found. Please run the evaluation scripts first.")
        return
    
    print(f"Loaded {len(combined_df)} evaluation results from {combined_df['model'].nunique()} models.")
    
    print("\nCreating comparison visualizations...")
    bar_chart_path = create_model_comparison_bar_chart(combined_df)
    print(f"Bar chart saved to: {bar_chart_path}")
    
    family_chart_path = create_model_family_comparison(combined_df)
    print(f"Model family comparison saved to: {family_chart_path}")
    
    tuning_chart_path = create_fine_tuning_comparison(combined_df)
    print(f"Fine-tuning comparison saved to: {tuning_chart_path}")
    
    heatmap_path = create_heatmap_comparison(combined_df)
    print(f"Heatmap saved to: {heatmap_path}")
    
    radar_path = create_radar_chart(combined_df)
    print(f"Radar chart saved to: {radar_path}")
    
    summary_path, summary_df = create_summary_table(combined_df)
    print(f"Summary table saved to: {summary_path}")
    
    print("\nModel Ranking Summary:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 100)
    print(summary_df.to_string(index=False))
    
    print("\nAll comparison visualizations have been created successfully!")

if __name__ == "__main__":
    main()