#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Seaborn style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# Custom colors for visualization
COLOR_PALETTE = {
    'qwen-l_output': '#3274A1',    # Blue
    'qwen-v_output': '#E1812C',    # Orange
    'qwen-vl_output': '#3A923A',   # Green
    'qwen-Z_output': '#D84B40',    # Red
    'llama-l_output': '#7D54B6',   # Purple
    'llama-v_output': '#D1B343',   # Yellow
    'llama-vl_output': '#45A2B8',  # Cyan
    'llama-Z_output': '#8B6834'    # Brown
}

# Define model groupings
MODEL_GROUPS = {
    'qwen': ['qwen-l_output', 'qwen-v_output', 'qwen-vl_output', 'qwen-Z_output'],
    'llama': ['llama-l_output', 'llama-v_output', 'llama-vl_output', 'llama-Z_output']
}

# Define fine-tuning groups
FINE_TUNING_GROUPS = {
    'l': ['qwen-l_output', 'llama-l_output'],
    'v': ['qwen-v_output', 'llama-v_output'],
    'vl': ['qwen-vl_output', 'llama-vl_output'],
    'z': ['qwen-Z_output', 'llama-Z_output']
}

def generate_visualizations(metrics, output_dir):
    """
    Generate comprehensive visualizations for evaluation results
    
    Args:
        metrics: Dictionary of evaluation metrics for all models
        output_dir: Directory to save visualizations
    """
    logger.info("Generating visualizations")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all visualizations
    plot_overall_metrics(metrics, output_dir)
    plot_model_type_comparison(metrics, output_dir)
    plot_fine_tuning_comparison(metrics, output_dir)
    plot_per_tool_performance(metrics, output_dir)
    plot_f1_distributions(metrics, output_dir)
    plot_iou_distributions(metrics, output_dir)
    create_dashboard(metrics, output_dir)
    
    logger.info(f"All visualizations saved to {output_dir}")


def plot_overall_metrics(metrics, output_dir):
    """Plot overall metrics comparison for all models"""
    logger.info("Plotting overall metrics comparison")
    
    # 1. Bar chart of overall metrics
    plt.figure(figsize=(14, 8))
    
    # Prepare data
    models = list(metrics.keys())
    metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'IoU']
    
    data = []
    for model in models:
        data.append({
            'Model': model,
            'Precision': metrics[model]["overall"]["precision"],
            'Recall': metrics[model]["overall"]["recall"],
            'F1 Score': metrics[model]["overall"]["f1"],
            'IoU': metrics[model]["average"]["iou"]
        })
    
    # Convert to long format for seaborn
    df = pd.DataFrame(data)
    df_long = pd.melt(df, id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Value')
    
    # Create bar chart
    ax = sns.barplot(x='Model', y='Value', hue='Metric', data=df_long, palette='viridis')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8)
    
    plt.title('Overall Performance Metrics by Model', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title='Metric', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'overall_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved overall metrics plot to {output_path}")
    
    # 2. Heatmap of overall metrics
    plt.figure(figsize=(12, 8))
    
    # Create matrix for heatmap
    df_pivot = df.set_index('Model')
    
    # Create heatmap
    sns.heatmap(df_pivot, annot=True, fmt='.3f', cmap='viridis', linewidths=.5, vmin=0, vmax=1)
    
    plt.title('Performance Metrics Heatmap', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'metrics_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved metrics heatmap to {output_path}")


def plot_model_type_comparison(metrics, output_dir):
    """Plot comparison between model types (qwen vs llama)"""
    logger.info("Plotting model type comparison")
    
    # Calculate average metrics per model type
    model_type_metrics = {}
    
    for model_type, models in MODEL_GROUPS.items():
        # Filter models that exist in our data
        available_models = [m for m in models if m in metrics]
        
        if not available_models:
            continue
        
        # Calculate average metrics
        precision = np.mean([metrics[m]["overall"]["precision"] for m in available_models])
        recall = np.mean([metrics[m]["overall"]["recall"] for m in available_models])
        f1 = np.mean([metrics[m]["overall"]["f1"] for m in available_models])
        iou = np.mean([metrics[m]["average"]["iou"] for m in available_models])
        tool_acc = np.mean([metrics[m]["overall"]["tool_accuracy"] for m in available_models])
        bbox_acc = np.mean([metrics[m]["overall"]["bbox_accuracy"] for m in available_models])
        
        model_type_metrics[model_type] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'IoU': iou,
            'Tool Accuracy': tool_acc,
            'Bbox Accuracy': bbox_acc
        }
    
    # Bar chart comparison
    plt.figure(figsize=(12, 8))
    
    # Convert to DataFrame for plotting
    data = []
    for model_type, model_metrics in model_type_metrics.items():
        for metric_name, value in model_metrics.items():
            data.append({
                'Model Type': model_type.capitalize(),
                'Metric': metric_name,
                'Value': value
            })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart
    ax = sns.barplot(x='Model Type', y='Value', hue='Metric', data=df, palette='viridis')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8)
    
    plt.title('Model Type Comparison (Average Metrics)', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Model Type', fontsize=14)
    plt.ylim(0, 1.1)
    plt.legend(title='Metric', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'model_type_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved model type comparison to {output_path}")


def plot_fine_tuning_comparison(metrics, output_dir):
    """Plot comparison between fine-tuning strategies"""
    logger.info("Plotting fine-tuning strategy comparison")
    
    # Calculate average metrics per fine-tuning strategy
    ft_metrics = {}
    
    for ft_type, models in FINE_TUNING_GROUPS.items():
        # Filter models that exist in our data
        available_models = [m for m in models if m in metrics]
        
        if not available_models:
            continue
        
        # Calculate average metrics
        precision = np.mean([metrics[m]["overall"]["precision"] for m in available_models])
        recall = np.mean([metrics[m]["overall"]["recall"] for m in available_models])
        f1 = np.mean([metrics[m]["overall"]["f1"] for m in available_models])
        iou = np.mean([metrics[m]["average"]["iou"] for m in available_models])
        tool_acc = np.mean([metrics[m]["overall"]["tool_accuracy"] for m in available_models])
        bbox_acc = np.mean([metrics[m]["overall"]["bbox_accuracy"] for m in available_models])
        
        ft_metrics[ft_type] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'IoU': iou,
            'Tool Accuracy': tool_acc,
            'Bbox Accuracy': bbox_acc
        }
    
    # Bar chart comparison
    plt.figure(figsize=(14, 8))
    
    # Convert to DataFrame for plotting
    data = []
    for ft_type, ft_metrics_values in ft_metrics.items():
        # Map fine-tuning codes to readable names
        ft_name = {
            'l': 'Language Only',
            'v': 'Vision Only',
            'vl': 'Vision+Language',
            'z': 'Zero-shot'
        }.get(ft_type, ft_type)
        
        for metric_name, value in ft_metrics_values.items():
            data.append({
                'Fine-tuning': ft_name,
                'Metric': metric_name,
                'Value': value
            })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart
    ax = sns.barplot(x='Fine-tuning', y='Value', hue='Metric', data=df, palette='viridis')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8)
    
    plt.title('Fine-tuning Strategy Comparison (Average Metrics)', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Fine-tuning Strategy', fontsize=14)
    plt.ylim(0, 1.1)
    plt.legend(title='Metric', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'fine_tuning_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved fine-tuning comparison to {output_path}")


def plot_per_tool_performance(metrics, output_dir):
    """Plot performance metrics for top tools across models"""
    logger.info("Plotting per-tool performance")
    
    # Get all tools from all models
    all_tools = set()
    for model_metrics in metrics.values():
        all_tools.update(model_metrics["per_tool"].keys())
    
    # Get frequency of each tool in ground truth
    tool_counts = {}
    for tool in all_tools:
        counts = []
        for model_name, model_metrics in metrics.items():
            if tool in model_metrics["per_tool"]:
                count = model_metrics["per_tool"][tool].get("count", 0)
                counts.append(count)
        
        # Use the maximum count
        tool_counts[tool] = max(counts) if counts else 0
    
    # Get top 15 most frequent tools
    top_tools = sorted(all_tools, key=lambda t: tool_counts.get(t, 0), reverse=True)[:15]
    
    # F1 score heatmap for top tools across models
    plt.figure(figsize=(16, 10))
    
    # Create data matrix for heatmap
    heatmap_data = []
    for tool in top_tools:
        row = []
        for model in metrics:
            if tool in metrics[model]["per_tool"]:
                f1 = metrics[model]["per_tool"][tool].get("f1", 0)
            else:
                f1 = 0
            row.append(f1)
        heatmap_data.append(row)
    
    # Convert to numpy array
    heatmap_array = np.array(heatmap_data)
    
    # Create heatmap
    ax = plt.gca()
    im = ax.imshow(heatmap_array, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('F1 Score', fontsize=12)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(top_tools)))
    ax.set_xticklabels(list(metrics.keys()), fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels([f"{t} (n={tool_counts[t]})" for t in top_tools], fontsize=10)
    
    # Add text annotations with F1 scores
    for i in range(len(top_tools)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f"{heatmap_array[i, j]:.2f}",
                          ha="center", va="center", color="w" if heatmap_array[i, j] < 0.5 else "k",
                          fontsize=9)
    
    plt.title('F1 Score by Tool and Model', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Tool', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'per_tool_f1_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved per-tool F1 heatmap to {output_path}")


def plot_f1_distributions(metrics, output_dir):
    """Plot F1 score distributions across models"""
    logger.info("Plotting F1 score distributions")
    
    # Violin plots of F1 distributions
    plt.figure(figsize=(14, 8))
    
    # Collect F1 scores from all models
    f1_data = []
    for model_name, model_metrics in metrics.items():
        f1_scores = model_metrics["per_image"]["f1"]
        for score in f1_scores:
            f1_data.append({
                'Model': model_name,
                'F1 Score': score
            })
    
    # Create DataFrame
    df = pd.DataFrame(f1_data)
    
    # Create violin plot
    ax = sns.violinplot(x='Model', y='F1 Score', data=df, 
                       palette=COLOR_PALETTE, inner='quartile')
    
    # Add median lines
    for model in metrics:
        scores = metrics[model]["per_image"]["f1"]
        if scores:
            median = np.median(scores)
            mean = np.mean(scores)
            plt.text(list(metrics.keys()).index(model), 0.05, 
                     f'Median: {median:.2f}\nMean: {mean:.2f}', 
                     ha='center', va='bottom', fontsize=9)
    
    plt.title('F1 Score Distribution by Model', fontsize=16)
    plt.ylabel('F1 Score', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'f1_distribution_violin.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved F1 distribution violin plot to {output_path}")


def plot_iou_distributions(metrics, output_dir):
    """Plot IoU distributions across models"""
    logger.info("Plotting IoU distributions")
    
    # Violin plots of IoU distributions
    plt.figure(figsize=(14, 8))
    
    # Collect IoU scores from all models
    iou_data = []
    for model_name, model_metrics in metrics.items():
        iou_scores = model_metrics["per_image"]["iou"]
        for score in iou_scores:
            iou_data.append({
                'Model': model_name,
                'IoU': score
            })
    
    # Create DataFrame
    df = pd.DataFrame(iou_data)
    
    # Create violin plot
    ax = sns.violinplot(x='Model', y='IoU', data=df, 
                       palette=COLOR_PALETTE, inner='quartile')
    
    # Add median lines
    for model in metrics:
        scores = metrics[model]["per_image"]["iou"]
        if scores:
            median = np.median(scores)
            mean = np.mean(scores)
            plt.text(list(metrics.keys()).index(model), 0.05, 
                     f'Median: {median:.2f}\nMean: {mean:.2f}', 
                     ha='center', va='bottom', fontsize=9)
    
    plt.title('IoU Distribution by Model', fontsize=16)
    plt.ylabel('IoU', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'iou_distribution_violin.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved IoU distribution violin plot to {output_path}")


def create_dashboard(metrics, output_dir):
    """Create a comprehensive dashboard with key metrics and visualizations"""
    logger.info("Creating metrics dashboard")
    
    # Set up figure
    fig = plt.figure(figsize=(24, 20))
    grid = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.2])
    
    # 1. Overall metrics bar chart
    ax1 = fig.add_subplot(grid[0, 0])
    
    # Prepare data
    models = list(metrics.keys())
    metrics_to_plot = ['Precision', 'Recall', 'F1 Score']
    
    data = []
    for model in models:
        data.append({
            'Model': model,
            'Precision': metrics[model]["overall"]["precision"],
            'Recall': metrics[model]["overall"]["recall"],
            'F1 Score': metrics[model]["overall"]["f1"]
        })
    
    # Convert to long format for seaborn
    df = pd.DataFrame(data)
    df_long = pd.melt(df, id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Value')
    
    # Create bar chart
    sns.barplot(x='Model', y='Value', hue='Metric', data=df_long, palette='viridis', ax=ax1)
    
    ax1.set_title('Overall Performance Metrics by Model', fontsize=14)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax1.legend(title='Metric', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. F1 Score violin plot
    ax2 = fig.add_subplot(grid[0, 1])
    
    # Collect F1 scores from all models
    f1_data = []
    for model_name, model_metrics in metrics.items():
        f1_scores = model_metrics["per_image"]["f1"]
        for score in f1_scores:
            f1_data.append({
                'Model': model_name,
                'F1 Score': score
            })
    
    # Create DataFrame
    df = pd.DataFrame(f1_data)
    
    # Create violin plot
    sns.violinplot(x='Model', y='F1 Score', data=df, 
                 palette=COLOR_PALETTE, inner='quartile', ax=ax2)
    
    ax2.set_title('F1 Score Distribution by Model', fontsize=14)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Model type comparison
    ax3 = fig.add_subplot(grid[1, 0])
    
    # Calculate average metrics per model type
    model_type_metrics = {}
    for model_type, models_list in MODEL_GROUPS.items():
        # Filter models that exist in our data
        available_models = [m for m in models_list if m in metrics]
        
        if not available_models:
            continue
        
        # Calculate average metrics
        precision = np.mean([metrics[m]["overall"]["precision"] for m in available_models])
        recall = np.mean([metrics[m]["overall"]["recall"] for m in available_models])
        f1 = np.mean([metrics[m]["overall"]["f1"] for m in available_models])
        iou = np.mean([metrics[m]["average"]["iou"] for m in available_models])
        
        model_type_metrics[model_type] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'IoU': iou
        }
    
    # Convert to DataFrame for plotting
    data = []
    for model_type, mt_metrics in model_type_metrics.items():
        for metric_name, value in mt_metrics.items():
            data.append({
                'Model Type': model_type.capitalize(),
                'Metric': metric_name,
                'Value': value
            })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart
    sns.barplot(x='Model Type', y='Value', hue='Metric', data=df, palette='viridis', ax=ax3)
    
    ax3.set_title('Model Type Comparison (Average Metrics)', fontsize=14)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_xlabel('Model Type', fontsize=12)
    ax3.set_ylim(0, 1.0)
    ax3.legend(title='Metric', fontsize=10)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Fine-tuning strategy comparison
    ax4 = fig.add_subplot(grid[1, 1])
    
    # Calculate average metrics per fine-tuning strategy
    ft_metrics = {}
    for ft_type, models_list in FINE_TUNING_GROUPS.items():
        # Filter models that exist in our data
        available_models = [m for m in models_list if m in metrics]
        
        if not available_models:
            continue
        
        # Calculate average metrics
        precision = np.mean([metrics[m]["overall"]["precision"] for m in available_models])
        recall = np.mean([metrics[m]["overall"]["recall"] for m in available_models])
        f1 = np.mean([metrics[m]["overall"]["f1"] for m in available_models])
        iou = np.mean([metrics[m]["average"]["iou"] for m in available_models])
        
        ft_metrics[ft_type] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'IoU': iou
        }
    
    # Map fine-tuning codes to readable names
    ft_names = {
        'l': 'Language Only',
        'v': 'Vision Only',
        'vl': 'Vision+Language',
        'z': 'Zero-shot'
    }
    
    # Convert to DataFrame for plotting
    data = []
    for ft_type, ft_metrics_values in ft_metrics.items():
        # Get readable name
        ft_name = ft_names.get(ft_type, ft_type)
        
        for metric_name, value in ft_metrics_values.items():
            data.append({
                'Fine-tuning': ft_name,
                'Metric': metric_name,
                'Value': value
            })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart
    sns.barplot(x='Fine-tuning', y='Value', hue='Metric', data=df, palette='viridis', ax=ax4)
    
    ax4.set_title('Fine-tuning Strategy Comparison (Average Metrics)', fontsize=14)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_xlabel('Fine-tuning Strategy', fontsize=12)
    ax4.set_ylim(0, 1.0)
    ax4.legend(title='Metric', fontsize=10)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 5. Per-tool performance heatmap
    ax5 = fig.add_subplot(grid[2, :])
    
    # Get all tools from all models
    all_tools = set()
    for model_metrics in metrics.values():
        all_tools.update(model_metrics["per_tool"].keys())
    
    # Get frequency of each tool in ground truth
    tool_counts = {}
    for tool in all_tools:
        counts = []
        for model_name, model_metrics in metrics.items():
            if tool in model_metrics["per_tool"]:
                count = model_metrics["per_tool"][tool].get("count", 0)
                counts.append(count)
        
        # Use the maximum count
        tool_counts[tool] = max(counts) if counts else 0
    
    # Get top 10 most frequent tools
    top_tools = sorted(all_tools, key=lambda t: tool_counts.get(t, 0), reverse=True)[:10]
    
    # Create data matrix for heatmap
    heatmap_data = []
    for tool in top_tools:
        row = []
        for model in metrics:
            if tool in metrics[model]["per_tool"]:
                f1 = metrics[model]["per_tool"][tool].get("f1", 0)
            else:
                f1 = 0
            row.append(f1)
        heatmap_data.append(row)
    
    # Convert to numpy array
    heatmap_array = np.array(heatmap_data)
    
    # Create heatmap
    im = ax5.imshow(heatmap_array, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('F1 Score', fontsize=12)
    
    # Set tick labels
    ax5.set_xticks(np.arange(len(metrics)))
    ax5.set_yticks(np.arange(len(top_tools)))
    ax5.set_xticklabels(list(metrics.keys()), fontsize=10, rotation=45, ha='right')
    ax5.set_yticklabels([f"{t} (n={tool_counts[t]})" for t in top_tools], fontsize=10)
    
    # Add text annotations with F1 scores
    for i in range(len(top_tools)):
        for j in range(len(metrics)):
            text = ax5.text(j, i, f"{heatmap_array[i, j]:.2f}",
                          ha="center", va="center", color="w" if heatmap_array[i, j] < 0.5 else "k",
                          fontsize=9)
    
    ax5.set_title('F1 Score by Tool and Model (Top 10 Tools)', fontsize=14)
    ax5.set_xlabel('Model', fontsize=12)
    ax5.set_ylabel('Tool', fontsize=12)
    
    # Add title to the figure
    fig.suptitle('VLM Tool Recognition Evaluation Dashboard', fontsize=20, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    output_path = os.path.join(output_dir, 'evaluation_dashboard.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved evaluation dashboard to {output_path}")