"""
Dataset Visualization Tool

This script generates a horizontal bar chart visualizing the distribution of 
tool categories across multiple datasets. It identifies underrepresented classes
and marks them accordingly in the visualization.

Author: akameswa, sselvam
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Dataset object category counts from analysis
data = {
    'Object Category': ['Wrench', 'Hammer', 'Pliers', 'Screwdriver', 'Bolt', 'Dynanometer', 'Tester', 'Tool Box', 'Tape measure', 'Ratchet', 'Drill', 'Calipers', 'Saw'],
    'Dataset 1': [1036, 434, 359, 1170, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Dataset 2': [3604, 193, 3541, 3556, 3303, 24, 17, 48, 0, 0, 0, 0, 4],
    'Dataset 3': [3012, 1284, 808, 3112, 0, 0, 0, 0, 199, 515, 75, 170, 96]
}

def create_distribution_visualization():
    """
    Creates and saves a horizontal bar chart showing the distribution of tools
    across different datasets.
    """
    # Create pandas DataFrame
    df = pd.DataFrame(data)
    df.set_index('Object Category', inplace=True)

    # Convert data to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Identify underrepresented classes (classes with count < 100 in all datasets)
    underrepresented_threshold = 100
    underrepresented_classes = df[(df['Dataset 1'] < underrepresented_threshold) & 
                                 (df['Dataset 2'] < underrepresented_threshold) & 
                                 (df['Dataset 3'] < underrepresented_threshold)].index

    # Define colors for each dataset
    dataset_colors = {
        'Dataset 1': '#1f77b4',  # muted blue
        'Dataset 2': '#2ca02c',  # vibrant green
        'Dataset 3': '#d62728'   # vivid red
    }

    # Plotting the stacked bar chart
    ax = df.plot(kind='barh', figsize=(12, 8), logx=True, stacked=True, 
                color=[dataset_colors[col] for col in df.columns])
    plt.title('Dataset Distribution of Object Categories')
    plt.ylabel('')  # Remove Object Category label
    plt.xlabel('Count (Log Scale)')
    plt.yticks(rotation=0)

    # Add legend with underrepresented classes
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label in underrepresented_classes:
            new_labels.append(f'{label} (Underrepresented)')
        else:
            new_labels.append(label)
    plt.legend(new_labels, title='Dataset')

    plt.tight_layout()

    # Save the figure
    plt.savefig('object_category_distribution.png')
    
    print("Bar chart generated and saved as 'object_category_distribution.png'")


if __name__ == "__main__":
    create_distribution_visualization()
