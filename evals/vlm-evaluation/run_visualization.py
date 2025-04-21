#!/usr/bin/env python3

import os
import sys
import argparse

def main():
    """Run the tool detection evaluation and visualization"""
    parser = argparse.ArgumentParser(description="Run tool detection evaluation and visualization")
    parser.add_argument("--gt_file", default="./results/hf_ground_truth.json", 
                      help="Path to ground truth JSON file (default: ./results/hf_ground_truth.json)")
    parser.add_argument("--pred_dir", default="../../evals/results", 
                      help="Directory containing prediction CSV files (default: ../../evals/results)")
    parser.add_argument("--output_dir", default="./results/visualizations", 
                      help="Directory to save visualization results (default: ./results/visualizations)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ensure the ground truth file exists
    if not os.path.exists(args.gt_file):
        print(f"Error: Ground truth file not found at {args.gt_file}")
        print("Please run the ground truth extraction script first:")
        print("python hf-ground-truth-extraction.py --prediction_dir PATH_TO_PREDICTIONS --output_dir ./results")
        return 1
    
    # Ensure prediction directory exists
    if not os.path.exists(args.pred_dir):
        print(f"Error: Prediction directory not found at {args.pred_dir}")
        return 1
    
    # Run the visualization script
    from metrics_visualization import ToolDetectionEvaluator
    
    print(f"Starting tool detection evaluation and visualization...")
    print(f"- Ground truth file: {args.gt_file}")
    print(f"- Prediction directory: {args.pred_dir}")
    print(f"- Output directory: {args.output_dir}")
    
    # Create evaluator
    evaluator = ToolDetectionEvaluator(
        ground_truth_file=args.gt_file,
        prediction_dir=args.pred_dir,
        output_dir=args.output_dir
    )
    
    # Generate visualizations
    evaluator.generate_visualizations()
    
    # Export metrics
    evaluator.export_metrics()
    
    print(f"\nVisualization complete! Results saved to {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())