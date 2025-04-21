#!/usr/bin/env python3
import os
import json
import pandas as pd
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_bbox_string(bbox_str):
    """Parse a bounding box string into a list of tool names"""
    tools = []
    
    try:
        # Clean the string
        if '\"\"' in bbox_str:
            bbox_str = bbox_str.replace('\"\"', '"')
        if bbox_str.startswith('"') and bbox_str.endswith('"'):
            bbox_str = bbox_str[1:-1]
        
        # Parse JSON
        bboxes = json.loads(bbox_str)
        
        # Extract tool names
        for box in bboxes:
            if 'tool' in box:
                tools.append(box['tool'].lower().strip())
    except Exception as e:
        logger.warning(f"Error parsing bounding box string: {e}")
    
    return tools

def create_tool_mapping(ground_truth_file, prediction_dir, output_file):
    """
    Create a tool name mapping between ground truth and predictions
    
    Args:
        ground_truth_file: Path to ground truth CSV file
        prediction_dir: Directory containing prediction CSV files
        output_file: Output JSON file for mapping
    """
    logger.info("Creating tool name mapping...")
    
    # Load ground truth tools
    gt_tools = set()
    try:
        gt_df = pd.read_csv(ground_truth_file)
        for _, row in gt_df.iterrows():
            tools = parse_bbox_string(row['bounding_boxes'])
            gt_tools.update(tools)
        
        logger.info(f"Found {len(gt_tools)} unique tools in ground truth")
    except Exception as e:
        logger.error(f"Error loading ground truth: {e}")
    
    # Load prediction tools
    pred_tools = set()
    pred_files = [f for f in os.listdir(prediction_dir) 
                 if f.endswith('.csv') and (f.startswith('qwen-') or f.startswith('llama-'))]
    
    for file in pred_files:
        try:
            pred_df = pd.read_csv(os.path.join(prediction_dir, file))
            for _, row in pred_df.iterrows():
                tools = parse_bbox_string(row['bounding_boxes'])
                pred_tools.update(tools)
        except Exception as e:
            logger.error(f"Error loading predictions from {file}: {e}")
    
    logger.info(f"Found {len(pred_tools)} unique tools in predictions")
    
    # Create mapping based on similarity
    mapping = {}
    
    # Standard tool categories
    standard_categories = {
        'screwdriver': ['screwdriver', 'driver', 'phillips'],
        'hammer': ['hammer', 'mallet', 'pein'],
        'wrench': ['wrench', 'spanner', 'ratchet'],
        'pliers': ['plier', 'tongs', 'pincer', 'nipper'],
        'cutters': ['cutter', 'snip', 'clipper'],
        'saw': ['saw', 'hacksaw', 'backsaw', 'handsaw'],
        'drill': ['drill', 'bit'],
        'tape measure': ['tape', 'measure', 'ruler'],
        'knife': ['knife', 'blade', 'utility knife'],
        'clamp': ['clamp', 'vise', 'vice'],
        'file': ['file', 'rasp'],
        'chisel': ['chisel'],
        'level': ['level', 'bubble'],
        'square': ['square'],
        'socket': ['socket']
    }
    
    # First, assign tools to standard categories
    for tool in gt_tools.union(pred_tools):
        assigned = False
        for category, keywords in standard_categories.items():
            if any(keyword in tool for keyword in keywords):
                mapping[tool] = category
                assigned = True
                break
        
        if not assigned:
            mapping[tool] = tool  # Keep as is if no category match
    
    # Try to find matches for remaining tools
    for pred_tool in pred_tools:
        if pred_tool in gt_tools:
            continue  # Already an exact match
        
        # Find the best matching ground truth tool based on substring matching
        best_match = None
        for gt_tool in gt_tools:
            if gt_tool in pred_tool or pred_tool in gt_tool:
                best_match = gt_tool
                break
        
        if best_match:
            # Map to same category as best match
            mapping[pred_tool] = mapping.get(best_match, best_match)
    
    # Save mapping to file
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    
    logger.info(f"Created tool mapping with {len(mapping)} entries")
    logger.info(f"Saved tool mapping to {output_file}")
    
    return mapping


def print_mapping_preview(mapping, max_items=20):
    """Print a preview of the mapping"""
    print("\nTool Mapping Preview:")
    print("-" * 50)
    print(f"{'Original Tool':<30} | {'Mapped To':<20}")
    print("-" * 50)
    
    for i, (original, mapped) in enumerate(sorted(mapping.items())):
        if i >= max_items:
            print(f"... and {len(mapping) - max_items} more items")
            break
        
        print(f"{original[:30]:<30} | {mapped[:20]:<20}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create tool name mapping between ground truth and predictions")
    parser.add_argument("--gt_file", required=True, help="Path to ground truth CSV file")
    parser.add_argument("--pred_dir", required=True, help="Directory containing prediction CSV files")
    parser.add_argument("--output_file", default="tool_mapping.json", help="Output JSON file for mapping")
    parser.add_argument("--preview", action="store_true", help="Print preview of the mapping")
    
    args = parser.parse_args()
    
    mapping = create_tool_mapping(
        ground_truth_file=args.gt_file,
        prediction_dir=args.pred_dir,
        output_file=args.output_file
    )
    
    if args.preview:
        print_mapping_preview(mapping)


if __name__ == "__main__":
    main()