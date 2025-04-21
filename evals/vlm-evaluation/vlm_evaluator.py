#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
import argparse

# Import visualization module
from visualization import generate_visualizations

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToolDetectionEvaluator:
    """Class for evaluating tool detection models"""
    
    def __init__(self, ground_truth_file, prediction_dir, output_dir, tool_mapping_file=None):
        """
        Initialize the evaluator
        
        Args:
            ground_truth_file: Path to ground truth CSV file
            prediction_dir: Directory containing prediction CSV files
            output_dir: Directory to save results
            tool_mapping_file: Optional path to tool name mapping JSON file
        """
        self.ground_truth_file = ground_truth_file
        self.prediction_dir = prediction_dir
        self.output_dir = output_dir
        self.tool_mapping_file = tool_mapping_file
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load tool name mapping if provided
        self.tool_mapping = {}
        if tool_mapping_file and os.path.exists(tool_mapping_file):
            try:
                with open(tool_mapping_file, 'r') as f:
                    self.tool_mapping = json.load(f)
                logger.info(f"Loaded tool mapping from {tool_mapping_file}")
            except Exception as e:
                logger.error(f"Error loading tool mapping: {e}")
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth()
        
        # Load predictions
        self.predictions = self._load_predictions()
        
        # Calculate metrics
        self.metrics = self._calculate_metrics()
    
    def _load_ground_truth(self):
        """Load ground truth from CSV file"""
        logger.info(f"Loading ground truth from {self.ground_truth_file}")
        
        try:
            gt_df = pd.read_csv(self.ground_truth_file)
            ground_truth = {}
            
            for _, row in gt_df.iterrows():
                filename = row['filename']
                
                try:
                    # Clean and parse the JSON string
                    bbox_str = row['bounding_boxes']
                    if '\"\"' in bbox_str:
                        bbox_str = bbox_str.replace('\"\"', '"')
                    if bbox_str.startswith('"') and bbox_str.endswith('"'):
                        bbox_str = bbox_str[1:-1]
                    
                    bboxes = json.loads(bbox_str)
                    
                    # Extract tools and bounding boxes
                    tools = [box['tool'] for box in bboxes]
                    coords = [box['bbox'] for box in bboxes]
                    
                    # Store in ground truth dictionary
                    ground_truth[filename] = {
                        "tools": tools,
                        "bboxes": coords
                    }
                    
                except Exception as e:
                    logger.error(f"Error parsing bounding boxes for {filename}: {e}")
            
            logger.info(f"Loaded ground truth for {len(ground_truth)} images")
            return ground_truth
        
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            return {}
    
    def _load_predictions(self):
        """Load predictions from CSV files"""
        logger.info(f"Loading predictions from {self.prediction_dir}")
        
        predictions = {}
        
        # Find all prediction CSV files
        prediction_files = [
            os.path.join(self.prediction_dir, f) 
            for f in os.listdir(self.prediction_dir) 
            if f.endswith('.csv') and (f.startswith('qwen-') or f.startswith('llama-'))
        ]
        
        # Process each prediction file
        for pred_file in prediction_files:
            model_name = os.path.basename(pred_file).replace('.csv', '')
            
            try:
                pred_df = pd.read_csv(pred_file)
                predictions_list = []
                
                for _, row in pred_df.iterrows():
                    filename = row['filename']
                    
                    try:
                        # Clean and parse the JSON string
                        bbox_str = row['bounding_boxes']
                        if '\"\"' in bbox_str:
                            bbox_str = bbox_str.replace('\"\"', '"')
                        if bbox_str.startswith('"') and bbox_str.endswith('"'):
                            bbox_str = bbox_str[1:-1]
                        
                        bboxes = json.loads(bbox_str)
                        
                        # Extract tools and bounding boxes
                        tools = [box['tool'] for box in bboxes]
                        coords = [box['bbox'] for box in bboxes]
                        
                        # Store prediction
                        predictions_list.append({
                            'filename': filename,
                            'tools': tools,
                            'bboxes': coords
                        })
                        
                    except Exception as e:
                        logger.error(f"Error parsing bounding boxes for {filename} in {model_name}")
                        
                        # Add empty prediction
                        predictions_list.append({
                            'filename': filename,
                            'tools': [],
                            'bboxes': []
                        })
                
                # Store predictions for this model
                predictions[model_name] = predictions_list
                
                logger.info(f"Loaded {len(predictions_list)} predictions for model {model_name}")
            
            except Exception as e:
                logger.error(f"Error loading predictions for {model_name}: {e}")
        
        return predictions
    
    def _normalize_tool_name(self, tool_name):
        """Normalize tool name using mapping or default normalization"""
        tool_lower = tool_name.lower().strip()
        
        # Apply mapping if available
        if self.tool_mapping and tool_lower in self.tool_mapping:
            return self.tool_mapping[tool_lower]
        
        # Default normalization - simplify to basic tool types
        if 'screwdriver' in tool_lower:
            return 'screwdriver'
        elif 'hammer' in tool_lower:
            return 'hammer'
        elif 'wrench' in tool_lower or 'spanner' in tool_lower:
            return 'wrench'
        elif 'plier' in tool_lower:
            return 'pliers'
        elif 'cutter' in tool_lower:
            return 'cutter'
        elif 'saw' in tool_lower:
            return 'saw'
        elif 'drill' in tool_lower:
            return 'drill'
        elif 'tape' in tool_lower and 'measure' in tool_lower:
            return 'tape measure'
        elif 'knife' in tool_lower:
            return 'knife'
        else:
            return tool_lower
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union for two bounding boxes
        
        Args:
            box1: First box [x1, y1, x2, y2] or [x, y, w, h]
            box2: Second box [x1, y1, x2, y2] or [x, y, w, h]
        
        Returns:
            IoU value between 0 and 1
        """
        # Handle different box formats and ensure they have 4 elements
        if not isinstance(box1, list) or not isinstance(box2, list):
            return 0
        
        if len(box1) != 4 or len(box2) != 4:
            return 0
        
        # Ensure boxes are lists or arrays of numbers
        if not all(isinstance(x, (int, float)) for x in box1 + box2):
            return 0
        
        # Convert [x, y, w, h] format to [x1, y1, x2, y2] if needed
        # We detect this if the third value (width) is small compared to the first value (x)
        # or if the fourth value (height) is small compared to the second value (y)
        def ensure_xyxy_format(box):
            if len(box) != 4:
                return box  # Can't convert
            
            x1, y1, x2, y2 = box
            
            # Check if likely in [x, y, w, h] format
            if (x2 < x1 or y2 < y1) or (x2 < 0.5 * x1 or y2 < 0.5 * y1):
                # Convert [x, y, w, h] to [x1, y1, x2, y2]
                return [x1, y1, x1 + x2, y1 + y2]
            return box
        
        box1 = ensure_xyxy_format(box1)
        box2 = ensure_xyxy_format(box2)
        
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate area of each box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate coordinates of intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Calculate area of intersection
        width_i = max(0, x2_i - x1_i)
        height_i = max(0, y2_i - y1_i)
        area_i = width_i * height_i
        
        # Calculate area of union
        area_u = area1 + area2 - area_i
        
        # Calculate IoU
        iou = area_i / area_u if area_u > 0 else 0
        
        return iou
    
    def _calculate_metrics(self):
        """Calculate metrics for each model"""
        logger.info("Calculating metrics")
        
        metrics = {}
        
        # Process each model
        for model_name, predictions_list in self.predictions.items():
            logger.info(f"Processing model: {model_name}")
            
            # Initialize metrics for this model
            model_metrics = {
                "per_image": {
                    "precision": [],
                    "recall": [],
                    "f1": [],
                    "iou": []
                },
                "per_tool": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
                "confusion": {
                    "true_tools": [],
                    "pred_tools": []
                },
                "summary": {
                    "total_gt_tools": 0,
                    "total_pred_tools": 0,
                    "correct_tools": 0,
                    "correct_bboxes": 0,
                    "total_images": 0
                }
            }
            
            # Create lookup dictionary for predictions based on filename
            pred_lookup = {p['filename']: p for p in predictions_list}
            
            # For each image in ground truth
            for filename, gt_data in self.ground_truth.items():
                gt_tools = gt_data["tools"]
                gt_bboxes = gt_data["bboxes"]
                
                # Skip if no prediction for this image
                if filename not in pred_lookup:
                    continue
                
                # Get predictions for this image
                pred_data = pred_lookup[filename]
                pred_tools = pred_data["tools"]
                pred_bboxes = pred_data["bboxes"]
                
                # Update summary counts
                model_metrics["summary"]["total_gt_tools"] += len(gt_tools)
                model_metrics["summary"]["total_pred_tools"] += len(pred_tools)
                model_metrics["summary"]["total_images"] += 1
                
                # Normalize tool names for comparison
                gt_tools_norm = [self._normalize_tool_name(t) for t in gt_tools]
                pred_tools_norm = [self._normalize_tool_name(t) for t in pred_tools]
                
                # Update confusion matrix data
                model_metrics["confusion"]["true_tools"].extend(gt_tools_norm)
                model_metrics["confusion"]["pred_tools"].extend(pred_tools_norm)
                
                # Calculate precision, recall, F1 for tool detection
                if not gt_tools_norm and not pred_tools_norm:
                    # No tools in ground truth or predicted
                    precision = 1.0
                    recall = 1.0
                    f1 = 1.0
                elif not gt_tools_norm:
                    # No tools in ground truth, but tools predicted (all false positives)
                    precision = 0.0
                    recall = 1.0
                    f1 = 0.0
                    
                    # Update per-tool metrics (false positives)
                    for tool in pred_tools_norm:
                        model_metrics["per_tool"][tool]["fp"] += 1
                    
                elif not pred_tools_norm:
                    # Tools in ground truth, but none predicted (all false negatives)
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                    
                    # Update per-tool metrics (false negatives)
                    for tool in gt_tools_norm:
                        model_metrics["per_tool"][tool]["fn"] += 1
                
                else:
                    # Calculate based on normalized tool names
                    true_positives = 0
                    matched_pred_indices = set()
                    matched_gt_indices = set()
                    
                    # For each ground truth tool, find the best matching prediction
                    for gt_idx, gt_tool in enumerate(gt_tools_norm):
                        best_match_idx = None
                        best_match_iou = 0
                        
                        for pred_idx, pred_tool in enumerate(pred_tools_norm):
                            if pred_idx in matched_pred_indices:
                                continue
                            
                            # Tool name match
                            if gt_tool == pred_tool:
                                # Calculate IoU if bounding boxes are available
                                if gt_idx < len(gt_bboxes) and pred_idx < len(pred_bboxes):
                                    iou = self._calculate_iou(gt_bboxes[gt_idx], pred_bboxes[pred_idx])
                                    
                                    # If better match than current best
                                    if iou > best_match_iou:
                                        best_match_idx = pred_idx
                                        best_match_iou = iou
                                else:
                                    # No bounding boxes, just use tool name match
                                    best_match_idx = pred_idx
                                    best_match_iou = 1.0
                                    break
                        
                        # If found a match
                        if best_match_idx is not None:
                            true_positives += 1
                            matched_pred_indices.add(best_match_idx)
                            matched_gt_indices.add(gt_idx)
                            
                            # Update per-tool metrics
                            model_metrics["per_tool"][gt_tool]["tp"] += 1
                            
                            # Count correct bounding boxes (IoU > 0.5)
                            if best_match_iou > 0.5:
                                model_metrics["summary"]["correct_bboxes"] += 1
                    
                    # Count correct tools
                    model_metrics["summary"]["correct_tools"] += true_positives
                    
                    # False positives: predicted tools that weren't matched
                    for pred_idx, pred_tool in enumerate(pred_tools_norm):
                        if pred_idx not in matched_pred_indices:
                            model_metrics["per_tool"][pred_tool]["fp"] += 1
                    
                    # False negatives: ground truth tools that weren't matched
                    for gt_idx, gt_tool in enumerate(gt_tools_norm):
                        if gt_idx not in matched_gt_indices:
                            model_metrics["per_tool"][gt_tool]["fn"] += 1
                    
                    # Calculate metrics
                    precision = true_positives / len(pred_tools_norm) if pred_tools_norm else 0
                    recall = true_positives / len(gt_tools_norm) if gt_tools_norm else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Calculate IoU for matched bounding boxes
                iou_values = []
                
                for gt_idx in matched_gt_indices:
                    for pred_idx in matched_pred_indices:
                        if gt_idx < len(gt_bboxes) and pred_idx < len(pred_bboxes):
                            iou = self._calculate_iou(gt_bboxes[gt_idx], pred_bboxes[pred_idx])
                            if iou > 0:  # Only include positive IoU values
                                iou_values.append(iou)
                
                # Add metrics for this image
                model_metrics["per_image"]["precision"].append(precision)
                model_metrics["per_image"]["recall"].append(recall)
                model_metrics["per_image"]["f1"].append(f1)
                if iou_values:
                    model_metrics["per_image"]["iou"].append(np.mean(iou_values))
            
            # Calculate average metrics
            avg_precision = np.mean(model_metrics["per_image"]["precision"]) if model_metrics["per_image"]["precision"] else 0
            avg_recall = np.mean(model_metrics["per_image"]["recall"]) if model_metrics["per_image"]["recall"] else 0
            avg_f1 = np.mean(model_metrics["per_image"]["f1"]) if model_metrics["per_image"]["f1"] else 0
            avg_iou = np.mean(model_metrics["per_image"]["iou"]) if model_metrics["per_image"]["iou"] else 0
            
            # Calculate overall metrics
            total_gt = model_metrics["summary"]["total_gt_tools"]
            total_pred = model_metrics["summary"]["total_pred_tools"]
            total_correct = model_metrics["summary"]["correct_tools"]
            total_correct_bbox = model_metrics["summary"]["correct_bboxes"]
            
            overall_precision = total_correct / total_pred if total_pred > 0 else 0
            overall_recall = total_correct / total_gt if total_gt > 0 else 0
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
            
            # Calculate per-tool metrics
            for tool, counts in model_metrics["per_tool"].items():
                tp = counts["tp"]
                fp = counts["fp"]
                fn = counts["fn"]
                
                # Calculate precision, recall, F1 for this tool
                tool_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                tool_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                tool_f1 = 2 * tool_precision * tool_recall / (tool_precision + tool_recall) if (tool_precision + tool_recall) > 0 else 0
                
                # Update per-tool metrics
                model_metrics["per_tool"][tool].update({
                    "precision": tool_precision,
                    "recall": tool_recall,
                    "f1": tool_f1,
                    "count": tp + fn  # Total ground truth count for this tool
                })
            
            # Store model metrics
            metrics[model_name] = {
                "average": {
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "f1": avg_f1,
                    "iou": avg_iou
                },
                "overall": {
                    "precision": overall_precision,
                    "recall": overall_recall,
                    "f1": overall_f1,
                    "tool_accuracy": total_correct / total_gt if total_gt > 0 else 0,
                    "bbox_accuracy": total_correct_bbox / total_gt if total_gt > 0 else 0
                },
                "per_image": model_metrics["per_image"],
                "per_tool": dict(model_metrics["per_tool"]),
                "confusion": model_metrics["confusion"],
                "summary": model_metrics["summary"]
            }
            
            logger.info(f"Model {model_name}:")
            logger.info(f"  Overall: Precision={overall_precision:.3f}, Recall={overall_recall:.3f}, F1={overall_f1:.3f}")
        
        return metrics
    
    def generate_summary_table(self, output_dir=None):
        """Generate summary table with metrics for all models"""
        if output_dir is None:
            output_dir = self.output_dir
            
        logger.info("Generating summary metrics table")
        
        # Prepare data for overall metrics
        overall_data = []
        for model_name, model_metrics in self.metrics.items():
            overall_data.append({
                "Model": model_name,
                "Precision": model_metrics["overall"]["precision"],
                "Recall": model_metrics["overall"]["recall"],
                "F1 Score": model_metrics["overall"]["f1"],
                "IoU": model_metrics["average"]["iou"],
                "Tool Accuracy": model_metrics["overall"]["tool_accuracy"],
                "Bbox Accuracy": model_metrics["overall"]["bbox_accuracy"],
                "Total GT Tools": model_metrics["summary"]["total_gt_tools"],
                "Total Pred Tools": model_metrics["summary"]["total_pred_tools"],
                "Correct Tools": model_metrics["summary"]["correct_tools"],
                "Correct Bboxes": model_metrics["summary"]["correct_bboxes"],
                "Images": model_metrics["summary"]["total_images"]
            })
        
        # Create DataFrame and save to CSV
        df_overall = pd.DataFrame(overall_data)
        overall_path = os.path.join(output_dir, "summary_metrics.csv")
        df_overall.to_csv(overall_path, index=False, float_format='%.4f')
        logger.info(f"Saved summary metrics to {overall_path}")
        
        # Prepare data for per-tool metrics
        all_tools = set()
        for model_metrics in self.metrics.values():
            all_tools.update(model_metrics["per_tool"].keys())
        
        # Get counts for each tool
        first_model = list(self.metrics.keys())[0]
        tool_counts = {tool: self.metrics[first_model]["per_tool"].get(tool, {}).get("count", 0) 
                      for tool in all_tools}
        
        # Sort tools by frequency
        sorted_tools = sorted(all_tools, key=lambda t: tool_counts.get(t, 0), reverse=True)
        
        # Create per-tool metrics for each model
        for model_name, model_metrics in self.metrics.items():
            tool_data = []
            
            for tool in sorted_tools:
                if tool in model_metrics["per_tool"]:
                    tool_metrics = model_metrics["per_tool"][tool]
                    tool_data.append({
                        "Tool": tool,
                        "Precision": tool_metrics.get("precision", 0),
                        "Recall": tool_metrics.get("recall", 0),
                        "F1 Score": tool_metrics.get("f1", 0),
                        "TP": tool_metrics.get("tp", 0),
                        "FP": tool_metrics.get("fp", 0),
                        "FN": tool_metrics.get("fn", 0),
                        "Count": tool_metrics.get("count", 0)
                    })
            
            # Create DataFrame and save to CSV
            df_tool = pd.DataFrame(tool_data)
            tool_path = os.path.join(output_dir, f"{model_name}_tool_metrics.csv")
            df_tool.to_csv(tool_path, index=False, float_format='%.4f')
            logger.info(f"Saved {model_name} tool metrics to {tool_path}")
    
    def export_metrics(self, output_dir=None):
        """Export metrics to JSON file"""
        if output_dir is None:
            output_dir = self.output_dir
            
        logger.info("Exporting metrics to JSON")
        
        # Save metrics to JSON file
        output_path = os.path.join(output_dir, 'evaluation_metrics.json')
        
        # Convert numpy values to Python native types for JSON serialization
        export_metrics = {}
        
        for model, metrics in self.metrics.items():
            export_metrics[model] = {
                "average": {k: float(v) for k, v in metrics["average"].items()},
                "overall": {k: float(v) for k, v in metrics["overall"].items()},
                "per_tool": {
                    tool: {k: float(v) if isinstance(v, (np.number, float)) else v 
                           for k, v in values.items()}
                    for tool, values in metrics["per_tool"].items()
                },
                "summary": metrics["summary"]
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_metrics, f, indent=2)
        
        logger.info(f"Exported metrics to {output_path}")
        
        return output_path


def main():
    """Main function to run the evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate VLM tool recognition models")
    parser.add_argument("--gt_file", required=True, help="Path to ground truth CSV file")
    parser.add_argument("--pred_dir", required=True, help="Directory containing prediction CSV files")
    parser.add_argument("--output_dir", default="./results", help="Directory to save results")
    parser.add_argument("--tool_mapping", help="Optional JSON file with tool name mapping")
    parser.add_argument("--no_visualizations", action="store_true", 
                      help="Skip generating visualizations (only calculate metrics)")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ToolDetectionEvaluator(
        ground_truth_file=args.gt_file,
        prediction_dir=args.pred_dir,
        output_dir=args.output_dir,
        tool_mapping_file=args.tool_mapping
    )
    
    # Generate summary table
    evaluator.generate_summary_table()
    
    # Export metrics
    metrics_path = evaluator.export_metrics()
    
    # Generate visualizations if not skipped
    if not args.no_visualizations:
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        generate_visualizations(evaluator.metrics, viz_dir)
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()