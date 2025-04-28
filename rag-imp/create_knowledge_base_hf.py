import os
import pandas as pd
import json
from datasets import load_dataset
import re

def create_knowledge_base_from_hf():
    """
    Extract tool information directly from the Hugging Face dataset
    """
    print("Loading dataset from Hugging Face...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("akameswa/tool-safety-dataset", split="valid")
        
        # Create a list to store tool information
        tool_info = []
        
        # Get list of all column names
        all_columns = dataset.column_names
        
        # Find the tool category names by looking for columns that end with "_bboxes"
        tool_categories = [col.replace("_bboxes", "") for col in all_columns if col.endswith("_bboxes")]
        
        print(f"Found {len(tool_categories)} tool categories: {', '.join(tool_categories)}")
        
        # Process each tool category
        for category in tool_categories:
            # Create base keys for this tool's properties
            bbox_key = f"{category}_bboxes"
            purpose_key = f"{category}_main_purpose"
            instructions_key = f"{category}_usage_instructions"
            ppe_key = f"{category}_required_ppe"
            hazards_key = f"{category}_primary_hazards"
            misuses_key = f"{category}_common_misuses"
            
            # Check if all required keys exist
            required_keys = [purpose_key, instructions_key, ppe_key, hazards_key, misuses_key]
            
            if not all(key in all_columns for key in required_keys):
                print(f"Warning: Some required information is missing for {category}")
                continue
            
            # We've found a valid tool category, process the first entry with non-empty information
            for i, example in enumerate(dataset):
                # Check if this example has this tool (non-empty bounding box)
                if bbox_key in example and example[bbox_key]:
                    # Extract the tool information
                    tool_data = {
                        'tool_name': category,
                        'primary_function': example.get(purpose_key, ""),
                        'usage_instructions': example.get(instructions_key, ""),
                        'safety_considerations': {
                            'required_ppe': example.get(ppe_key, ""),
                            'primary_hazards': example.get(hazards_key, ""),
                            'common_misuses': example.get(misuses_key, "")
                        }
                    }
                    
                    # Add to tool info list
                    tool_info.append(tool_data)
                    print(f"Added information for {category}")
                    # Only need one entry per tool category
                    break
        
        # Convert to DataFrame
        df = pd.DataFrame(tool_info)
        
        # Save to CSV for easier access
        output_path = 'tool_knowledge_base.csv'
        df.to_csv(output_path, index=False)
        
        print(f"Knowledge base created with {len(df)} tools and saved to {output_path}")
        return df
        
    except Exception as e:
        print(f"Error accessing dataset: {e}")
        return None

# Example usage
if __name__ == "__main__":
    knowledge_df = create_knowledge_base_from_hf()
    
    if knowledge_df is not None:
        # Display first few entries
        print("\nFirst few entries in the knowledge base:")
        pd.set_option('display.max_colwidth', 30)  # Limit column width for display
        print(knowledge_df.head())