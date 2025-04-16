import os
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
import io
from datasets import Dataset, Image as HFImage
from huggingface_hub import login
import glob
import json
from tqdm import tqdm

# Set your HF token
HF_TOKEN = ""  # Replace with your token
login(token=HF_TOKEN)

def load_tool_info():
    """Load tool information from tool_use.xml"""
    tree = ET.parse('tool_use.xml')
    root = tree.getroot()
    
    tool_info = {}
    for tool in root.findall('.//tool'):
        category = tool.find('.//tool_category').text
        info = {
            'tool_category': category,
            'main_purpose': tool.find('.//main_purpose').text,
            'usage_instructions': tool.find('.//usage_instructions').text,
            'required_ppe': tool.find('.//required_ppe').text,
            'primary_hazards': tool.find('.//primary_hazards').text,
            'common_misuses': tool.find('.//common_misuses').text,
            'cleaning': tool.find('.//cleaning').text,
            'storage': tool.find('.//storage').text,
            'maintenance_frequency': tool.find('.//maintenance_frequency').text
        }
        tool_info[category] = info
    return tool_info

def load_image(image_path):
    """Load and convert image to bytes"""
    image = Image.open(image_path)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    return image_bytes.getvalue()

def parse_annotation_xml(xml_path, tool_info):
    """Parse annotation XML file and extract tool information with bounding boxes"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    objects = root.findall('object')
    
    tools_in_image = {}
    for obj in objects:
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        bbox_coords = [
            int(bbox.find('xmin').text),
            int(bbox.find('ymin').text),
            int(bbox.find('xmax').text),
            int(bbox.find('ymax').text)
        ]
        
        if name not in tools_in_image:
            tools_in_image[name] = {
                'bboxes': [bbox_coords],
                'info': tool_info.get(name, {})
            }
        else:
            tools_in_image[name]['bboxes'].append(bbox_coords)
    
    return filename, tools_in_image

def create_dataset(split_name, tool_info):
    """Create dataset for a specific split (train/test/valid)"""
    data = {
        'image': [],
        'image_name': [],
    }
    
    # Initialize columns for each tool category
    for tool_category in tool_info.keys():
        # Initialize bounding boxes as empty list - will be filled with lists of [x1,y1,x2,y2]
        data[f'{tool_category}_bboxes'] = []
        data[f'{tool_category}_main_purpose'] = []
        data[f'{tool_category}_usage_instructions'] = []
        data[f'{tool_category}_required_ppe'] = []
        data[f'{tool_category}_primary_hazards'] = []
        data[f'{tool_category}_common_misuses'] = []
        data[f'{tool_category}_cleaning'] = []
        data[f'{tool_category}_storage'] = []
        data[f'{tool_category}_maintenance_frequency'] = []
    
    # Get all XML files in the split directory
    xml_dir = f'xml/{split_name}'
    image_dir = f'images/{split_name}'
    xml_files = glob.glob(f'{xml_dir}/*.xml')
    
    # Create progress bar
    pbar = tqdm(xml_files, desc=f"Processing {split_name} split", unit="file")
    
    for xml_path in pbar:
        filename, tools_in_image = parse_annotation_xml(xml_path, tool_info)
        image_path = os.path.join(image_dir, filename)
        
        if os.path.exists(image_path):
            # Load image
            image_data = load_image(image_path)
            
            # Add basic image info
            data['image'].append(image_data)
            data['image_name'].append(filename)
            
            # Add tool information
            for tool_category in tool_info.keys():
                tool_data = tools_in_image.get(tool_category, {})
                tool_info_data = tool_data.get('info', {})
                
                # Add bounding boxes (empty list if tool not present)
                data[f'{tool_category}_bboxes'].append(tool_data.get('bboxes', []))
                
                # Add tool information (empty string if tool not present)
                data[f'{tool_category}_main_purpose'].append(tool_info_data.get('main_purpose', ''))
                data[f'{tool_category}_usage_instructions'].append(tool_info_data.get('usage_instructions', ''))
                data[f'{tool_category}_required_ppe'].append(tool_info_data.get('required_ppe', ''))
                data[f'{tool_category}_primary_hazards'].append(tool_info_data.get('primary_hazards', ''))
                data[f'{tool_category}_common_misuses'].append(tool_info_data.get('common_misuses', ''))
                data[f'{tool_category}_cleaning'].append(tool_info_data.get('cleaning', ''))
                data[f'{tool_category}_storage'].append(tool_info_data.get('storage', ''))
                data[f'{tool_category}_maintenance_frequency'].append(tool_info_data.get('maintenance_frequency', ''))
            
            # Update progress bar description with current file
            pbar.set_postfix(file=filename)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def save_preview(df, split_name):
    """Save a preview of the DataFrame to a file"""
    # Create a copy of the DataFrame without the image column for preview
    preview_df = df.drop('image', axis=1)
    
    # Save first 5 rows to a JSON file
    preview_file = f'preview_{split_name}.json'
    preview_df.head().to_json(preview_file, orient='records', indent=2)
    print(f"\nPreview saved to {preview_file}")
    
    # Also print some basic information
    print(f"\nDataset Info for {split_name}:")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"\nFirst few rows preview saved to {preview_file}")

def main():
    # Load tool information from tool_use.xml
    tool_info = load_tool_info()
    
    # Create datasets for each split
    dataset_name = "akameswa/tool-safety-dataset"  # Replace with your desired name
    
    splits = ['train', 'valid', 'test']
    datasets = {}
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        df = create_dataset(split, tool_info)
        datasets[split] = df
        
        # Save and show preview
        save_preview(df, split)
    
    # Ask user if they want to push to HuggingFace
    while True:
        response = input("\nDo you want to push the datasets to HuggingFace? (yes/no): ").lower()
        if response in ['yes', 'no']:
            break
        print("Please enter 'yes' or 'no'")
    
    if response == 'yes':
        print("\nPushing datasets to HuggingFace...")
        for split, df in datasets.items():
            print(f"\nPushing {split} split...")
            
            # Validate bounding box structures
            for column in df.columns:
                if column.endswith('_bboxes'):
                    # Ensure empty bounding boxes are represented as empty lists, not null
                    df[column] = df[column].apply(lambda x: [] if not x else x)
            
            dataset = Dataset.from_pandas(df).cast_column("image", HFImage())
            dataset.push_to_hub(dataset_name, split=split)
            print(f"Finished uploading {split} split")
        print("\nAll datasets have been pushed to HuggingFace successfully!")
    else:
        print("\nDataset push cancelled. You can find the preview files in the current directory.")

if __name__ == "__main__":
    main()