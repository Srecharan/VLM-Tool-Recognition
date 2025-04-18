import os
import pandas as pd
import xml.etree.ElementTree as ET
import glob
import json
from tqdm import tqdm

def parse_annotation_xml(xml_path):
    """Parse annotation XML file and extract tool information with bounding boxes"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    objects = root.findall('object')
    
    bounding_boxes = []
    for obj in objects:
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        bbox_coords = [
            int(bbox.find('xmin').text),
            int(bbox.find('ymin').text),
            int(bbox.find('xmax').text),
            int(bbox.find('ymax').text)
        ]
        
        bounding_boxes.append({
            'tool': name,
            'bbox': bbox_coords
        })
    
    return filename, bounding_boxes

def process_dataset(split_name='valid', xml_dir='xml', output_path='output.csv'):
    """Process XML files and create CSV with filename and bounding boxes"""
    data = []
    
    # Get all XML files in the directory
    xml_files = glob.glob(f'{xml_dir}/{split_name}/*.xml')
    
    # Process each XML file with progress bar
    for xml_path in tqdm(xml_files, desc="Processing XML files"):
        filename, bounding_boxes = parse_annotation_xml(xml_path)
        
        # Add to data list
        data.append({
            'filename': filename,
            'bounding_boxes': json.dumps(bounding_boxes)  # Convert to JSON string
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"\nOutput saved to {output_path}")

if __name__ == "__main__":
    process_dataset()