import os
import xml.etree.ElementTree as ET
from pathlib import Path
import xml.dom.minidom as minidom

def clean_xml(input_xml_path, output_xml_path, split_type):
    """Clean and simplify XML file with only required information."""
    tree = ET.parse(input_xml_path)
    root = tree.getroot()
    
    # Create new XML structure
    new_root = ET.Element('annotation')
    
    # Add filename
    filename = root.find('filename').text
    ET.SubElement(new_root, 'filename').text = filename
    
    # Add split type
    ET.SubElement(new_root, 'split').text = split_type
    
    # Add objects with only name and bndbox
    objects = root.findall('object')
    for obj in objects:
        new_obj = ET.SubElement(new_root, 'object')
        # Copy name
        ET.SubElement(new_obj, 'name').text = obj.find('name').text
        
        # Copy bounding box
        old_bbox = obj.find('bndbox')
        new_bbox = ET.SubElement(new_obj, 'bndbox')
        for coord in ['xmin', 'ymin', 'xmax', 'ymax']:
            ET.SubElement(new_bbox, coord).text = old_bbox.find(coord).text
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    
    # Convert to string and prettify
    xml_str = ET.tostring(new_root, encoding='unicode')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent='    ')
    
    # Remove extra blank lines (minidom can add these)
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    
    # Write the cleaned and formatted XML file
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

def process_dataset(base_dir):
    """Process all XML files and create cleaned versions."""
    # Create clean directory structure
    clean_base = os.path.join(base_dir, 'xml')
    
    # Process each split type
    for split in ['train', 'test', 'valid']:
        input_dir = os.path.join(base_dir, 'roboflow', split)
        output_dir = os.path.join(clean_base, split)
        
        if not os.path.exists(input_dir):
            print(f"Warning: {split} directory not found")
            continue
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each XML file in the split
        for xml_file in Path(input_dir).glob('*.xml'):
            output_file = os.path.join(output_dir, xml_file.name)
            try:
                clean_xml(xml_file, output_file, split)
                print(f"Processed: {xml_file.name}")
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    process_dataset(base_dir)
    print("Processing complete. Cleaned XML files are in the 'clean' directory.")
