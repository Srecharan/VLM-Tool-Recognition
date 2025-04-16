import os
import xml.etree.ElementTree as ET
from pathlib import Path
import xml.dom.minidom as minidom

def load_tool_usage_info(tool_use_file):
    """Load tool usage information from the tool_use.xml file."""
    tree = ET.parse(tool_use_file)
    root = tree.getroot()
    
    # Create a dictionary to store tool information
    tool_info = {}
    
    for tool in root.findall('tool'):
        category = tool.find('tool_identification/tool_category').text
        tool_info[category] = {
            'main_purpose': tool.find('primary_function/main_purpose').text,
            'usage_instructions': tool.find('usage_instructions').text,
            'safety_considerations': {
                'required_ppe': tool.find('safety_considerations/required_ppe').text,
                'primary_hazards': tool.find('safety_considerations/primary_hazards').text,
                'common_misuses': tool.find('safety_considerations/common_misuses').text
            },
            'maintenance_guidance': {
                'cleaning': tool.find('maintenance_guidance/cleaning').text,
                'storage': tool.find('maintenance_guidance/storage').text,
                'maintenance_frequency': tool.find('maintenance_guidance/maintenance_frequency').text
            }
        }
    
    return tool_info

def add_tool_info_to_xml(xml_file, tool_info):
    """Add tool usage information to an XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get all object elements
    objects = root.findall('object')
    
    for obj in objects:
        tool_name = obj.find('name').text
        
        # Check if we have info for this tool
        if tool_name in tool_info:
            info = tool_info[tool_name]
            
            # Add tool usage information
            usage_info = ET.SubElement(obj, 'tool_usage')
            
            # Add main purpose
            main_purpose = ET.SubElement(usage_info, 'main_purpose')
            main_purpose.text = info['main_purpose']
            
            # Add usage instructions
            instructions = ET.SubElement(usage_info, 'usage_instructions')
            instructions.text = info['usage_instructions']
            
            # Add safety considerations
            safety = ET.SubElement(usage_info, 'safety_considerations')
            ppe = ET.SubElement(safety, 'required_ppe')
            ppe.text = info['safety_considerations']['required_ppe']
            hazards = ET.SubElement(safety, 'primary_hazards')
            hazards.text = info['safety_considerations']['primary_hazards']
            misuses = ET.SubElement(safety, 'common_misuses')
            misuses.text = info['safety_considerations']['common_misuses']
            
            # Add maintenance guidance
            maintenance = ET.SubElement(usage_info, 'maintenance_guidance')
            cleaning = ET.SubElement(maintenance, 'cleaning')
            cleaning.text = info['maintenance_guidance']['cleaning']
            storage = ET.SubElement(maintenance, 'storage')
            storage.text = info['maintenance_guidance']['storage']
            frequency = ET.SubElement(maintenance, 'maintenance_frequency')
            frequency.text = info['maintenance_guidance']['maintenance_frequency']
    
    # Convert to string and prettify
    xml_str = ET.tostring(root, encoding='unicode')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent='    ')
    
    # Remove extra blank lines
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    
    # Write back to file
    with open(xml_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

def process_clean_directory(clean_dir, tool_use_file):
    """Process all XML files in clean directory and its subdirectories."""
    # Load tool usage information
    tool_info = load_tool_usage_info(tool_use_file)
    
    # Process each split directory (train, test, valid)
    for split in ['train', 'test', 'valid']:
        split_dir = os.path.join(clean_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split} directory not found")
            continue
        
        # Process each XML file in the split directory
        for xml_file in Path(split_dir).glob('*.xml'):
            try:
                add_tool_info_to_xml(str(xml_file), tool_info)
                print(f"Processed: {xml_file.name}")
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean_dir = os.path.join(base_dir, 'xml')
    tool_use_file = os.path.join(base_dir, 'tool_use.xml')
    
    if not os.path.exists(tool_use_file):
        print(f"Error: tool_use.xml not found at {tool_use_file}")
    else:
        process_clean_directory(clean_dir, tool_use_file)
        print("Processing complete. Tool usage information added to all XML files.")
