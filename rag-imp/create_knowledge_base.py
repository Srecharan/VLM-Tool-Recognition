import os
import pandas as pd
import xml.etree.ElementTree as ET
import json
import sys

def create_knowledge_base_from_xml(xml_file_path):
    """
    Extract tool information from the tool_use.xml file to create a knowledge base
    """
    # Check if the file exists
    if not os.path.exists(xml_file_path):
        print(f"Error: XML file not found at {xml_file_path}")
        return None
        
    # Parse the XML file
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Create a list to store tool information
        tool_info = []
        
        # Extract information for each tool
        for tool in root.findall('.//tool'):
            try:
                # Extract tool category
                category_elem = tool.find('.//tool_category')
                if category_elem is None:
                    print("Warning: Tool without category found, skipping")
                    continue
                    
                category = category_elem.text
                
                # Extract primary function
                main_purpose_elem = tool.find('.//main_purpose')
                main_purpose = main_purpose_elem.text if main_purpose_elem is not None else ""
                
                # Extract usage instructions
                usage_elem = tool.find('.//usage_instructions')
                usage_instructions = usage_elem.text if usage_elem is not None else ""
                
                # Extract safety considerations
                safety_ppe_elem = tool.find('.//safety_considerations/required_ppe')
                safety_ppe = safety_ppe_elem.text if safety_ppe_elem is not None else ""
                
                safety_hazards_elem = tool.find('.//safety_considerations/primary_hazards')
                safety_hazards = safety_hazards_elem.text if safety_hazards_elem is not None else ""
                
                safety_misuses_elem = tool.find('.//safety_considerations/common_misuses')
                safety_misuses = safety_misuses_elem.text if safety_misuses_elem is not None else ""
                
                # Extract maintenance info
                cleaning_elem = tool.find('.//maintenance_guidance/cleaning')
                cleaning = cleaning_elem.text if cleaning_elem is not None else ""
                
                storage_elem = tool.find('.//maintenance_guidance/storage')
                storage = storage_elem.text if storage_elem is not None else ""
                
                maint_freq_elem = tool.find('.//maintenance_guidance/maintenance_frequency')
                maint_freq = maint_freq_elem.text if maint_freq_elem is not None else ""
                
                # Create tool data dictionary
                tool_data = {
                    'tool_name': category,
                    'primary_function': main_purpose,
                    'usage_instructions': usage_instructions,
                    'safety_considerations': {
                        'required_ppe': safety_ppe,
                        'primary_hazards': safety_hazards,
                        'common_misuses': safety_misuses
                    },
                    'maintenance': {
                        'cleaning': cleaning,
                        'storage': storage,
                        'maintenance_frequency': maint_freq
                    }
                }
                
                tool_info.append(tool_data)
                print(f"Processed tool: {category}")
                
            except Exception as e:
                print(f"Error processing tool: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(tool_info)
        
        # Save to CSV for easier access
        output_path = 'tool_knowledge_base.csv'
        df.to_csv(output_path, index=False)
        
        print(f"Knowledge base created with {len(df)} tools and saved to {output_path}")
        return df
        
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Use command line argument if provided, otherwise use default path
    xml_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/tool_use.xml'
    
    print(f"Creating knowledge base from {xml_path}")
    knowledge_df = create_knowledge_base_from_xml(xml_path)
    
    if knowledge_df is not None:
        # Display first few entries
        print("\nFirst few entries in the knowledge base:")
        pd.set_option('display.max_colwidth', 30)  # Limit column width for display
        print(knowledge_df.head())