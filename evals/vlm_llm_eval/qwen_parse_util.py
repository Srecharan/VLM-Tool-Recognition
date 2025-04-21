import pandas as pd
import json
import re

def parse_qwen_response(file_path):
    """
    Parse a Qwen CSV file and extract structured information from the full_response column.
    
    Args:
        file_path (str): Path to the Qwen CSV file
        
    Returns:
        pd.DataFrame: DataFrame with parsed tool information
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Initialize lists to store parsed data
    parsed_data = []
    
    # Process each row
    for idx, row in df.iterrows():
        filename = row['filename']
        full_response = row['full_response']
        
        try:
            # First, try to extract the JSON part
            # The pattern is typically: [bounding boxes]{"tools_information": [...]}
            json_part = None
            
            # If there's a clear JSON structure after the bounding boxes
            if ']{"tools_information"' in full_response:
                json_part = full_response.split(']', 1)[1]
            else:
                # Try to find a JSON object in the string
                matches = re.findall(r'(\{.*\})', full_response, re.DOTALL)
                if matches:
                    for match in matches:
                        try:
                            # Try to parse each potential JSON match
                            json_obj = json.loads(match)
                            if "tools_information" in json_obj:
                                json_part = match
                                break
                        except:
                            continue
            
            # If we found a valid JSON part
            if json_part:
                data = json.loads(json_part)
                
                # Extract tools information
                if "tools_information" in data:
                    tools = data["tools_information"]
                    
                    # Process each tool
                    for tool in tools:
                        tool_name = tool.get("tool", "Unknown")
                        primary_function = tool.get("primary_function", "")
                        
                        # Extract safety considerations
                        safety = tool.get("safety_considerations", {})
                        required_ppe = safety.get("required_ppe", "")
                        primary_hazards = ", ".join(safety.get("primary_hazards", []))
                        common_misuses = ", ".join(safety.get("common_misuses", []))
                        
                        # Add to parsed data
                        parsed_data.append({
                            "filename": filename,
                            "tool": tool_name,
                            "primary_function": primary_function,
                            "required_ppe": required_ppe,
                            "primary_hazards": primary_hazards,
                            "common_misuses": common_misuses
                        })
        except Exception as e:
            print(f"Error parsing row {idx} for file {filename}: {e}")
            # Add a row with error information
            parsed_data.append({
                "filename": filename,
                "tool": "ERROR",
                "primary_function": f"Parse error: {str(e)}",
                "required_ppe": "",
                "primary_hazards": "",
                "common_misuses": ""
            })
    
    # Create DataFrame from parsed data
    result_df = pd.DataFrame(parsed_data)
    
    return result_df

# Example usage
if __name__ == "__main__":
    # Test parsing with a sample file
    file_path = "/home/rex/VLM-Tool-Recognition/evals/results/qwen-v.csv"
    parsed_df = parse_qwen_response(file_path)
    
    # Display sample of parsed data
    print(parsed_df.head())
    
    # Save to a new CSV file
    parsed_df.to_csv("qwen-v_parsed.csv", index=False)
    print(f"Parsed data saved to qwen-v_parsed.csv")