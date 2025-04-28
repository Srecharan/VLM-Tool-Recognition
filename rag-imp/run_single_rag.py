import os
import argparse
import sys
from create_knowledge_base import create_knowledge_base_from_xml
from rag_implementation import run_rag_inference

def parse_args():
    parser = argparse.ArgumentParser(description='Run RAG on a single image')
    parser.add_argument('--model_id', type=str, 
                      default='akameswa/Qwen2.5-VL-7B-Instruct-bnb-4bit-finetune-vision-language', 
                      help='Hugging Face model ID')
    parser.add_argument('--xml_path', type=str, default='../dataset/tool_use.xml', help='Path to tool_use.xml file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to test image')
    parser.add_argument('--output_dir', type=str, default='rag_results', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Create knowledge base
    print("Step 1: Creating knowledge base...")
    knowledge_df = create_knowledge_base_from_xml(args.xml_path)
    
    if knowledge_df is None:
        print("Failed to create knowledge base. Exiting.")
        return
    
    # Step 2: Run RAG on a single image
    print(f"\nStep 2: Running RAG on image: {args.image_path}")
    print(f"Using model: {args.model_id}")
    
    results = run_rag_inference(
        image_path=args.image_path,
        model_path=args.model_id,  # Now using the model ID directly
        knowledge_df=knowledge_df
    )
    
    # Print a sample of the results
    print("\nRAG Results Summary:")
    print(f"Standard response length: {len(results['standard_response'])} characters")
    print(f"Enhanced response length: {len(results['enhanced_response'])} characters")
    print(f"Retrieved {len(results['retrieved_info'])} relevant tool information entries")
    
    print(f"\nResults saved to: {os.path.join(args.output_dir, 'single_image_results.json')}")
    
if __name__ == "__main__":
    main()