import os
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Run the complete RAG workflow')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--xml_path', type=str, default='dataset/tool_use.xml', help='Path to the tool_use.xml file')
    parser.add_argument('--test_csv', type=str, required=True, help='CSV file with test image paths')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='rag_workflow_outputs', help='Base output directory')
    parser.add_argument('--max_images', type=int, default=10, help='Maximum number of images to process')
    return parser.parse_args()

def run_workflow(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Create knowledge base from XML
    print("\n=== Step 1: Creating Knowledge Base ===")
    knowledge_base_output = os.path.join(args.output_dir, 'tool_knowledge_base.csv')
    
    command = [
        sys.executable,
        'create_knowledge_base.py',
        args.xml_path
    ]
    
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        
        # Check if knowledge base was created
        if not os.path.exists('tool_knowledge_base.csv'):
            print("Error: Knowledge base not created. Exiting workflow.")
            return False
        
        # Move knowledge base to output directory
        if os.path.exists('tool_knowledge_base.csv'):
            import shutil
            shutil.copy('tool_knowledge_base.csv', knowledge_base_output)
            print(f"Knowledge base moved to {knowledge_base_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating knowledge base: {e}")
        return False
    
    # Step 2: Run batch RAG
    print("\n=== Step 2: Running Batch RAG ===")
    batch_output_dir = os.path.join(args.output_dir, 'batch_results')
    
    command = [
        sys.executable,
        'batch_rag.py',
        '--model_path', args.model_path,
        '--knowledge_base', knowledge_base_output,
        '--test_csv', args.test_csv,
        '--image_dir', args.image_dir,
        '--output_dir', batch_output_dir,
        '--max_images', str(args.max_images)
    ]
    
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        
        # Check if batch results were created
        batch_results_file = os.path.join(batch_output_dir, 'batch_results.csv')
        if not os.path.exists(batch_results_file):
            print("Error: Batch results not created. Skipping analysis.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error running batch RAG: {e}")
        return False
    
    # Step 3: Analyze results
    print("\n=== Step 3: Analyzing Results ===")
    analysis_output_dir = os.path.join(args.output_dir, 'analysis')
    
    batch_results_file = os.path.join(batch_output_dir, 'batch_results.csv')
    
    command = [
        sys.executable,
        'analyze_rag.py',
        '--results_path', batch_results_file,
        '--output_dir', analysis_output_dir
    ]
    
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing results: {e}")
        return False
    
    print("\n=== Workflow Complete! ===")
    print(f"All outputs saved to {args.output_dir}")
    
    return True

def main():
    args = parse_args()
    success = run_workflow(args)
    
    if success:
        print("\nRAG implementation workflow completed successfully!")
    else:
        print("\nRAG implementation workflow completed with errors.")

if __name__ == "__main__":
    main()