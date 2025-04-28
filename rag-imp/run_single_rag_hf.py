import os
import argparse
import sys
from create_knowledge_base_hf import create_knowledge_base_from_hf
from rag_implementation import run_rag_inference
import requests
from PIL import Image
from io import BytesIO
import tempfile
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Run RAG on a single image')
    parser.add_argument('--model_id', type=str, 
                      default='akameswa/Qwen2.5-VL-7B-Instruct-bnb-4bit-finetune-vision-language', 
                      help='Hugging Face model ID')
    parser.add_argument('--image_path', type=str, default='', 
                       help='Path to test image (local file or URL)')
    parser.add_argument('--output_dir', type=str, default='rag_results', 
                       help='Output directory')
    parser.add_argument('--use_sample_image', action='store_true',
                      help='Use a sample image from the dataset')
    return parser.parse_args()

def get_sample_image():
    """Get a sample image directly from the dataset"""
    print("Loading a sample image from the dataset...")
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("akameswa/tool-safety-dataset", split="valid")
        
        # Get the first image from the dataset
        for i, example in enumerate(dataset):
            if 'image' in example and example['image'] is not None:
                print(f"Found image at index {i}")
                
                # Save image to a temporary file
                img = example['image']
                if isinstance(img, bytes):
                    img = Image.open(BytesIO(img))
                
                # Create a test_images directory if it doesn't exist
                os.makedirs('test_images', exist_ok=True)
                image_path = os.path.join('test_images', f'sample_image_{i}.jpg')
                
                # Save the image
                img.save(image_path)
                print(f"Saved sample image to {image_path}")
                return image_path
                
        print("No valid images found in the dataset")
        return None
    except Exception as e:
        print(f"Error fetching sample image: {e}")
        return None

def load_image(image_path_or_url):
    """Load an image from a local path or URL"""
    try:
        if image_path_or_url.startswith(('http://', 'https://')):
            # Load from URL
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content))
            
            # Save a local copy of the image for testing
            os.makedirs('test_images', exist_ok=True)
            local_path = os.path.join('test_images', 'downloaded_image.jpg')
            image.save(local_path)
            return local_path
        else:
            # Load from local path to verify it exists and is an image
            image = Image.open(image_path_or_url)
            return image_path_or_url
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Create knowledge base from HuggingFace dataset
    print("Step 1: Creating knowledge base from HuggingFace dataset...")
    knowledge_df = create_knowledge_base_from_hf()
    
    if knowledge_df is None:
        print("Failed to create knowledge base. Exiting.")
        return
    
    # Step 2: Get the image
    image_path = args.image_path
    if args.use_sample_image or not image_path:
        image_path = get_sample_image()
        if not image_path:
            print("Failed to get a sample image. Exiting.")
            return
    elif image_path.startswith(('http://', 'https://')):
        image_path = load_image(image_path)
    
    if not image_path or not os.path.exists(image_path):
        print(f"Image path {image_path} does not exist. Exiting.")
        return
    
    # Step 3: Run RAG on the image
    print(f"\nStep 3: Running RAG on image: {image_path}")
    print(f"Using model: {args.model_id}")
    
    results = run_rag_inference(
        image_path=image_path,
        model_path=args.model_id,
        knowledge_df=knowledge_df
    )
    
    # Print a sample of the results
    print("\nRAG Results Summary:")
    print(f"Standard response length: {len(results['standard_response'])} characters")
    print(f"Enhanced response length: {len(results['enhanced_response'])} characters")
    print(f"Retrieved {len(results['retrieved_info'])} relevant tool information entries")
    
    # Save results
    result_path = os.path.join(args.output_dir, 'single_image_results.json')
    
    import json
    with open(result_path, 'w') as f:
        # Convert retrieved_info to a serializable format
        serializable_results = {
            "standard_response": results["standard_response"],
            "enhanced_response": results["enhanced_response"],
            "retrieved_info": json.dumps(results["retrieved_info"])
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {result_path}")
    
if __name__ == "__main__":
    main()