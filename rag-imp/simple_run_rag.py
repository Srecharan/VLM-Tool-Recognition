import os
from create_knowledge_base_hf import create_knowledge_base_from_hf
from rag_implementation import run_rag_inference

def main():
    # Create output directory
    output_dir = 'rag_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create knowledge base from HF dataset
    print("Step 1: Creating knowledge base from HuggingFace dataset...")
    knowledge_df = create_knowledge_base_from_hf()
    
    if knowledge_df is None:
        print("Failed to create knowledge base. Exiting.")
        return
    
    # Step 2: Set sample image and use fallback model
    test_image_path = "test_images/sample_image_0.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Image not found at {test_image_path}. Please run run_single_rag_hf.py first to download a sample image.")
        return
    
    # Step 3: Run RAG on the image using a known working model
    print(f"\nStep 3: Running RAG on image: {test_image_path}")
    print("Using fallback model: Qwen/Qwen-VL-Chat")
    
    # Run with fallback model
    results = run_rag_inference(
        image_path=test_image_path,
        model_path="Qwen/Qwen-VL-Chat",  # Use known working model
        knowledge_df=knowledge_df
    )
    
    # Print a sample of the results
    print("\nRAG Results Summary:")
    print(f"Standard response length: {len(results['standard_response'])} characters")
    print(f"Enhanced response length: {len(results['enhanced_response'])} characters")
    print(f"Retrieved {len(results['retrieved_info'])} relevant tool information entries")
    
    # Save results
    import json
    result_path = os.path.join(output_dir, 'rag_results.json')
    
    with open(result_path, 'w') as f:
        # Convert retrieved_info to a serializable format
        serializable_results = {
            "standard_response": results["standard_response"],
            "enhanced_response": results["enhanced_response"],
            "retrieved_info": json.dumps(results["retrieved_info"])
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {result_path}")
    print("\nSuccessfully implemented RAG for the VLM-Tool-Recognition project!")
    
if __name__ == "__main__":
    main()