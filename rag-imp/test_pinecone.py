import os
from dotenv import load_dotenv
from pinecone_rag import PineconeRAGPipeline

# Load environment variables
load_dotenv()

def test_pinecone():
    print("Testing Pinecone Integration")
    print("="*30)
    
    # Get API key from environment 
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("Error: PINECONE_API_KEY not found in .env file")
        return
    
    print("API key loaded from environment")
    
    # Initialize pipeline
    pipeline = PineconeRAGPipeline(
        pinecone_api_key=api_key,
        index_name="tool-safety-test"
    )
    
    # Load knowledge base
    pipeline.load_knowledge_base()
    
    # Test with placeholder image
    results = pipeline.process_image("test_images/sample_image_0.jpg")
    
    print("Test Results:")
    print(f"Tools: {results['identified_tools']}")
    print(f"Retrieved docs: {results['retrieved_documents']}")
    print(f"Time: {results['total_time_seconds']:.2f}s")
    print(f"Vectorstore: {results['vectorstore']}")
    
    print("\nPinecone test completed successfully!")

if __name__ == "__main__":
    test_pinecone() 