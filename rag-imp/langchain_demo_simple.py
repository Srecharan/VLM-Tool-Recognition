import os
import pandas as pd
import json
from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from datetime import datetime

# LangChain imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain wrapper for SentenceTransformer"""
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, batch_size=8, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding[0].tolist()


class MockVLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"Mock VLM initialized for: {model_name}")
    
    def generate_response(self, image: Image.Image, prompt: str, max_tokens: int = 200) -> str:
        if "identify" in prompt.lower():
            return "Adjustable wrench, Phillips screwdriver, flathead screwdriver, needle-nose pliers, measuring tape, safety glasses"
        elif "safety" in prompt.lower():
            return """Required PPE: Safety glasses, work gloves, closed-toe shoes
            Primary Hazards: Pinch points, sharp edges, eye injury from debris
            Common Misuses: Using wrench as hammer, wrong size tool for job
            Best Practices: Inspect tools before use, maintain proper grip, use correct tool for task"""
        else:
            return f"Mock VLM response for: {prompt[:50]}..."


def create_knowledge_base():
    print("Creating tool safety knowledge base...")
    data = [
        {
            "tool_name": "adjustable_wrench",
            "primary_function": "Gripping and turning nuts, bolts, and pipe fittings",
            "usage_instructions": "Adjust jaw size to fit fastener snugly, pull rather than push when possible",
            "safety_considerations": json.dumps({
                "required_ppe": "Safety glasses, work gloves",
                "primary_hazards": "Pinch points between jaws, tool slipping off fastener",
                "common_misuses": "Using as hammer, using wrong size opening"
            })
        },
        {
            "tool_name": "screwdriver",
            "primary_function": "Driving screws into materials and removing screws",
            "usage_instructions": "Match screwdriver tip to screw head type and size exactly",
            "safety_considerations": json.dumps({
                "required_ppe": "Safety glasses to protect from metal fragments",
                "primary_hazards": "Sharp tip can cause puncture wounds, hand slipping off handle",
                "common_misuses": "Using as chisel, pry bar, or punch"
            })
        },
        {
            "tool_name": "pliers",
            "primary_function": "Gripping, twisting, and cutting wire and small objects",
            "usage_instructions": "Use appropriate pliers type for task, grip firmly near pivot point",
            "safety_considerations": json.dumps({
                "required_ppe": "Safety glasses, work gloves recommended",
                "primary_hazards": "Pinch points, sharp cutting edges, wire spring-back",
                "common_misuses": "Using as wrench for large fasteners, using wrong type for electrical work"
            })
        }
    ]
    
    df = pd.DataFrame(data)
    df.to_csv('tool_knowledge_base.csv', index=False)
    print(f"Knowledge base created with {len(df)} tools")
    return df


def demo_langchain_workflow():
    print("\n" + "="*60)
    print("LANGCHAIN RAG WORKFLOW DEMONSTRATION")
    print("="*60)
    print("Using mock VLM for demonstration")
    
    # Create knowledge base
    kb_df = create_knowledge_base()
    
    # Setup LangChain components
    print("\nSetting up LangChain components...")
    embeddings = SentenceTransformerEmbeddings()
    
    # Convert knowledge base to LangChain Documents
    documents = []
    for _, row in kb_df.iterrows():
        content = f"""Tool: {row['tool_name']}
Primary Function: {row['primary_function']}
Usage Instructions: {row['usage_instructions']}
Safety Information: {row['safety_considerations']}"""
        
        metadata = {
            "tool_name": row['tool_name'],
            "source": "safety_knowledge_base"
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    # Create FAISS vectorstore with LangChain
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print("LangChain FAISS vectorstore created")
    
    # Initialize Mock VLM
    vlm = MockVLM("akameswa/Llama-3.2-11B-Vision-Instruct-bnb-4bit-finetune-vision-language")
    
    # Create prompt templates
    identification_prompt = PromptTemplate(
        template="Identify all mechanical tools visible in this image. List each tool clearly.",
        input_variables=[]
    )
    
    safety_prompt = PromptTemplate(
        template="""Based on these identified tools: {tools}
        And this safety information: {safety_info}
        
        Provide comprehensive safety guidance including:
        - Required PPE
        - Primary hazards  
        - Best practices
        """,
        input_variables=["tools", "safety_info"]
    )
    
    # Process test image
    image_path = "rag-imp/test_images/sample_image_0.jpg"
    if not os.path.exists(image_path):
        print(f"Test image not found at {image_path}")
        print("Creating placeholder for demonstration...")
        image = Image.new('RGB', (224, 224), color='white')
    else:
        image = Image.open(image_path)
        print(f"Processing image: {image_path} ({image.size})")
    
    # Run RAG workflow
    print("\nRunning LangChain RAG Pipeline...")
    start_time = datetime.now()
    
    # Tool identification
    print("Tool Identification...")
    identified_tools = vlm.generate_response(image, identification_prompt.format())
    print(f"Found: {identified_tools}")
    
    # Information retrieval
    print("Information Retrieval...")
    docs = retriever.get_relevant_documents(identified_tools)
    retrieved_info = "\n---\n".join([doc.page_content for doc in docs])
    print(f"Retrieved info from {len(docs)} knowledge base entries")
    
    # Safety analysis generation
    print("Safety Analysis Generation...")
    safety_analysis = vlm.generate_response(
        image, 
        safety_prompt.format(tools=identified_tools, safety_info=retrieved_info)
    )
    print(f"Analysis: {safety_analysis}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_reference": "akameswa/Llama-3.2-11B-Vision-Instruct-bnb-4bit-finetune-vision-language",
        "identified_tools": identified_tools,
        "retrieved_documents": len(docs),
        "retrieved_info": retrieved_info,
        "safety_analysis": safety_analysis,
        "total_time_seconds": total_time,
        "vectorstore": "FAISS",
        "embeddings": "SentenceTransformer"
    }
    
    # Save results
    os.makedirs("langchain_lite_outputs", exist_ok=True)
    output_file = "langchain_lite_outputs/langchain_demo_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display summary
    print("\n" + "="*60)
    print("LANGCHAIN DEMO COMPLETED")
    print("="*60)
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Results saved: {output_file}")
    print(f"Tools Identified: {identified_tools}")
    print(f"Knowledge Retrieved: {len(docs)} documents")
    
    return results


if __name__ == "__main__":
    demo_langchain_workflow() 