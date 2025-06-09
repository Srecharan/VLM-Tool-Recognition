import os
import pandas as pd
import json
import time
from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
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
            return "Adjustable wrench, Phillips screwdriver, needle-nose pliers, measuring tape, safety glasses"
        elif "safety" in prompt.lower():
            return "Required PPE: Safety glasses, work gloves. Hazards: Pinch points, sharp edges. Best practices: Inspect tools, use proper technique."
        else:
            return f"Mock response for: {prompt[:50]}..."


class PineconeRAGPipeline:
    def __init__(self, pinecone_api_key: str, index_name: str = "tool-safety-index", 
                 model_name: str = "akameswa/Llama-3.2-11B-Vision-Instruct-bnb-4bit-finetune-vision-language"):
        
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.model_name = model_name
        
        # Initialize components
        self.embeddings = SentenceTransformerEmbeddings()
        self.vlm = MockVLM(model_name)
        
        self._initialize_pinecone()
        self._setup_prompts()
    
    def _initialize_pinecone(self):
        print("Initializing Pinecone...")
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        existing_indexes = self.pc.list_indexes().names()
        
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embeddings.dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            while not self.pc.describe_index(self.index_name).status['ready']:
                print("Waiting for index...")
                time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone index: {self.index_name}")
        
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text"
        )
    
    def _setup_prompts(self):
        self.identification_prompt = PromptTemplate(
            template="Identify all mechanical tools visible in this image.",
            input_variables=[]
        )
        
        self.safety_prompt = PromptTemplate(
            template="Based on tools: {tools} and safety info: {safety_info}, provide safety guidance.",
            input_variables=["tools", "safety_info"]
        )
    
    def load_knowledge_base(self):
        print("Loading knowledge base into Pinecone...")
        
        index_stats = self.index.describe_index_stats()
        if index_stats['total_vector_count'] > 0:
            print(f"Index already contains {index_stats['total_vector_count']} vectors")
            return
        
        data = [
            {
                "tool_name": "adjustable_wrench",
                "content": "Tool: adjustable_wrench. Function: Gripping nuts and bolts. PPE: Safety glasses, gloves. Hazards: Pinch points."
            },
            {
                "tool_name": "screwdriver", 
                "content": "Tool: screwdriver. Function: Driving screws. PPE: Safety glasses. Hazards: Sharp tip, hand slipping."
            },
            {
                "tool_name": "pliers",
                "content": "Tool: pliers. Function: Gripping wire and objects. PPE: Safety glasses, gloves. Hazards: Pinch points, sharp edges."
            }
        ]
        
        documents = []
        for item in data:
            doc = Document(
                page_content=item["content"],
                metadata={"tool_name": item["tool_name"], "source": "safety_kb"}
            )
            documents.append(doc)
        
        self.vectorstore.add_documents(documents)
        print(f"Added {len(documents)} documents to Pinecone")
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        print(f"Processing image: {image_path}")
        
        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            image = Image.new('RGB', (224, 224), color='white')
        
        start_time = datetime.now()
        
        # Tool identification
        identified_tools = self.vlm.generate_response(image, self.identification_prompt.format())
        print(f"Found: {identified_tools}")
        
        # Retrieve from Pinecone
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(identified_tools)
        retrieved_info = "\n".join([doc.page_content for doc in docs])
        print(f"Retrieved {len(docs)} documents from Pinecone")
        
        # Safety analysis
        safety_analysis = self.vlm.generate_response(
            image, 
            self.safety_prompt.format(tools=identified_tools, safety_info=retrieved_info)
        )
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "model_reference": self.model_name,
            "identified_tools": identified_tools,
            "retrieved_documents": len(docs),
            "safety_analysis": safety_analysis,
            "total_time_seconds": total_time,
            "vectorstore": "Pinecone",
            "pinecone_index": self.index_name
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pinecone RAG for tool safety")
    parser.add_argument("--pinecone_api_key", required=True, help="Pinecone API key")
    parser.add_argument("--index_name", default="tool-safety-index", help="Index name")
    parser.add_argument("--image_path", default="test_images/sample_image_0.jpg", help="Image path")
    parser.add_argument("--output_dir", default="pinecone_outputs", help="Output directory")
    
    args = parser.parse_args()
    
    print("Pinecone RAG Pipeline")
    print("="*40)
    
    pipeline = PineconeRAGPipeline(
        pinecone_api_key=args.pinecone_api_key,
        index_name=args.index_name
    )
    
    pipeline.load_knowledge_base()
    results = pipeline.process_image(args.image_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"pinecone_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Pinecone RAG Demo Completed")
    print(f"Total Time: {results['total_time_seconds']:.2f} seconds")
    print(f"Results saved: {output_file}")


if __name__ == "__main__":
    main()
