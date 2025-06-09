import os
import pandas as pd
import numpy as np
import json
import torch
import gc
from PIL import Image
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import unsloth
from typing import List, Dict, Any, Optional
from datetime import datetime

# LangChain imports
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS

# Import unsloth first to ensure optimizations are applied
from unsloth import FastVisionModel

# Disable torch compilation to avoid CUDA errors
os.environ["TORCH_COMPILE_MODE"] = "reduce-overhead"
os.environ["TORCH_INDUCTOR_DISABLE_CUDAGRAPHS"] = "1"

# Clear GPU memory at start
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, batch_size=8, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding[0].tolist()


class UnslothVLM:
    """Wrapper for Unsloth FastVisionModel to work with LangChain"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the Unsloth model"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"Loading Unsloth model from {self.model_path}")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.model_path,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        FastVisionModel.for_inference(self.model)
    
    def generate_response(self, image: Image.Image, prompt: str, max_tokens: int = 200) -> str:
        """Generate response for image and text prompt"""
        try:
            # Resize image to reduce memory usage
            resized_image = image.resize((224, 224))
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            input_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            
            inputs = self.tokenizer(
                [resized_image],
                text=[input_text],
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            print(f"Error in VLM generation: {e}")
            return f"Error processing image: {str(e)}"


class ToolSafetyKnowledgeBase:
    """Enhanced tool safety knowledge base with LangChain integration"""
    
    def __init__(self, csv_path: str = 'tool_knowledge_base.csv'):
        self.csv_path = csv_path
        self.knowledge_df = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the knowledge base"""
        self.knowledge_df = self._create_or_load_knowledge_base()
        self.embeddings = SentenceTransformerEmbeddings()
        self._create_vectorstore()
    
    def _create_or_load_knowledge_base(self) -> pd.DataFrame:
        """Create or load knowledge base from HuggingFace dataset"""
        if os.path.exists(self.csv_path):
            print(f"Loading existing knowledge base from {self.csv_path}")
            return pd.read_csv(self.csv_path)
        
        print("Creating knowledge base from HuggingFace dataset...")
        dataset = load_dataset("akameswa/tool-safety-dataset", split="valid")
        
        # Find all tool categories
        tool_categories = []
        for column in dataset.column_names:
            if column.endswith("_bboxes"):
                tool_name = column.replace("_bboxes", "")
                tool_categories.append(tool_name)
        
        print(f"Found {len(tool_categories)} tool categories: {', '.join(tool_categories)}")
        
        # Create knowledge base
        knowledge_base = []
        for tool in tool_categories:
            for example in dataset:
                bbox_key = f"{tool}_bboxes"
                if bbox_key in example and example[bbox_key]:
                    entry = {
                        "tool_name": tool,
                        "primary_function": example.get(f"{tool}_main_purpose", ""),
                        "usage_instructions": example.get(f"{tool}_usage_instructions", ""),
                        "safety_considerations": {
                            "required_ppe": example.get(f"{tool}_required_ppe", ""),
                            "primary_hazards": example.get(f"{tool}_primary_hazards", ""),
                            "common_misuses": example.get(f"{tool}_common_misuses", "")
                        }
                    }
                    knowledge_base.append(entry)
                    print(f"Added information for {tool}")
                    break
        
        # Save to CSV
        df = pd.DataFrame(knowledge_base)
        df.to_csv(self.csv_path, index=False)
        print(f"Knowledge base created with {len(df)} tools")
        
        return df
    
    def _create_vectorstore(self):
        """Create LangChain FAISS vectorstore"""
        print("Creating LangChain FAISS vectorstore...")
        
        # Prepare documents for vectorstore
        documents = []
        for _, row in self.knowledge_df.iterrows():
            safety_info = row['safety_considerations']
            if isinstance(safety_info, str):
                try:
                    safety_info = json.loads(safety_info)
                except:
                    pass
            
            # Create comprehensive text content
            content = f"Tool: {row['tool_name']}\n"
            content += f"Primary Function: {row['primary_function']}\n"
            content += f"Usage Instructions: {row['usage_instructions']}\n"
            
            if isinstance(safety_info, dict):
                content += f"Required PPE: {safety_info.get('required_ppe', '')}\n"
                content += f"Primary Hazards: {safety_info.get('primary_hazards', '')}\n"
                content += f"Common Misuses: {safety_info.get('common_misuses', '')}\n"
            
            # Create metadata
            metadata = {
                "tool_name": row['tool_name'],
                "source": "tool-safety-dataset",
                "type": "safety_information"
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Create vectorstore
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        print(f"Created vectorstore with {len(documents)} documents")
    
    def get_retriever(self) -> BaseRetriever:
        """Get the LangChain retriever"""
        return self.retriever


class LangChainRAGPipeline:
    """Enhanced RAG pipeline using LangChain orchestration"""
    
    def __init__(self, model_path: str, knowledge_base_path: str = 'tool_knowledge_base.csv'):
        self.vlm = UnslothVLM(model_path)
        self.knowledge_base = ToolSafetyKnowledgeBase(knowledge_base_path)
        self.retriever = self.knowledge_base.get_retriever()
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup LangChain chains for RAG workflow"""
        
        # Tool identification prompt
        self.tool_identification_prompt = PromptTemplate(
            template="""Analyze this image and identify all mechanical tools visible. 
            Be specific about the types of tools you can see.
            
            Focus on identifying:
            - Hand tools (wrenches, screwdrivers, pliers, etc.)
            - Power tools (drills, grinders, etc.)
            - Measuring tools (calipers, rules, etc.)
            - Cutting tools (saws, knives, etc.)
            - Safety equipment visible
            
            Provide a clear, concise list of tools identified.""",
            input_variables=[]
        )
        
        # Safety analysis prompt template
        self.safety_prompt = ChatPromptTemplate.from_template(
            """Based on the tools identified in the image and the retrieved safety information, 
            provide a comprehensive safety analysis.

            Identified Tools: {identified_tools}

            Retrieved Safety Information:
            {retrieved_info}

            Please provide a detailed safety analysis including:
            1. Primary Functions of identified tools
            2. Required Personal Protective Equipment (PPE)
            3. Primary Hazards associated with these tools
            4. Common Misuses to avoid
            5. Best practices for safe usage

            Format your response as a structured JSON with the following keys:
            - "tools_identified": list of tools
            - "primary_functions": dict mapping tool to function
            - "required_ppe": list of PPE items
            - "primary_hazards": list of hazards
            - "common_misuses": list of common misuses
            - "safety_recommendations": list of safety recommendations
            """
        )
        
        # Create the RAG chain
        self.rag_chain = (
            RunnablePassthrough.assign(
                retrieved_info=lambda x: self._format_retrieved_docs(
                    self.retriever.get_relevant_documents(x["identified_tools"])
                )
            )
            | self.safety_prompt
            | RunnableLambda(self._generate_safety_analysis)
            | JsonOutputParser()
        )
    
    def _format_retrieved_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for the prompt"""
        formatted = []
        for doc in docs:
            formatted.append(f"Tool: {doc.metadata.get('tool_name', 'Unknown')}")
            formatted.append(f"Information: {doc.page_content}")
            formatted.append("---")
        return "\n".join(formatted)
    
    def _generate_safety_analysis(self, prompt_value) -> str:
        """Generate safety analysis using the VLM (placeholder for now)"""
        # For now, return the formatted prompt - in a real implementation,
        # this would call your VLM or another language model
        return str(prompt_value)
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process an image through the complete RAG pipeline"""
        print(f"Processing image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Step 1: Tool identification using VLM
        print("Step 1: Identifying tools in image...")
        identified_tools = self.vlm.generate_response(
            image, 
            self.tool_identification_prompt.format(),
            max_tokens=150
        )
        
        print(f"Identified tools: {identified_tools}")
        
        # Step 2: Retrieve relevant safety information
        print("Step 2: Retrieving safety information...")
        try:
            safety_analysis = self.rag_chain.invoke({
                "identified_tools": identified_tools
            })
        except Exception as e:
            print(f"Error in RAG chain: {e}")
            # Fallback to basic retrieval
            docs = self.retriever.get_relevant_documents(identified_tools)
            safety_analysis = {
                "tools_identified": [identified_tools],
                "retrieved_docs": [doc.page_content for doc in docs],
                "error": str(e)
            }
        
        # Step 3: Generate comprehensive safety response
        print("Step 3: Generating comprehensive safety analysis...")
        comprehensive_prompt = f"""Based on the identified tools and retrieved safety information, 
        provide a comprehensive safety analysis:

        Identified Tools: {identified_tools}
        
        Safety Information Retrieved: {safety_analysis}
        
        Please provide detailed safety guidance including required PPE, hazards, and best practices."""
        
        final_response = self.vlm.generate_response(
            image,
            comprehensive_prompt,
            max_tokens=300
        )
        
        # Compile results
        results = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "identified_tools": identified_tools,
            "retrieved_safety_info": safety_analysis,
            "comprehensive_analysis": final_response,
            "processing_status": "success"
        }
        
        return results


def main():
    """Main function to run the LangChain RAG pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangChain-enhanced RAG for tool safety")
    parser.add_argument("--model_path", required=True, help="Path to the fine-tuned VLM model")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--output_dir", default="langchain_rag_outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize RAG pipeline
    print("Initializing LangChain RAG pipeline...")
    rag_pipeline = LangChainRAGPipeline(args.model_path)
    
    # Process image
    results = rag_pipeline.process_image(args.image_path)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"langchain_rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_file}")
    print("\nProcessing complete!")
    
    # Print summary
    print("\n" + "="*50)
    print("LANGCHAIN RAG ANALYSIS SUMMARY")
    print("="*50)
    print(f"Identified Tools: {results['identified_tools']}")
    print(f"Comprehensive Analysis: {results['comprehensive_analysis']}")


if __name__ == "__main__":
    main() 