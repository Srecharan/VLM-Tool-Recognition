# LangChain Integration for VLM Tool Recognition

## Overview

This integration enhances your existing FAISS-based RAG system with LangChain orchestration, providing:

- **Better workflow management** through LangChain chains
- **Enhanced prompt templating** and management
- **Improved error handling** and retry logic
- **Modular pipeline components** for easier maintenance
- **Memory and context management** capabilities

## Key Components

### 1. SentenceTransformerEmbeddings
- LangChain wrapper for your existing sentence-transformers model
- Maintains compatibility with your current embedding approach
- Enables seamless integration with LangChain vectorstores

### 2. UnslothVLM
- Wrapper for your Unsloth FastVisionModel
- Provides consistent interface for image+text processing
- Maintains your existing 4-bit quantization and optimization settings

### 3. ToolSafetyKnowledgeBase
- Enhanced version of your knowledge base creation
- Uses LangChain's FAISS vectorstore instead of raw FAISS
- Structured document management with metadata

### 4. LangChainRAGPipeline
- Orchestrates the complete RAG workflow
- Implements chain-based processing with error handling
- Provides structured prompt templates for consistency

## Improvements Over Original RAG

### Original (simplified_rag.py)
```python
# Manual workflow orchestration
# Basic error handling
# Simple retrieval and generation
```

### Enhanced (langchain_rag.py)
```python
# Chain-based workflow management
# Robust error handling with fallbacks
# Structured prompt templates
# Better document management
# Extensible pipeline design
```

## Usage

### Basic Usage
```bash
./rag-imp/run_langchain_demo.sh <model_path> <image_path> [output_dir]
```

### Advanced Usage
```python
from langchain_rag import LangChainRAGPipeline

# Initialize pipeline
pipeline = LangChainRAGPipeline(model_path="your/model/path")

# Process image
results = pipeline.process_image("path/to/image.jpg")
```

## Migration Benefits

1. **Maintainability**: Modular components make updates easier
2. **Extensibility**: Easy to add new chain components
3. **Error Handling**: Robust fallback mechanisms
4. **Monitoring**: Better logging and debugging capabilities
5. **Integration**: Ready for Pinecone migration (next step)

## Next Steps

The LangChain integration serves as a foundation for:

1. **Pinecone Migration**: LangChain's vectorstore abstraction makes switching from FAISS to Pinecone seamless
2. **Kubernetes Deployment**: Chain-based architecture is easier to containerize and scale
3. **Advanced Features**: Memory, conversation chains, and multi-step reasoning

## Testing

Run the demo to test the integration:

```bash
# Make sure you're in the project root
cd /path/to/VLM-Tool-Recognition

# Run with your model and image
./rag-imp/run_langchain_demo.sh ./path/to/your/model ./path/to/test/image.jpg

# Or let it auto-detect test images
./rag-imp/run_langchain_demo.sh ./path/to/your/model
```

## Configuration

The system automatically:
- Creates knowledge base from HuggingFace dataset if not present
- Uses existing CSV knowledge base if available
- Handles GPU memory management
- Provides fallback error handling

## Performance

Expected improvements:
- **Better error recovery**: Graceful handling of VLM failures
- **Consistent outputs**: Structured prompt templates
- **Easier debugging**: Clear separation of pipeline stages
- **Memory efficiency**: Maintains your existing optimizations 