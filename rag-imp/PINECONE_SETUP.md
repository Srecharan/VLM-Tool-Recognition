# Pinecone Integration Setup

## Quick Start

1. **Get Pinecone API Key**
   - Sign up at [pinecone.io](https://pinecone.io)
   - Copy your API key from the dashboard

2. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your actual API key
   ```

3. **Install Dependencies**
   ```bash
   pip install -r rag_requirements.txt
   ```

4. **Test Integration**
   ```bash
   python test_pinecone.py
   ```

## Usage

### Basic Usage
```python
from pinecone_rag import PineconeRAGPipeline

pipeline = PineconeRAGPipeline(
    pinecone_api_key="your-api-key",
    index_name="tool-safety-index"
)

pipeline.load_knowledge_base()
results = pipeline.process_image("path/to/image.jpg")
```

### Command Line
```bash
python pinecone_rag.py --pinecone_api_key YOUR_KEY --image_path image.jpg
```

## Features

- **Cloud Vector Database**: Scalable Pinecone integration
- **LangChain Orchestration**: Structured workflow management  
- **Production Ready**: Environment variable configuration
- **Fast Retrieval**: Optimized similarity search

## Architecture

- **Vector Store**: Pinecone cloud database
- **Embeddings**: SentenceTransformer models
- **Framework**: LangChain for orchestration
- **Security**: Environment-based API key management 