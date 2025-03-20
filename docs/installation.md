# Installation Guide

This guide provides detailed instructions for installing and setting up the PatternRAG system.

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB, 16GB+ recommended for large document collections
- **Storage**: 10GB+ free space (depends on your document collection size)
- **OS**: Linux, macOS, or Windows
- **Dependencies**: LLM provider (e.g., Ollama, OpenAI API, or other compatible service)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pattern-rag.git
cd pattern-rag
```

### 2. Create a Virtual Environment (Recommended)

Using a virtual environment is recommended to avoid dependency conflicts:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Or using conda
conda create -n patternrag python=3.10
conda activate patternrag
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all the required packages including:
- langchain and langchain-community
- chromadb (vector database)
- networkx (knowledge graph)
- spacy (NLP processing)
- fastapi and uvicorn (API server)
- huggingface_hub and sentence-transformers (embeddings)

### 4. Install the spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

For better entity recognition with larger documents, consider installing the larger model:

```bash
python -m spacy download en_core_web_lg
```

### 5. Create Required Directories

```bash
mkdir -p data/db data/metadata data/graph documents
```

### 6. Configure the System

Copy the default configuration file and modify it as needed:

```bash
cp config/default_config.yaml config/config.yaml
```

Edit the `config.yaml` file to set your:
- Document source directory
- LLM API endpoint
- Embedding model preference
- Other settings

### 7. Install as a Package (Optional)

If you want to use PatternRAG as a Python package:

```bash
pip install -e .
```

## Setting Up an LLM Provider

PatternRAG requires access to a language model API for pattern analysis and response generation.

### Option 1: Ollama (Local)

1. Install [Ollama](https://ollama.ai/) following their installation instructions
2. Pull a model:
   ```bash
   ollama pull llama2 # or another model of your choice
   ```
3. Start Ollama:
   ```bash
   ollama serve
   ```
4. Update your `config.yaml` to point to Ollama:
   ```yaml
   llm_api_url: "http://localhost:11434"
   model: "llama2" # or your chosen model
   ```

### Option 2: OpenAI-compatible API

If you have access to an OpenAI-compatible API:

1. Update your `config.yaml`:
   ```yaml
   llm_api_url: "https://your-openai-compatible-endpoint.com"
   model: "gpt-3.5-turbo" # or your preferred model
   ```
2. Set up authentication in environment variables:
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```

## Verifying Installation

To verify that your installation is working correctly:

1. Run a document ingestion test:
   ```bash
   python -m patternrag.ingest --config config/config.yaml --test
   ```

2. Test the API service:
   ```bash
   python -m patternrag.service --host localhost --port 8000 --config config/config.yaml
   ```

3. In another terminal, send a test query:
   ```bash
   curl -X POST http://localhost:8000/health
   ```

   You should receive a response indicating the service is running properly.

## Troubleshooting

### Common Issues

#### ChromaDB Installation Problems

If you encounter issues with ChromaDB:

```bash
pip uninstall chromadb
pip install chromadb==0.4.18  # Try a specific version
```

#### spaCy Model Loading

If spaCy has trouble loading models:

```bash
python -m spacy validate
```

#### Vector Database Connection

If you see errors connecting to the vector database:

```bash
rm -rf data/db/* # WARNING: This deletes your vector database
python -m patternrag.ingest --config config/config.yaml --new-vectordb
```

### Getting Help

If you encounter any other issues:
1. Check the [GitHub Issues](https://github.com/yourusername/pattern-rag/issues) for similar problems
2. Review your log files in the `logs` directory
3. Open a new issue with details about your environment and the error messages

## Next Steps

Now that you've installed PatternRAG, proceed to:
- [Adding Your First Documents](getting_started.md)
- [API Usage Examples](api_examples.md)
- [Configuration Options](configuration.md)
