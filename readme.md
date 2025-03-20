# PatternRAG

PatternRAG is an advanced Retrieval-Augmented Generation system designed to identify non-obvious connections and patterns across documents. It combines vector search, knowledge graph analysis, and LLM reasoning to discover relationships between concepts that might be missed by traditional RAG systems.

## 🌟 Key Features

- **Multi-perspective Retrieval**: Expands queries to look for connections across domains
- **Knowledge Graph Integration**: Uses entity and relationship extraction to build a knowledge graph
- **Pattern Detection**: Specialized prompting to identify meaningful patterns and connections
- **Hierarchical Chunking**: Processes documents at both paragraph and sentence levels
- **OpenAI-compatible API**: Drop-in replacement for OpenAI's chat completions API

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- An LLM API service like [Ollama](https://github.com/ollama/ollama)
- At least 8GB RAM (16GB+ recommended)
- 10GB+ storage space for document processing
- Docker installation of OpenWebUI or equivalent - for a front-end for the utility

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Robert-Beken/PatternRAG.git
cd pattern-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Create directories:
```bash
mkdir -p data/db data/metadata data/graph documents
```

5. Configure settings:
```bash
cp config/default_config.yaml config/config.yaml
# Edit config.yaml as needed
```

### Basic Usage

1. **Add documents**:
   
   Place your documents in the `documents` directory or specify a custom location in the config file.

2. **Process documents**:
```bash
python -m patternrag.ingest --config config/config.yaml
```

3. **Start the API service**:
```bash
python -m patternrag.service --config config/config.yaml
```

4. **Query the system**:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pattern-rag",
    "messages": [
      {"role": "user", "content": "What connections exist between mathematics and music?"}
    ]
  }'
```

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Options](docs/configuration.md)
- [API Reference](docs/api_reference.md)
- [Pattern Detection](docs/pattern_detection.md)
- [Architecture Overview](docs/architecture.md)
- [Performance Tuning](docs/performance.md)

## 💡 How It Works

PatternRAG works by:

1. **Document Processing**: Documents are loaded, chunked, and embedded into a vector database. Entities and relationships are extracted to build a knowledge graph.

2. **Query Analysis**: User queries are analyzed for entities and expanded to look for potential connections across domains.

3. **Multi-angle Retrieval**: 
   - Vector similarity search with expanded queries
   - Knowledge graph traversal to find related entities
   - Predefined pattern-based searches

4. **Pattern Identification**: An LLM analyzes retrieved documents to identify meaningful patterns and connections.

5. **Response Generation**: The system synthesizes findings into a coherent response that highlights discovered patterns.

## 🔄 Searching Modes

PatternRAG offers two search modes:

1. **Pattern Mode** (default): Full pattern-finding capabilities, query expansion, and connection analysis.

2. **Standard Mode**: Simple retrieval without extensive pattern finding. Activate by prefixing your query with "standard search".

## 🛠️ Advanced Configuration

PatternRAG is highly configurable. Key configuration options include:

- **Custom Pattern Templates**: Define patterns to guide the system's search
- **Embedding Model**: Choose the embedding model for vector search
- **Chunking Parameters**: Adjust document chunking for different document types
- **LLM Settings**: Configure which model to use for reasoning

See the [Configuration Guide](docs/configuration.md) for detailed options.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [ChromaDB](https://github.com/chroma-core/chroma) for vector database functionality
- [NetworkX](https://github.com/networkx/networkx) for knowledge graph capabilities
- [spaCy](https://github.com/explosion/spaCy) for NLP processing