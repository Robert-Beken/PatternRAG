# Configuration Guide

PatternRAG offers extensive configuration options to customize its behavior and adapt to different use cases. This guide explains all available configuration settings and how to adjust them for your needs.

## Configuration File

The main configuration file is located at `config/config.yaml`. You can create this file by copying the default template:

```bash
cp config/default_config.yaml config/config.yaml
```

You can also specify an alternative configuration file when running PatternRAG:

```bash
python -m patternrag.ingest --config path/to/your/config.yaml
python -m patternrag.service --config path/to/your/config.yaml
```

## Available Configuration Options

### Base Settings

```yaml
# Base directory for all data
base_dir: "./data"

# Source directory for documents
source_directory: "./documents"
```

| Setting | Description | Default |
|---------|-------------|---------|
| `base_dir` | Root directory for all PatternRAG data | `"./data"` |
| `source_directory` | Directory containing documents to process | `"./documents"` |

### LLM Settings

```yaml
# LLM settings
model: "llama3"  # Default model to use
llm_api_url: "http://localhost:11434"  # Ollama API or compatible service
```

| Setting | Description | Default |
|---------|-------------|---------|
| `model` | LLM model to use for pattern analysis | `"llama3"` |
| `llm_api_url` | URL for LLM API service | `"http://localhost:11434"` |

### Embedding Settings

```yaml
# Embedding settings
embeddings_model_name: "all-MiniLM-L6-v2"  # HuggingFace model for embeddings
```

| Setting | Description | Default |
|---------|-------------|---------|
| `embeddings_model_name` | HuggingFace model for document embeddings | `"all-MiniLM-L6-v2"` |

#### Supported Embedding Models

PatternRAG supports various embedding models from HuggingFace:

- `all-MiniLM-L6-v2` (Default) - Good balance of speed and accuracy
- `all-mpnet-base-v2` - Higher accuracy but slower
- `all-MiniLM-L12-v2` - More features than L6 with modest performance impact
- `paraphrase-MiniLM-L3-v2` - Fastest option, lower accuracy
- `sentence-t5-base` - Good for semantic similarity

### Processing Settings

```yaml
# Processing settings
max_workers: 16  # Number of worker threads for concurrent operations
search_depth: 2  # Depth of search for patterns
```

| Setting | Description | Default |
|---------|-------------|---------|
| `max_workers` | Number of worker threads for concurrent operations | `16` |
| `search_depth` | Depth of search for patterns | `2` |

### Chunking Settings

```yaml
# Chunking settings
chunk_size: 1000  # Character size for paragraph chunks
chunk_overlap: 200  # Overlap between paragraph chunks
sentence_chunk_size: 250  # Character size for sentence chunks
sentence_chunk_overlap: 25  # Overlap between sentence chunks
```

| Setting | Description | Default |
|---------|-------------|---------|
| `chunk_size` | Character size for paragraph chunks | `1000` |
| `chunk_overlap` | Overlap between paragraph chunks | `200` |
| `sentence_chunk_size` | Character size for sentence chunks | `250` |
| `sentence_chunk_overlap` | Overlap between sentence chunks | `25` |

#### Chunking Strategy Guide

- **Large documents with distinct sections**: Increase `chunk_size` to 1500-2000
- **Technical documents with detailed information**: Decrease `chunk_size` to 800-900 and increase `overlap`
- **Narrative content**: Larger `chunk_size` (1500+) to maintain context
- **For better pattern finding**: Keep `sentence_chunk_size` smaller to allow fine-grained connections

### Batch Settings

```yaml
# Batch settings
batch_size: 500  # Batch size for vector db operations
```

| Setting | Description | Default |
|---------|-------------|---------|
| `batch_size` | Batch size for vector db operations | `500` |

### Pattern-Finding Settings

```yaml
# Pattern-finding settings
custom_patterns:
  - "similarities between distinct domains of knowledge"
  - "recurring themes across different time periods"
  - "mathematical or structural patterns in complex systems"
  # Add more patterns...
```

The `custom_patterns` list guides the system in finding specific types of connections across documents. You can add, remove, or modify these patterns based on your use case.

## Environment Variables

PatternRAG also respects the following environment variables which override configuration file settings:

| Environment Variable | Corresponding Config | Default |
|----------------------|----------------------|---------|
| `PATTERN_BASE_DIR` | `base_dir` | `"./data"` |
| `PATTERN_SOURCE_DIR` | `source_directory` | `"./documents"` |
| `PATTERN_LLM_MODEL` | `model` | `"llama3"` |
| `PATTERN_LLM_API_URL` | `llm_api_url` | `"http://localhost:11434"` |
| `PATTERN_EMBEDDINGS_MODEL` | `embeddings_model_name` | `"all-MiniLM-L6-v2"` |
| `PATTERN_MAX_WORKERS` | `max_workers` | `16` |

## Configuration Examples

### Minimal Configuration

```yaml
base_dir: "./data"
source_directory: "./documents"
llm_api_url: "http://localhost:11434"
model: "llama2"
```

### Optimized for Speed

```yaml
embeddings_model_name: "paraphrase-MiniLM-L3-v2"
chunk_size: 1500
chunk_overlap: 150
sentence_chunk_size: 350
batch_size: 800
max_workers: 24
```

### Optimized for Accuracy

```yaml
embeddings_model_name: "all-mpnet-base-v2"
chunk_size: 800
chunk_overlap: 200
sentence_chunk_size: 200
sentence_chunk_overlap: 50
batch_size: 400
```

### Domain-Specific Pattern Finding (Scientific Research)

```yaml
custom_patterns:
  - "methodological similarities across scientific disciplines"
  - "recurring experimental designs in different fields"
  - "parallel theoretical developments across sciences"
  - "contradictory findings on similar phenomena"
  - "evidence convergence from multiple disciplines"
```

## Advanced Configuration

### Using a Custom Cache Directory

To specify a custom location for cache files (embeddings, model weights):

```yaml
cache_dir: "/path/to/cache"
```

### Custom Logging Configuration

```yaml
logging:
  level: "INFO"  # Can be DEBUG, INFO, WARNING, ERROR
  file: "logs/pattern_rag.log"
  max_size_mb: 10
  backup_count: 5
```

### Setting Up Authentication

```yaml
auth:
  enabled: true
  api_key: "your-secure-api-key"
```

## Applying Configuration Changes

After modifying the configuration:

1. If you changed document processing settings, run:
   ```bash
   python -m patternrag.ingest --config config/config.yaml
   ```

2. Restart the service to apply API and runtime changes:
   ```bash
   python -m patternrag.service --config config/config.yaml
   ```
