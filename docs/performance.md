# Performance Tuning

This guide provides recommendations for optimizing PatternRAG's performance for different workloads and hardware configurations.

## Performance Characteristics

PatternRAG has several performance-critical operations:

1. **Document Processing**: CPU and memory intensive, especially for large document collections
2. **Vector Search**: Depends on vector database size and embedding dimensions
3. **Knowledge Graph Operations**: Memory-bound for large graphs
4. **LLM Inference**: The most computationally expensive operation, depends on external LLM service

## System Requirements

### Minimum Requirements

- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 10GB SSD
- **Network**: Stable connection to LLM API service

### Recommended Requirements

- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 50GB+ SSD
- **GPU**: Optional, for local LLM inference

### Large-Scale Deployment Requirements

- **CPU**: 16+ cores
- **RAM**: 32GB+
- **Storage**: 200GB+ SSD
- **GPU**: Recommended for local LLM inference
- **Network**: High bandwidth for distributed operation

## Performance Bottlenecks

Understanding the common bottlenecks will help you optimize effectively:

### 1. Document Ingestion

**Bottlenecks:**
- NLP processing for entity extraction
- Embedding generation
- Database write operations

**Solutions:**
- Batch processing with appropriate batch sizes
- Use smaller spaCy models when processing large collections
- Reduce chunk overlap for faster processing
- Use multiprocessing for parallel document processing

### 2. Query Processing

**Bottlenecks:**
- Multiple retrieval operations in parallel
- Knowledge graph traversal for large graphs
- LLM inference time

**Solutions:**
- Limit the number of expanded queries
- Cache frequent query results
- Reduce retrieval batch size
- Optimize knowledge graph operations

### 3. Memory Usage

**Bottlenecks:**
- Loading large knowledge graphs into memory
- ChromaDB's in-memory operations
- Entity extraction for large documents

**Solutions:**
- Limit the size of the knowledge graph
- Use batched operations for ChromaDB
- Process very large documents in sections
- Implement memory-efficient graph algorithms

## Configuration Optimizations

### Optimizing for Speed

For faster performance at the cost of some accuracy:

```yaml
# Embeddings
embeddings_model_name: "paraphrase-MiniLM-L3-v2"  # Faster, less accurate model

# Chunking
chunk_size: 1500  # Larger chunks, fewer documents to process
chunk_overlap: 100  # Less overlap
sentence_chunk_size: 300  # Larger sentence chunks
sentence_chunk_overlap: 20  # Less overlap

# Processing
max_workers: 24  # More parallel workers (adjust based on CPU cores)
search_depth: 1  # Less deep graph traversal
```

### Optimizing for Memory Efficiency

For systems with limited RAM:

```yaml
# Batch processing
batch_size: 200  # Smaller batches to reduce memory pressure

# Knowledge graph 
prune_graph_nodes: true  # Enable graph pruning
max_graph_nodes: 10000  # Limit total nodes in graph

# Chunking
chunk_size: 800  # Smaller chunks
chunk_overlap: 100
```

### Optimizing for Accuracy

For highest quality results:

```yaml
# Embeddings
embeddings_model_name: "all-mpnet-base-v2"  # More accurate model

# Chunking
chunk_size: 800  # Smaller, more precise chunks
chunk_overlap: 200  # More overlap for context preservation
sentence_chunk_size: 200  # Smaller sentence chunks
sentence_chunk_overlap: 50  # More overlap

# Processing
search_depth: 3  # Deeper graph traversal
```

## Scaling Strategies

### Vertical Scaling

- Increase RAM to hold larger knowledge graphs
- Add CPU cores for parallel document processing
- Use faster SSD storage for database operations
- Add GPU for local LLM inference

### Horizontal Scaling

For very large document collections:

1. **Shard by Document Type**:
   - Split documents into logical categories
   - Process each category on separate instances
   - Merge results at query time

2. **Separate Processing and Query Services**:
   - Dedicated instances for document processing
   - Dedicated instances for query handling
   - Shared storage layer

3. **Distributed Vector Database**:
   - Use ChromaDB's distributed deployment options
   - Split vector index across multiple nodes
   - Implement load balancing for queries

## Query Optimization

### Cold Start Performance

The first query after starting the service will be slow due to:
- Loading the vector database
- Building the knowledge graph
- Warming up the LLM

**Solutions:**
- Implement a service warm-up script that sends a dummy query on startup
- Pre-load critical components during initialization
- Use persistent connections to the LLM service

### Caching Strategies

Implement caching at multiple levels:

1. **Query Result Caching**:
   ```python
   # Use LRU cache for query results
   @lru_cache(maxsize=100)
   def process_pattern_query(self, query):
       # Implementation
   ```

2. **Vector Search Caching**:
   ```python
   # Cache common vector search results
   @lru_cache(maxsize=50)
   def vector_search(self, query, k=10):
       # Implementation
   ```

3. **LLM Response Caching**:
   ```python
   # Cache LLM responses for identical prompts
   @lru_cache(maxsize=200)
   def query_llm(self, prompt):
       # Implementation
   ```

## Database Optimization

### ChromaDB Optimization

1. **Collection Strategies**:
   - Single collection with metadata filtering vs. multiple collections
   - Benchmark to find the optimal approach for your data

2. **Index Settings**:
   ```python
   # Configure optimal settings during creation
   client_settings = chromadb.config.Settings(
       anonymized_telemetry=False,
       chroma_db_impl="duckdb+parquet",  # For larger datasets
       persist_directory=persist_directory
   )
   ```

3. **Query Optimization**:
   - Use `where` filters to narrow search space
   - Adjust `n_results` based on query complexity
   - Set appropriate `include` parameters

### SQLite Optimization

1. **Indexing**:
   - Ensure proper indexes on frequently queried columns
   - Add indexes for entity and relationship queries

2. **Transaction Management**:
   - Use transaction batching for bulk operations
   - Commit at appropriate intervals

3. **Query Structure**:
   - Optimize complex joins with temporary tables
   - Use prepared statements for repeated queries

## LLM Provider Optimization

### Local Inference (Ollama)

For Ollama-based deployment:

1. **Model Selection**:
   - Choose appropriate model size for your hardware
   - Consider quantized models for resource-constrained environments

2. **Concurrent Requests**:
   - Adjust max_concurrent_requests based on your hardware
   - Benchmark different concurrency settings

3. **Context Window Usage**:
   - Optimize prompt structure to use context window efficiently
   - Prioritize most relevant information first in prompts

### Remote API Services

For cloud-based LLM services:

1. **Concurrent Connections**:
   - Implement connection pooling
   - Use proper rate limiting

2. **Request Batching**:
   - Batch similar requests when possible
   - Implement request queuing during high load

## Monitoring Performance

Implement monitoring to track key performance metrics:

1. **System Metrics**:
   - CPU, memory, disk usage
   - Network I/O for LLM API calls

2. **Application Metrics**:
   - Document processing throughput
   - Query latency
   - Cache hit rates
   - Vector search performance

3. **User Experience Metrics**:
   - End-to-end query time
   - Response quality scores

## Benchmarking

Benchmark your PatternRAG deployment:

```bash
# Document processing benchmark
python -m patternrag.benchmark.ingest --docs /path/to/test/docs --count 100

# Query performance benchmark 
python -m patternrag.benchmark.query --queries /path/to/test/queries --iterations 10
```

Analyze results to identify optimization opportunities.

## Memory Profile Optimization

For improved memory usage:

1. **Knowledge Graph Optimizations**:
   - Use `graph.remove_nodes_from()` to prune infrequent entities
   - Implement edge weight thresholds to limit relationships

2. **Embedding Storage**:
   - Use dimension reduction techniques if appropriate
   - Consider pruning redundant embeddings

3. **Chunk Management**:
   - Implement document retention policies
   - Periodically clean up and consolidate chunks

## Production Deployment Recommendations

### Docker Deployment

1. **Resource Allocation**:
   ```yaml
   # docker-compose.yml example
   services:
     patternrag:
       image: patternrag:latest
       deploy:
         resources:
           limits:
             cpus: '4'
             memory: 16G
   ```

2. **Volume Configuration**:
   ```yaml
   volumes:
     - ./data:/app/data:rw  # Persistent storage
     - ./config:/app/config:ro  # Read-only configuration
   ```

### Kubernetes Deployment

1. **Resource Requests and Limits**:
   ```yaml
   resources:
     requests:
       memory: "8Gi"
       cpu: "2"
     limits:
       memory: "16Gi"
       cpu: "4"
   ```

2. **Storage Configuration**:
   ```yaml
   volumeMounts:
     - name: data-volume
       mountPath: /app/data
   volumes:
     - name: data-volume
       persistentVolumeClaim:
         claimName: patternrag-data-pvc
   ```

## Real-World Performance Examples

| Document Collection Size | Hardware | Processing Time | Query Time |
|--------------------------|----------|----------------|------------|
| 1,000 pages (~300 docs)  | 8C/16GB  | 10-15 min      | 2-5 sec    |
| 10,000 pages (~3K docs)  | 16C/32GB | 2-3 hours      | 5-10 sec   |
| 100,000+ pages           | 32C/64GB | 1-2 days       | 10-20 sec  |

*Note: Actual performance will vary based on document complexity, hardware configuration, and LLM service performance.*

## Advanced Optimization Techniques

### Custom Embeddings Pipeline

For specialized domains, consider training custom embedding models:

```python
from sentence_transformers import SentenceTransformer, models

# Create custom embedding model with domain adaptation
word_embedding_model = models.Transformer('bert-base-uncased')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(pooling_model.get_sentence_embedding_dimension(), 384)

embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
embedding_model.fit(domain_specific_training_data)
```

### Hybrid Retrieval

Implement hybrid retrieval combining vector and keyword search:

```python
def hybrid_retrieval(query, k=10):
    # Vector search
    vector_results = vector_retriever.get_relevant_documents(query)
    
    # Keyword search (example using SQLite)
    keywords = extract_keywords(query)
    keyword_results = keyword_retriever.get_relevant_documents(keywords)
    
    # Combine and deduplicate results
    combined_results = merge_and_rank_results(vector_results, keyword_results)
    return combined_results[:k]
```

### Tiered Storage Architecture

For very large document collections:
- Hot tier: Frequently accessed documents in fast storage
- Warm tier: Less frequently accessed documents
- Cold tier: Archival documents loaded on demand

## Final Notes

- Performance optimization is an iterative process
- Regularly benchmark your deployment
- Monitor resource usage over time
- Balance performance with result quality based on your use case
