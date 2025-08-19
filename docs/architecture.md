# Architecture Overview

This document provides a detailed overview of the PatternRAG system architecture, explaining how the different components work together to identify patterns and connections across documents.

## High-Level Architecture

PatternRAG consists of two main subsystems:

1. **Document Processing Subsystem** - Responsible for ingesting, analyzing, and indexing documents
2. **Retrieval and Analysis Subsystem** - Handles queries, pattern identification, and response generation

The following diagram illustrates the high-level architecture:

```
┌─────────────────┐               ┌─────────────────┐
│                 │               │                 │
│    Document     │               │  Query Analysis │
│   Processing    │◄──────────────│   & Pattern     │
│    Pipeline     │               │    Detection    │
│                 │               │                 │
└─────────────────┘               └─────────────────┘
        │                                  ▲
        │                                  │
        ▼                                  │
┌─────────────────┐               ┌─────────────────┐
│                 │               │                 │
│  Storage Layer  │◄──────────────│  API Service    │
│                 │               │                 │
└─────────────────┘               └─────────────────┘
```

## Component Architecture

### 1. Document Processing Subsystem

```
┌───────────────────────────────────────────────────────────┐
│                Document Processing Pipeline               │
├───────────────┬─────────────────────────┬─────────────────┤
│               │                         │                 │
│ Document      │ Document                │ Entity &        │
│ Loading       │ Chunking                │ Relationship    │
│               │                         │ Extraction      │
└───────────────┴─────────────────────────┴─────────────────┘
         │                 │                      │
         ▼                 ▼                      ▼
┌───────────────┐ ┌───────────────────┐ ┌──────────────────┐
│               │ │                   │ │                  │
│ Vector        │ │ Metadata          │ │ Knowledge        │
│ Database      │ │ Database          │ │ Graph            │
│ (ChromaDB)    │ │ (SQLite)          │ │ (NetworkX)       │
│               │ │                   │ │                  │
└───────────────┘ └───────────────────┘ └──────────────────┘
```

#### Key Components:

1. **Document Loading**
   - Supports multiple file formats (PDF, DOCX, TXT, HTML, etc.)
   - Extracts text content and basic metadata
   - Uses appropriate loaders for each document type

2. **Document Chunking**
   - Hierarchical chunking strategy:
     - Paragraph-level chunks (default 1000 chars)
     - Sentence-level chunks (default 250 chars)
   - Retains positional information and relationships between chunks

3. **Entity & Relationship Extraction**
   - Uses spaCy for named entity recognition (NER)
   - Extracts subject-verb-object relationships
   - Identifies key concepts and themes

4. **Storage Components**
   - **Vector Database (ChromaDB)**: Stores document embeddings for semantic search
   - **Metadata Database (SQLite)**: Stores document metadata, chunks, entities, and relationships
   - **Knowledge Graph (NetworkX)**: Represents entity relationships as a graph

### 2. Retrieval and Analysis Subsystem

```
┌───────────────────────────────────────────────────────────┐
│                      API Service                          │
└───────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                    Query Processing                       │
├────────────────┬─────────────────────┬───────────────────┤
│                │                     │                   │
│ Entity         │ Query               │ Knowledge Graph   │
│ Extraction     │ Expansion           │ Traversal         │
│                │                     │                   │
└────────────────┴─────────────────────┴───────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                    Retrieval Layer                        │
├────────────────┬─────────────────────┬───────────────────┤
│                │                     │                   │
│ Vector         │ Pattern-Guided      │ Entity-Based      │
│ Retrieval      │ Retrieval           │ Retrieval         │
│                │                     │                   │
└────────────────┴─────────────────────┴───────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                    Pattern Analysis                       │
├────────────────┬─────────────────────┬───────────────────┤
│                │                     │                   │
│ LLM Pattern    │ Connection          │ Response          │
│ Detection      │ Extraction          │ Generation        │
│                │                     │                   │
└────────────────┴─────────────────────┴───────────────────┘
```

#### Key Components:

1. **Query Processing**
   - **Entity Extraction**: Identifies entities in the user's query
   - **Query Expansion**: Generates multiple perspectives to broaden search
   - **Knowledge Graph Traversal**: Finds related entities using graph algorithms

2. **Retrieval Layer**
   - **Vector Retrieval**: Semantic search using embeddings
   - **Pattern-Guided Retrieval**: Uses predefined patterns to inform search
   - **Entity-Based Retrieval**: Retrieves documents related to graph entities

3. **Pattern Analysis**
   - **LLM Pattern Detection**: Specialized prompting for identifying connections
   - **Connection Extraction**: Extracts specific connections from LLM output
   - **Response Generation**: Synthesizes findings into a coherent response

### 3. API Layer

```
┌───────────────────────────────────────────────────────────┐
│                     FastAPI Application                    │
├────────────────┬─────────────────────┬───────────────────┤
│                │                     │                   │
│ Chat           │ Models              │ Health & Status   │
│ Completions    │ Endpoints           │ Endpoints         │
│                │                     │                   │
└────────────────┴─────────────────────┴───────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                    Service Integrations                    │
├────────────────┬─────────────────────┬───────────────────┤
│                │                     │                   │
│ PatternRAG     │ LLM API             │ Client            │
│ Processing     │ Integration         │ Applications      │
│                │                     │                   │
└────────────────┴─────────────────────┴───────────────────┘
```

#### Key Components:

1. **FastAPI Application**
   - **Chat Completions Endpoint**: OpenAI-compatible interface
   - **Models Endpoints**: Lists available models
   - **Health & Status Endpoints**: Provides system status

2. **Service Integrations**
   - **PatternRAG Processing**: Core pattern detection logic
   - **LLM API Integration**: Connects to external LLM services (Ollama, etc.)
   - **Client Applications**: Integrates with frontends like OpenWebUI

## Data Flow

### Document Ingestion Flow

1. Documents are loaded from the source directory
2. Documents are chunked into paragraphs and sentences
3. Embeddings are generated for each chunk and stored in ChromaDB
4. Entities and relationships are extracted and stored in SQLite
5. Knowledge graph is built from entities and relationships
6. Metadata is recorded in SQLite database

### Query Processing Flow

1. User sends a query to the API
2. Query is analyzed for entities and expanded
3. Multiple retrieval strategies are executed in parallel
4. Retrieved documents are deduplicated and prioritized
5. LLM analyzes documents for patterns and connections
6. Connections are extracted and organized
7. Final response is generated and returned to the user

## Storage Architecture

### Vector Database (ChromaDB)

The vector database stores document chunks with their embeddings for semantic retrieval:

```
Collection: "documents"
- Embeddings (vector representations of text)
- Document content (text chunks)
- Metadata:
  - source: document path
  - doc_id: unique identifier
  - title: document title
  - author: document author
  - chunk_type: "paragraph" or "sentence"
  - start_pos: position in original document
  - entities: comma-separated entity names
```

### Metadata Database (SQLite)

The SQLite database stores structured information about documents, chunks, entities and relationships:

**Tables:**
- `documents`: Document metadata (path, title, author, file_type, etc.)
- `chunks`: Individual text chunks with positions and entity references
- `entities`: Named entities with types and frequencies
- `relationships`: Entity relationships with types and frequencies

### Knowledge Graph (NetworkX)

The knowledge graph represents entities and their relationships:

- **Nodes**: Entities with attributes (type, weight)
- **Edges**: Relationships with attributes (type, weight)

## Component Dependencies

PatternRAG depends on several key libraries:

- **LangChain**: For RAG framework and document processing
- **ChromaDB**: Vector database for storing embeddings
- **spaCy**: NLP processing and entity extraction
- **NetworkX**: Knowledge graph representation
- **FastAPI**: API server
- **HuggingFace/SentenceTransformers**: Embedding models

## Extensibility Points

PatternRAG is designed to be extensible in several ways:

1. **Custom Loaders**: Add support for new document formats
2. **Custom Embedding Models**: Replace the default embedding model
3. **Custom Pattern Templates**: Define domain-specific patterns
4. **LLM Providers**: Connect to different LLM backends
5. **Storage Backends**: Replace ChromaDB with alternative vector stores

## Performance Considerations

See [Performance Tuning](performance.md) for detailed information on performance characteristics and optimization strategies.
