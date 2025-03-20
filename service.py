#!/usr/bin/env python3
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import networkx as nx
import sqlite3
import json
import time
import os
import sys
import yaml
import logging
import requests
import pickle
from concurrent.futures import ThreadPoolExecutor, wait
from langchain_core.runnables import RunnableParallel 
from functools import lru_cache 
from threading import Lock
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pattern-rag-service')

class PatternRAGService:
    """
    PatternRAG Service - A system for finding semantic and structural patterns across documents.
    
    This service provides a FastAPI interface for querying a Pattern RAG system,
    which combines vector search, knowledge graph analysis, and LLM reasoning 
    to identify connections between concepts.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        "base_dir": "./data",
        "model": "llama3",
        "llm_api_url": "http://localhost:11434",
        "embeddings_model_name": "all-MiniLM-L6-v2",
        "max_workers": 16,
        "search_depth": 2,
        "custom_patterns": [
            "similarities between distinct domains of knowledge",
            "recurring themes across different time periods",
            "mathematical or structural patterns in complex systems",
            "symbolic parallels across different disciplines",
            "causal relationships between seemingly unrelated events"
        ]
    }
    
    def __init__(self, config_path=None):
        """
        Initialize the Pattern RAG Service.
        
        Args:
            config_path (str, optional): Path to the configuration file
        """
        # Load configuration
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self.config.update(user_config)
        
        # Set up paths
        self.base_dir = self.config["base_dir"]
        self.persist_directory = os.path.join(self.base_dir, "db")
        self.metadata_db = os.path.join(self.base_dir, "metadata/metadata.db")
        self.graph_file = os.path.join(self.base_dir, "graph/knowledge_graph.pickle")
        
        # Other configuration
        self.model = self.config["model"]
        self.llm_api_url = self.config["llm_api_url"]
        self.patterns = self.config["custom_patterns"]
        
        # Initialize components
        self.retriever = None
        self.embeddings = None
        self.knowledge_graph = None
        self.executor = ThreadPoolExecutor(max_workers=self.config["max_workers"])
        self.request_lock = Lock()
        
        # Initialize components
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize the necessary components for the service."""
        try:
            logger.info("Initializing embeddings...")
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=self.config["embeddings_model_name"])
            
            logger.info(f"Loading Chroma database from {self.persist_directory}...")
            from langchain_chroma import Chroma
            import chromadb
            
            # Try to initialize with default settings
            try:
                self.db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
                logger.info("Successfully loaded Chroma DB with default settings")
            except ValueError as e:
                logger.warning(f"Failed to initialize with default settings: {str(e)}. Trying with explicit client settings...")
                # Try with explicit settings
                client_settings = chromadb.config.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
                
                # Try with explicit client and more settings
                try:
                    chroma_client = chromadb.PersistentClient(
                        path=self.persist_directory,
                        settings=client_settings
                    )
                    
                    self.db = Chroma(
                        client=chroma_client,
                        embedding_function=self.embeddings,
                    )
                    logger.info("Successfully loaded Chroma DB with explicit client")
                except Exception as e2:
                    logger.error(f"Second attempt failed: {str(e2)}")
                    raise
            
            # Log collections and document count
            try:
                if hasattr(self.db, "_client"):
                    chroma_client = self.db._client
                    collection_names = chroma_client.list_collections()
                    logger.info(f"Chroma DB loaded with collections: {collection_names}")
                    
                    # Check if db has documents
                    if hasattr(self.db, "_collection") and hasattr(self.db._collection, "count"):
                        try:
                            doc_count = self.db._collection.count()
                            logger.info(f"DB contains {doc_count} documents")
                        except Exception as e:
                            logger.warning(f"Could not count documents directly: {str(e)}")
                    
                    # Try direct access through ChromaDB client
                    if collection_names:
                        try:
                            # In v0.6.0 we use the name directly
                            collection = chroma_client.get_collection(name=collection_names[0])
                            count = collection.count()
                            logger.info(f"Found {count} documents in collection {collection_names[0]} via direct access")
                            if count == 0:
                                logger.warning("Chroma DB collection exists but is empty")
                        except Exception as e:
                            logger.warning(f"Error accessing collection: {str(e)}")
                    else:
                        logger.warning("Chroma DB is empty or misconfigured - no collections found")
                else:
                    logger.warning("Could not access Chroma client directly, skipping collection count")
            except Exception as e:
                logger.warning(f"Error checking collections: {str(e)}")
                # Continue even if this fails - it's just informational logging
            
            # Setup retriever
            logger.info("Setting up retriever...")
            self.retriever = self.db.as_retriever(search_kwargs={"k": 10})
            
            # Load knowledge graph
            logger.info("Loading knowledge graph...")
            if os.path.exists(self.graph_file):
                with open(self.graph_file, 'rb') as f:
                    self.knowledge_graph = pickle.load(f)
                logger.info(f"Graph loaded with {self.knowledge_graph.number_of_nodes()} nodes and {self.knowledge_graph.number_of_edges()} edges")
            else:
                logger.warning("Knowledge graph file not found. Some features will be limited.")
                self.knowledge_graph = nx.Graph()
            
            logger.info("Initialization complete! Service ready.")
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            if 'db' in locals() and self.db is not None:
                logger.info("Vector store loaded, continuing with limited functionality")
                
                # Continue by setting up the retriever even if we had some errors
                try:
                    # Setup retriever
                    logger.info("Setting up retriever...")
                    self.retriever = self.db.as_retriever(search_kwargs={"k": 10})
                    logger.info("Retriever setup successful")
                except Exception as retriever_error:
                    logger.error(f"Failed to setup retriever: {str(retriever_error)}")
                    self.retriever = None
            else:
                logger.error("Critical error: Vector store failed to load")
                raise
    
    def find_related_entities(self, query_entities: List[str], top_n: int = 5) -> List[str]:
        """
        Find related entities in the knowledge graph.
        
        Args:
            query_entities (List[str]): List of entities from the query
            top_n (int): Number of top related entities to return
            
        Returns:
            List[str]: List of related entities
        """
        if not self.knowledge_graph or self.knowledge_graph.number_of_nodes() == 0:
            return []

        related_entities = {}

        for entity in query_entities:
            if entity in self.knowledge_graph:
                # Get direct connections
                for neighbor in self.knowledge_graph.neighbors(entity):
                    score = self.knowledge_graph[entity][neighbor].get('weight', 1)
                    if neighbor in related_entities:
                        related_entities[neighbor] += score
                    else:
                        related_entities[neighbor] = score

                # Use graph measures for broader connections
                try:
                    # PageRank to find important connected entities
                    pr = nx.pagerank(self.knowledge_graph, personalization={entity: 1.0})
                    for node, score in pr.items():
                        if node != entity and node not in query_entities:
                            if node in related_entities:
                                related_entities[node] += score * 10  # Scale up pagerank scores
                            else:
                                related_entities[node] = score * 10
                except:
                    # If PageRank fails, fall back to simpler methods
                    pass

        # Sort by score and return top N
        sorted_entities = sorted(related_entities.items(), key=lambda x: x[1], reverse=True)
        return [entity for entity, score in sorted_entities[:top_n]]
    
    def extract_entities_from_db(self, query: str) -> List[str]:
        """
        Extract entities that might appear in the query using the metadata database.
        
        Args:
            query (str): User query
            
        Returns:
            List[str]: List of entities found in the query
        """
        try:
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.cursor()

            # Get all entities from DB
            cursor.execute("SELECT name FROM entities ORDER BY frequency DESC LIMIT 1000")
            all_entities = [row[0] for row in cursor.fetchall()]
            conn.close()

            # Find which entities appear in the query
            query_lower = query.lower()
            found_entities = []
            for entity in all_entities:
                if entity.lower() in query_lower:
                    found_entities.append(entity)

            return found_entities
        except Exception as e:
            logger.error(f"Error extracting entities from DB: {str(e)}")
            return []
    
    def expand_query_for_patterns(self, query: str) -> List[str]:
        """
        Generate multiple queries that look for patterns and connections.
        
        Args:
            query (str): Original user query
            
        Returns:
            List[str]: List of expanded queries
        """
        try:
            expansion_prompt = f"""
            I'm searching for connections and patterns related to this query.
            Generate 3 expanded queries that help find related concepts across different domains.
            Focus on identifying meaningful relationships that might not be immediately obvious.

            Original query: {query}

            Format your response as three separate queries only, one per line.
            """

            # Get expanded queries from LLM
            response = requests.post(
                f"{self.llm_api_url}/api/generate",
                json={"model": self.model, "prompt": expansion_prompt, "stream": False},
                timeout=60
            )

            if response.status_code == 200:
                expanded_text = response.json().get('response', '')
                # Parse the output into individual queries
                expanded_queries = [line.strip() for line in expanded_text.split('\n') if line.strip()]
                # Add the original query and limit to max 4 queries total
                expanded_queries = [query] + expanded_queries[:3]
                return expanded_queries
            else:
                logger.warning(f"Error expanding query: {response.status_code}")
                return [query]
        except requests.exceptions.Timeout:
            logger.warning("Timeout in query expansion, using original query only")
            return [query]
        except Exception as e:
            logger.error(f"Error in query expansion: {str(e)}")
            return [query]
    
    @lru_cache(maxsize=100)
    def query_llm(self, prompt_text: str) -> str:
        """
        Query the language model.
        
        Args:
            prompt_text (str): Prompt to send to LLM
            
        Returns:
            str: Response from LLM
        """
        try:
            response = requests.post(
                f"{self.llm_api_url}/api/generate",
                json={"model": self.model, "prompt": prompt_text, "stream": False},
                timeout=300
            )

            if response.status_code == 200:
                return response.json().get('response', 'No response received')
            else:
                logger.error(f"LLM API error: {response.status_code}: {response.text}")
                return f"Error from model API: {response.status_code}"
        except requests.exceptions.Timeout:
            logger.error("Timeout querying LLM")
            return "Error: Request to the language model timed out. Please try a simpler query or try again later."
        except Exception as e:
            logger.error(f"Error querying LLM: {str(e)}")
            return f"Error: {str(e)}"
    
    def process_pattern_query(self, query: str, search_mode: str = "pattern", depth: int = 2) -> dict:
        """
        Process a query to find patterns and connections.
        
        Args:
            query (str): User query
            search_mode (str): 'pattern' for advanced pattern finding, 'standard' for basic RAG
            depth (int): Search depth
            
        Returns:
            dict: Results including answer, sources, connections, etc.
        """
        start_time = time.time()
        request_id = f"req-{int(start_time)}"
        
        # Results container
        results = {
            "query": query,
            "expanded_queries": [],
            "sources": [],
            "related_entities": [],
            "connections_found": [],
            "answer": "",
            "processing_time": 0
        }

        try:
            logger.info(f"[{request_id}] Processing pattern query: {query}")

            # 1. Extract entities from query
            query_entities = self.extract_entities_from_db(query)
            results["query_entities"] = query_entities
            logger.info(f"[{request_id}] Extracted entities: {query_entities}")

            # 2. Find related entities in knowledge graph
            related_entities = self.find_related_entities(query_entities)
            results["related_entities"] = related_entities
            logger.info(f"[{request_id}] Related entities: {related_entities}")

            # 3. Expand query with alternative perspectives
            if search_mode == "pattern":
                expand_start = time.time()
                expanded_queries = self.expand_query_for_patterns(query)
                logger.info(f"[{request_id}] Expanded queries in {time.time() - expand_start:.2f}s: {expanded_queries}")
                results["expanded_queries"] = expanded_queries
            else:
                expanded_queries = [query]
                results["expanded_queries"] = expanded_queries

            # 4. Gather documents from multiple search angles
            all_docs = []
            if self.retriever:
                queries = expanded_queries + related_entities[:5] + (self.patterns[:3] if search_mode == "pattern" else [])
                try:
                    batch_start = time.time()  # Profiling
                    retriever_batch = RunnableParallel({"doc": self.retriever})
                    batch_results = retriever_batch.invoke(queries)  # Batch process all queries
                    all_docs = [doc for sublist in batch_results["doc"] for doc in sublist]  # Flatten correctly
                    logger.info(f"[{request_id}] Retrieved {len(all_docs)} docs from batch query in {time.time() - batch_start:.2f}s")
                    
                    # Optionally log individual counts for debugging
                    start_idx = 0
                    for q in queries:
                        end_idx = start_idx + (10 if q in expanded_queries else 3 if q in related_entities else 2)
                        if end_idx <= len(all_docs):
                            logger.info(f"[{request_id}] Retrieved {len(all_docs[start_idx:end_idx])} docs for: {q}")
                        start_idx = end_idx
                except Exception as e:
                    logger.error(f"[{request_id}] Error in batch retrieval: {str(e)}")
                    # Fallback to sequential
                    for q in queries:
                        try:
                            docs = self.retriever.invoke(q)
                            all_docs.extend(docs[:10 if q in expanded_queries else 3 if q in related_entities else 2])
                            logger.info(f"[{request_id}] Retrieved {len(docs[:10 if q in expanded_queries else 3 if q in related_entities else 2])} docs for: {q}")
                        except Exception as e2:
                            logger.error(f"[{request_id}] Error retrieving docs for '{q}': {str(e2)}")

            # 5. Deduplicate and prioritize docs
            unique_docs = {}
            for doc in all_docs:
                doc_id = doc.metadata.get('doc_id', 'unknown')
                content = doc.page_content

                # Use a unique key combining doc_id and first 100 chars of content
                key = f"{doc_id}:{content[:100]}"

                if key not in unique_docs:
                    unique_docs[key] = doc

            # Convert back to list
            deduplicated_docs = list(unique_docs.values())

            # Sort documents by relevance (currently just using original order)
            # Future enhancement: implement better relevance scoring
            final_docs = deduplicated_docs[:15]  # Limit to top 15 for manageability
            
            # 6. Prepare sources for response
            sources = []
            for doc in final_docs:
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "content": doc.page_content,
                    "title": doc.metadata.get("title", "Unknown"),
                    "author": doc.metadata.get("author", "Unknown"),
                    "chunk_type": doc.metadata.get("chunk_type", "Unknown")
                })
            
            results["sources"] = sources
            logger.info(f"[{request_id}] Final sources count: {len(sources)}")
            
            # 7. Analyze patterns and connections
            if len(sources) > 0:
                # Create a prompt that specifically asks for pattern identification
                pattern_prompt = f"""
                Analyze the following information and identify meaningful patterns, connections, or correlations that might not be immediately obvious. 
                Focus on finding unexpected connections between different concepts, time periods, or domains of knowledge.
                Look for:
                1. Similarities across seemingly unrelated domains
                2. Recurring themes, symbols, or structures
                3. Causal relationships that span multiple fields
                4. Temporal patterns that suggest deeper connections
                5. Structural or conceptual parallels
                
                QUERY: {query}
                
                INFORMATION SOURCES:
                """
                for i, source in enumerate(sources):
                    pattern_prompt += f"\n\nSOURCE {i+1} - {source['title']} by {source['author']}:\n{source['content']}"
                
                # Submit pattern analysis to thread pool
                logger.info(f"[{request_id}] Requesting pattern analysis from LLM")
                futures = []
                pattern_start = time.time()  # Profiling
                pattern_future = self.executor.submit(self.query_llm, pattern_prompt)
                futures.append(pattern_future)
                
                # Wait for pattern analysis to finish
                wait([pattern_future])
                pattern_analysis = pattern_future.result()
                logger.info(f"[{request_id}] Pattern analysis took {time.time() - pattern_start:.2f}s")
                
                # Extract connections found
                connections_prompt = f"""
                Based on this pattern analysis, list the 3-5 most significant connections or patterns identified.
                Format each as a single sentence that clearly states the connection.
                
                Pattern analysis:
                {pattern_analysis}
                
                List of connections (format as numbered list):
                """
                
                connections_start = time.time()  # Profiling
                connections_future = self.executor.submit(self.query_llm, connections_prompt)
                futures.append(connections_future)
                
                # Wait for connections
                wait([connections_future])
                connections_text = connections_future.result()
                logger.info(f"[{request_id}] Connections extraction took {time.time() - connections_start:.2f}s")
                
                # Parse connections into list
                connections = [line.lstrip("0123456789.- ").strip() for line in connections_text.split('\n') if line.strip() and (line[0].isdigit() or line.startswith("- "))]
                results["connections_found"] = connections
                
                # 8. Generate final answer that highlights the patterns
                answer_prompt = f"""
                You are a pattern-finding assistant that explores connections across different domains of knowledge.
                QUERY: {query}
                Based on your analysis, you've found these potential connections:
                {connections_text}
                Now provide a clear, engaging response to the original query. Weave in the connections you've identified and cite specific evidence from the sources. Focus on the most intriguing possibilities while keeping it grounded in the data.
                """
                
                logger.info(f"[{request_id}] Generating final response")
                answer_start = time.time()  # Profiling
                answer_future = self.executor.submit(self.query_llm, answer_prompt)
                results["answer"] = answer_future.result()
                logger.info(f"[{request_id}] Final answer generation took {time.time() - answer_start:.2f}s")
            else:
                # Fallback if no sources found
                logger.info(f"[{request_id}] No sources found, querying LLM directly")
                answer_prompt = f"Query: {query}\nProvide a concise response as a search query completion."
                results["answer"] = self.query_llm(answer_prompt)
                results["sources"] = [{"source": "LLM", "content": "Direct LLM response"}]
                
        except Exception as e:
            logger.error(f"[{request_id}] Error in pattern processing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            results["error"] = str(e)
            results["answer"] = f"An error occurred while processing your query: {str(e)}"
        
        # Calculate processing time
        results["processing_time"] = time.time() - start_time
        logger.info(f"[{request_id}] Processing completed in {results['processing_time']:.2f} seconds")
        
        return results

# Create the FastAPI application with a lifespan manager to handle startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    # Create service instance on startup
    global service
    try:
        config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
        service = PatternRAGService(config_path)
    except Exception as e:
        logger.error(f"Error initializing service: {str(e)}")
    
    yield  # Application runs here
    
    # Shutdown logic
    logger.info("Shutting down...")
    if hasattr(service, 'executor'):
        service.executor.shutdown(wait=False)

# Initialize FastAPI with lifespan
app = FastAPI(title="PatternRAG API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
service = None

# API endpoint for chat completions (OpenAI-compatible interface)
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    """
    Chat completions endpoint compatible with OpenAI/Ollama format.
    
    This endpoint accepts chat messages in the OpenAI/Ollama format and processes
    them using PatternRAG if requested, or forwards to a LLM service otherwise.
    """
    start_time = time.time()
    request_id = f"req-{int(start_time)}"
    
    try:
        data = await request.json()
        requested_model = data.get("model", "pattern-rag")
        logger.info(f"[{request_id}] Requested model: {requested_model}")
        
        messages = data.get("messages", [])
        if not messages:
            return JSONResponse(content={"error": "No messages provided"}, status_code=400)
        
        query = next((msg["content"] for msg in reversed(messages) if msg.get("role") == "user" and msg.get("content")), None)
        if not query:
            return JSONResponse(content={"error": "No valid user query found"}, status_code=400)
        
        logger.info(f"[{request_id}] Received query: {query}")
        
        if requested_model == "pattern-rag":
            # Determine search mode from query content
            search_mode = "pattern"
            if "standard search" in query.lower() or "normal search" in query.lower():
                search_mode = "standard"
                query = query.replace("standard search", "").replace("normal search", "").strip()
            
            # Process the pattern query
            results = service.process_pattern_query(query, search_mode)
            
            # Format response for OpenAI compatibility
            response_json = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "model": "pattern-rag",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": results["answer"],
                        "context": {  # Optional: include extra context
                            "sources": results["sources"],
                            "connections": results["connections_found"],
                            "related_entities": results["related_entities"],
                            "metadata": {
                                "processing_time": f"{results['processing_time']:.2f}s",
                                "search_mode": search_mode,
                                "expanded_queries": results["expanded_queries"]
                            }
                        }
                    },
                    "finish_reason": "stop"
                }]
            }
            return response_json
        else:
            # Forward to LLM service if not pattern-rag
            llm_api_url = service.llm_api_url
            
            # Stream response if requested
            if data.get("stream", False):
                ollama_request = {
                    "model": requested_model,
                    "messages": [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
                    "stream": True
                }
                
                async def stream_response():
                    try:
                        with requests.post(f"{llm_api_url}/api/chat", json=ollama_request, stream=True, timeout=300) as response:
                            response.raise_for_status()
                            for chunk in response.iter_lines():
                                if chunk:
                                    chunk_data = json.loads(chunk.decode('utf-8'))
                                    if "message" in chunk_data:
                                        yield f"data: {json.dumps({'id': f'chatcmpl-{request_id}', 'object': 'chat.completion.chunk', 'model': requested_model, 'choices': [{'index': 0, 'delta': {'content': chunk_data['message']['content']}, 'finish_reason': None}]})}\n\n"
                                    if chunk_data.get("done"):
                                        yield f"data: {json.dumps({'id': f'chatcmpl-{request_id}', 'object': 'chat.completion.chunk', 'model': requested_model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                                        break
                    except Exception as e:
                        logger.error(f"[{request_id}] Streaming error: {str(e)}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
                return StreamingResponse(stream_response(), media_type="text/event-stream")
            else:
                # Non-streaming response
                ollama_request = {
                    "model": requested_model,
                    "messages": [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
                    "stream": False
                }
                
                try:
                    response = requests.post(f"{llm_api_url}/api/chat", json=ollama_request, timeout=300)
                    response.raise_for_status()
                    ollama_response = response.json()
                    
                    return {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion",
                        "model": requested_model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": ollama_response.get("message", {}).get("content", "")
                            },
                            "finish_reason": "stop"
                        }]
                    }
                except Exception as e:
                    logger.error(f"[{request_id}] Error forwarding to LLM: {str(e)}")
                    return JSONResponse(content={"error": str(e)}, status_code=500)
    
    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Simple models listing endpoint
@app.get("/models")
async def list_models():
    """List available models."""
    try:
        models = ["pattern-rag"]
        
        # Try to get models from LLM service
        if service:
            try:
                response = requests.get(f"{service.llm_api_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    llm_models = response.json().get("models", [])
                    for model in llm_models:
                        if "name" in model:
                            models.append(model["name"])
            except Exception as e:
                logger.warning(f"Could not fetch LLM models: {str(e)}")
        
        return {"models": models}
    except Exception as e:
        return {"error": str(e), "models": ["pattern-rag"]}

# OpenAI compatible models endpoint
@app.get("/v1/models")
async def get_openai_models():
    """List models in OpenAI format."""
    try:
        models_data = []
        
        # Always add pattern-rag
        models_data.append({
            "id": "pattern-rag",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "user"
        })
        
        # Try to get models from LLM service
        if service:
            try:
                response = requests.get(f"{service.llm_api_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    llm_models = response.json().get("models", [])
                    for model in llm_models:
                        if "name" in model:
                            models_data.append({
                                "id": model["name"],
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "llm-service"
                            })
            except Exception as e:
                logger.warning(f"Could not fetch LLM models: {str(e)}")
        
        return {
            "object": "list",
            "data": models_data
        }
    except Exception as e:
        return {
            "object": "list",
            "data": [{
                "id": "pattern-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user"
            }]
        }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "service": "PatternRAG",
        "timestamp": time.time()
    }

# Main execution
def main():
    """Main function to start the service."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="PatternRAG Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Configuration file")
    args = parser.parse_args()
    
    # Set config path environment variable
    os.environ["CONFIG_PATH"] = args.config
    
    # Start the service
    uvicorn.run("service:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()