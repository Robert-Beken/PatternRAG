"""Vector store utility functions for PatternRAG.

This module provides functions for interacting with the vector database,
including initialization, embeddings generation, and retrieval.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from langchain_chroma import Chroma


def get_embeddings_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Get a HuggingFace embeddings model.
    
    Args:
        model_name (str): Name of the HuggingFace embeddings model
        
    Returns:
        HuggingFaceEmbeddings: Embeddings model
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)


def init_vector_store(persist_directory: str, embedding_model_name: str = "all-MiniLM-L6-v2",
                     create_new: bool = False) -> Optional[Chroma]:
    """
    Initialize or load the vector store.
    
    Args:
        persist_directory (str): Directory to persist the vector store
        embedding_model_name (str): Name of the embeddings model
        create_new (bool): Whether to create a new vector store
        
    Returns:
        Optional[Chroma]: Vector store instance or None if initialization fails
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Get embeddings model
        embeddings = get_embeddings_model(embedding_model_name)
        
        # Configure Chroma settings
        import chromadb
        client_settings = chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
        
        # If creating new or if store doesn't exist yet
        if create_new or not os.path.exists(os.path.join(persist_directory, 'index')):
            # Create an empty collection
            db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                client_settings=client_settings
            )
            db.persist()
            return db
        
        # Load existing vector store
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=client_settings
        )
    
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        return None


def add_documents_to_vector_store(db: Chroma, documents: List[Document], 
                                 batch_size: int = 500) -> bool:
    """
    Add documents to the vector store.
    
    Args:
        db (Chroma): Vector store
        documents (List[Document]): Documents to add
        batch_size (int): Size of batches for processing
        
    Returns:
        bool: Success status
    """
    if not db or not documents:
        return False
    
    try:
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            db.add_documents(batch)
            
        # Persist the database
        db.persist()
        return True
    
    except Exception as e:
        print(f"Error adding documents to vector store: {str(e)}")
        return False


def search_vector_store(db: Chroma, query: str, k: int = 10, 
                       filter_criteria: Optional[Dict[str, Any]] = None) -> List[Document]:
    """
    Search the vector store for relevant documents.
    
    Args:
        db (Chroma): Vector store
        query (str): Query string
        k (int): Number of results to return
        filter_criteria (Dict[str, Any], optional): Filters to apply
        
    Returns:
        List[Document]: Retrieved documents
    """
    if not db:
        return []
    
    try:
        # Create retriever with proper config
        search_kwargs = {"k": k}
        if filter_criteria:
            search_kwargs["filter"] = filter_criteria
            
        retriever = db.as_retriever(search_kwargs=search_kwargs)
        
        # Perform retrieval
        return retriever.get_relevant_documents(query)
    
    except Exception as e:
        print(f"Error searching vector store: {str(e)}")
        return []


def get_vector_store_stats(db: Chroma) -> Dict[str, Any]:
    """
    Get statistics about the vector store.
    
    Args:
        db (Chroma): Vector store
        
    Returns:
        Dict[str, Any]: Statistics about the vector store
    """
    if not db:
        return {"count": 0}
    
    try:
        # Get collection info
        count = db._collection.count()
        
        # Get collection name
        collection_name = "default"
        if hasattr(db, "_collection") and hasattr(db._collection, "name"):
            collection_name = db._collection.name
        
        return {
            "count": count,
            "collection": collection_name,
            "embedding_function": str(db._embedding_function)
        }
    except Exception as e:
        print(f"Error getting vector store stats: {str(e)}")
        return {"count": 0, "error": str(e)}


def batch_similarity_search(db: Chroma, queries: List[str], k: int = 10) -> Dict[str, List[Document]]:
    """
    Perform similarity search for multiple queries.
    
    Args:
        db (Chroma): Vector store
        queries (List[str]): List of queries
        k (int): Number of results to return per query
        
    Returns:
        Dict[str, List[Document]]: Results for each query
    """
    results = {}
    
    for query in queries:
        results[query] = search_vector_store(db, query, k)
    
    return results


def search_by_metadata(db: Chroma, metadata_filter: Dict[str, Any], 
                      k: int = 10) -> List[Document]:
    """
    Search for documents by metadata.
    
    Args:
        db (Chroma): Vector store
        metadata_filter (Dict[str, Any]): Metadata filter criteria
        k (int): Maximum number of results
        
    Returns:
        List[Document]: Retrieved documents
    """
    if not db:
        return []
    
    try:
        # Search using empty query but with metadata filter
        return db.similarity_search(
            query="",
            k=k,
            filter=metadata_filter
        )
    except Exception as e:
        print(f"Error searching by metadata: {str(e)}")
        return []


def delete_documents(db: Chroma, document_ids: List[str]) -> bool:
    """
    Delete documents from the vector store.
    
    Args:
        db (Chroma): Vector store
        document_ids (List[str]): IDs of documents to delete
        
    Returns:
        bool: Success status
    """
    if not db:
        return False
    
    try:
        # Delete documents
        db._collection.delete(ids=document_ids)
        db.persist()
        return True
    except Exception as e:
        print(f"Error deleting documents: {str(e)}")
        return False