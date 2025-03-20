"""Database utility functions for PatternRAG.

This module provides functions for interacting with the SQLite metadata database,
including initialization, storing document metadata, and retrieving information.
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


def init_metadata_db(metadata_db_path: str) -> None:
    """
    Initialize the SQLite database for metadata storage.
    
    Args:
        metadata_db_path (str): Path to the SQLite database file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(metadata_db_path), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(metadata_db_path)
    cursor = conn.cursor()
    
    # Check if relationship exists
    cursor.execute(
        "SELECT id, chunks FROM relationships WHERE source_id = ? AND target_id = ? AND type = ?",
        (source_id, target_id, relationship_type)
    )
    result = cursor.fetchone()
    
    if result:
        # Relationship exists, update frequency and add chunk
        rel_id, chunks_str = result
        chunks = chunks_str.split(",") if chunks_str else []
        chunks.append(str(chunk_id))
        
        cursor.execute(
            "UPDATE relationships SET frequency = frequency + 1, chunks = ? WHERE id = ?",
            (",".join(chunks), rel_id)
        )
        relationship_id = rel_id
    else:
        # Create new relationship
        cursor.execute(
            "INSERT INTO relationships (source_id, target_id, type, frequency, chunks) VALUES (?, ?, ?, ?, ?)",
            (source_id, target_id, relationship_type, 1, str(chunk_id))
        )
        relationship_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return relationship_id


def get_document_metadata(metadata_db_path: str, doc_id: int = None, path: str = None) -> Optional[Dict[str, Any]]:
    """
    Get document metadata from the database.
    
    Args:
        metadata_db_path (str): Path to the SQLite database file
        doc_id (int, optional): Document ID
        path (str, optional): Document path
        
    Returns:
        Optional[Dict[str, Any]]: Document metadata or None if not found
    """
    if not doc_id and not path:
        raise ValueError("Either doc_id or path must be provided")
    
    conn = sqlite3.connect(metadata_db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    cursor = conn.cursor()
    
    if doc_id:
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    else:
        cursor.execute("SELECT * FROM documents WHERE path = ?", (path,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_document_chunks(metadata_db_path: str, doc_id: int) -> List[Dict[str, Any]]:
    """
    Get all chunks for a document.
    
    Args:
        metadata_db_path (str): Path to the SQLite database file
        doc_id (int): Document ID
        
    Returns:
        List[Dict[str, Any]]: List of chunks
    """
    conn = sqlite3.connect(metadata_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY start_pos", (doc_id,))
    chunks = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return chunks


def extract_entities_from_db(metadata_db_path: str, query: str) -> List[str]:
    """
    Extract entities that might appear in the query using the metadata database.
    
    Args:
        metadata_db_path (str): Path to the SQLite database file
        query (str): User query
        
    Returns:
        List[str]: List of entities found in the query
    """
    conn = sqlite3.connect(metadata_db_path)
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


def get_top_entities(metadata_db_path: str, limit: int = 100) -> List[Tuple[str, str, int]]:
    """
    Get the most frequent entities from the database.
    
    Args:
        metadata_db_path (str): Path to the SQLite database file
        limit (int): Maximum number of entities to return
        
    Returns:
        List[Tuple[str, str, int]]: List of (name, type, frequency) tuples
    """
    conn = sqlite3.connect(metadata_db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT name, type, frequency FROM entities ORDER BY frequency DESC LIMIT ?",
        (limit,)
    )
    entities = cursor.fetchall()
    
    conn.close()
    return entities


def get_entity_relationships(metadata_db_path: str, entity_name: str) -> List[Dict[str, Any]]:
    """
    Get all relationships for an entity.
    
    Args:
        metadata_db_path (str): Path to the SQLite database file
        entity_name (str): Entity name
        
    Returns:
        List[Dict[str, Any]]: List of relationships
    """
    conn = sqlite3.connect(metadata_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get entity ID
    cursor.execute("SELECT id FROM entities WHERE name = ?", (entity_name,))
    result = cursor.fetchone()
    if not result:
        conn.close()
        return []
    
    entity_id = result[0]
    
    # Get relationships where this entity is source
    cursor.execute("""
        SELECT r.*, e.name as target_name, e.type as target_type
        FROM relationships r
        JOIN entities e ON r.target_id = e.id
        WHERE r.source_id = ?
    """, (entity_id,))
    outgoing = [dict(row) for row in cursor.fetchall()]
    
    # Get relationships where this entity is target
    cursor.execute("""
        SELECT r.*, e.name as source_name, e.type as source_type
        FROM relationships r
        JOIN entities e ON r.source_id = e.id
        WHERE r.target_id = ?
    """, (entity_id,))
    incoming = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    # Tag relationships as incoming or outgoing
    for rel in outgoing:
        rel['direction'] = 'outgoing'
    
    for rel in incoming:
        rel['direction'] = 'incoming'
    
    return outgoing + incoming.cursor()
    
    # Create document table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE,
        title TEXT,
        author TEXT,
        file_type TEXT,
        size INTEGER,
        chunk_count INTEGER,
        indexed_date TIMESTAMP,
        themes TEXT,
        categories TEXT
    )
    ''')
    
    # Create chunks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER,
        content TEXT,
        chunk_type TEXT,
        start_pos INTEGER,
        entities TEXT,
        timestamp TIMESTAMP,
        FOREIGN KEY (doc_id) REFERENCES documents (id)
    )
    ''')
    
    # Create entities table for graph connections
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        type TEXT,
        frequency INTEGER,
        last_seen TIMESTAMP
    )
    ''')
    
    # Create relationships table for graph connections
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER,
        target_id INTEGER,
        type TEXT,
        frequency INTEGER,
        chunks TEXT,
        FOREIGN KEY (source_id) REFERENCES entities (id),
        FOREIGN KEY (target_id) REFERENCES entities (id)
    )
    ''')
    
    conn.commit()
    conn.close()


def store_document_metadata(metadata_db_path: str, document_metadata: Dict[str, Any]) -> int:
    """
    Store document metadata in the SQLite database.
    
    Args:
        metadata_db_path (str): Path to the SQLite database file
        document_metadata (Dict[str, Any]): Document metadata to store
        
    Returns:
        int: Document ID in the database
    """
    conn = sqlite3.connect(metadata_db_path)
    cursor = conn.cursor()
    
    try:
        # Insert document metadata
        cursor.execute(
            "INSERT INTO documents (path, title, author, file_type, size, indexed_date, themes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                document_metadata["path"],
                document_metadata.get("title", "Unknown"),
                document_metadata.get("author", "Unknown"),
                document_metadata.get("file_type", ""),
                document_metadata.get("size", 0),
                datetime.now(),
                document_metadata.get("themes", "")
            )
        )
        
        # Get the document ID
        doc_id = cursor.lastrowid
        conn.commit()
        return doc_id
    
    except sqlite3.IntegrityError:
        # Document already exists, get its ID
        cursor.execute("SELECT id FROM documents WHERE path = ?", (document_metadata["path"],))
        result = cursor.fetchone()
        if result:
            return result[0]
        return -1
    
    finally:
        conn.close()


def update_document_chunk_count(metadata_db_path: str, doc_id: int, chunk_count: int) -> None:
    """
    Update the chunk count for a document.
    
    Args:
        metadata_db_path (str): Path to the SQLite database file
        doc_id (int): Document ID
        chunk_count (int): Number of chunks
    """
    conn = sqlite3.connect(metadata_db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE documents SET chunk_count = ? WHERE id = ?",
        (chunk_count, doc_id)
    )
    
    conn.commit()
    conn.close()


def store_chunk(metadata_db_path: str, doc_id: int, content: str, 
                chunk_type: str, start_pos: int, entities: str) -> int:
    """
    Store a document chunk in the database.
    
    Args:
        metadata_db_path (str): Path to the SQLite database file
        doc_id (int): Document ID
        content (str): Chunk content
        chunk_type (str): Type of chunk (paragraph, sentence)
        start_pos (int): Position in the original document
        entities (str): Comma-separated list of entities
        
    Returns:
        int: Chunk ID
    """
    conn = sqlite3.connect(metadata_db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO chunks (doc_id, content, chunk_type, start_pos, entities, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (doc_id, content, chunk_type, start_pos, entities, datetime.now())
    )
    
    chunk_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return chunk_id


def store_entity(metadata_db_path: str, name: str, entity_type: str) -> int:
    """
    Store an entity in the database or update its frequency if it exists.
    
    Args:
        metadata_db_path (str): Path to the SQLite database file
        name (str): Entity name
        entity_type (str): Entity type (PERSON, ORG, etc.)
        
    Returns:
        int: Entity ID
    """
    conn = sqlite3.connect(metadata_db_path)
    cursor = conn.cursor()
    
    try:
        # Try to insert the entity
        cursor.execute(
            "INSERT INTO entities (name, type, frequency, last_seen) VALUES (?, ?, ?, ?)",
            (name, entity_type, 1, datetime.now())
        )
        entity_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        # Entity already exists, update frequency
        cursor.execute(
            "UPDATE entities SET frequency = frequency + 1, last_seen = ? WHERE name = ?",
            (datetime.now(), name)
        )
        cursor.execute("SELECT id FROM entities WHERE name = ?", (name,))
        entity_id = cursor.fetchone()[0]
    
    conn.commit()
    conn.close()
    
    return entity_id


def store_relationship(metadata_db_path: str, source_id: int, target_id: int, 
                       relationship_type: str, chunk_id: int) -> int:
    """
    Store a relationship between entities in the database.
    
    Args:
        metadata_db_path (str): Path to the SQLite database file
        source_id (int): Source entity ID
        target_id (int): Target entity ID
        relationship_type (str): Type of relationship
        chunk_id (int): ID of the chunk where the relationship was found
        
    Returns:
        int: Relationship ID
    """
    conn = sqlite3.connect(metadata_db_path)
    cursor = conn