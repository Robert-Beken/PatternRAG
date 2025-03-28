#!/usr/bin/env python3
import os
import glob
import spacy
import sqlite3
import networkx as nx
from typing import List, Dict, Any
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from datetime import datetime
import json
import yaml
import argparse

from langchain_community.document_loaders import (
    CSVLoader, PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader, UnstructuredHTMLLoader, UnstructuredEPubLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

class PatternIngest:
    """
    Document ingestion system for Pattern RAG.
    
    This class handles the processing of documents into a form suitable for
    the Pattern RAG system, including:
    - Document loading and text extraction
    - Metadata extraction
    - Entity and relationship extraction
    - Knowledge graph creation
    - Vector database creation
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        "base_dir": "./data",
        "source_directory": "./documents",
        "embeddings_model_name": "all-MiniLM-L6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "sentence_chunk_size": 250,
        "sentence_chunk_overlap": 25,
        "spacy_model": "en_core_web_sm",
        "batch_size": 500
    }
    
    # Map file extensions to document loaders and their arguments
    LOADER_MAPPING = {
        ".csv": (CSVLoader, {}),
        ".doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".epub": (UnstructuredEPubLoader, {}),
        ".html": (UnstructuredHTMLLoader, {}),
        ".md": (UnstructuredMarkdownLoader, {}),
        ".pdf": (PyMuPDFLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"}),
    }
    
    def __init__(self, config_path=None):
        """
        Initialize the PatternIngest with configuration settings.
        
        Args:
            config_path (str, optional): Path to configuration YAML file.
        """
        # Load config
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
        self.source_directory = self.config["source_directory"]
        
        # Create directories if they don't exist
        for dir_path in [self.persist_directory, 
                         os.path.dirname(self.metadata_db), 
                         os.path.dirname(self.graph_file)]:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created/verified directory: {dir_path}")
        
        # Initialize spaCy for NLP tasks
        self.load_spacy_model()
        
        # Initialize components
        self.init_metadata_db()
        self.graph = self.init_knowledge_graph()
        
    def load_spacy_model(self):
        """Load the appropriate spaCy model based on availability."""
        try:
            self.nlp = spacy.load(self.config["spacy_model"])
            print(f"Loaded spaCy model {self.config['spacy_model']}")
        except:
            print("Specified spaCy model not available. Installing smaller model...")
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
                print("Loaded spaCy model en_core_web_sm")
            except Exception as e:
                print(f"Error loading spaCy model: {str(e)}")
                raise
    
    def init_metadata_db(self):
        """Initialize SQLite database schema for metadata storage."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        
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
        print("Database schema initialized or verified")
    
    def init_knowledge_graph(self):
        """Initialize or load existing knowledge graph with error handling."""
        if os.path.exists(self.graph_file) and os.path.getsize(self.graph_file) > 0:
            try:
                with open(self.graph_file, 'rb') as f:
                    graph = pickle.load(f)
                    print(f"Loaded existing knowledge graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
                    return graph
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Error loading knowledge graph: {str(e)}")
                print("Creating new knowledge graph")
                return nx.Graph()
        print("Creating new knowledge graph")
        return nx.Graph()
    
    def load_single_document(self, file_path: str) -> List[Document]:
        """
        Load a single document using the appropriate loader.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            List[Document]: List of document chunks
        """
        ext = "." + file_path.rsplit(".", 1)[-1].lower()
        if ext in self.LOADER_MAPPING:
            loader_class, loader_args = self.LOADER_MAPPING[ext]
            try:
                loader = loader_class(file_path, **loader_args)
                return loader.load()
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                return []
        
        print(f"Unsupported file extension '{ext}' for {file_path}")
        return []
    
    def extract_document_metadata(self, content, file_path):
        """
        Extract metadata from document content.
        
        Args:
            content (str): Document text content
            file_path (str): Path to the document file
            
        Returns:
            dict: Extracted metadata
        """
        try:
            # Process first 5000 chars for efficiency
            doc = self.nlp(content[:5000] if len(content) > 5000 else content)
            
            # Extract potential title (first sentence)
            title = next((sent.text for sent in doc.sents), os.path.basename(file_path))
            
            # Extract potential author (look for patterns like "by Author Name")
            author_patterns = ["by ", "author:", "written by"]
            author = "Unknown"
            for pattern in author_patterns:
                if pattern in content.lower():
                    idx = content.lower().find(pattern) + len(pattern)
                    end_idx = content.find("\n", idx)
                    if end_idx > 0:
                        author = content[idx:end_idx].strip()
                        break
            
            # Extract main themes based on key noun phrases and entities
            themes = []
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART"]:
                    themes.append(ent.text)
            
            # Get significant noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) > 1 and chunk.root.pos_ == "NOUN":
                    themes.append(chunk.text)
            
            # Limit to top 10 most common themes
            from collections import Counter
            top_themes = [item[0] for item in Counter(themes).most_common(10)]
            
            return {
                "title": title,
                "author": author,
                "themes": ",".join(top_themes),
                "file_type": os.path.splitext(file_path)[1],
                "size": os.path.getsize(file_path)
            }
        except Exception as e:
            print(f"Error extracting metadata from {file_path}: {str(e)}")
            return {
                "title": os.path.basename(file_path),
                "author": "Unknown",
                "themes": "",
                "file_type": os.path.splitext(file_path)[1],
                "size": os.path.getsize(file_path)
            }
    
    def extract_graph_elements(self, text):
        """
        Extract entities and relationships for knowledge graph.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            tuple: (entities, relationships)
        """
        try:
            doc = self.nlp(text[:10000] if len(text) > 10000 else text)  # Limit size for processing speed
            
            entities = []
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "EVENT", "DATE", "WORK_OF_ART"]:
                    entities.append({
                        "text": ent.text,
                        "type": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
            
            relationships = []
            # Extract subject-verb-object relationships
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ == "ROOT" and token.pos_ == "VERB":
                        subj = None
                        obj = None
                        
                        for child in token.children:
                            if child.dep_ in ["nsubj", "nsubjpass"] and not subj:
                                subj = child
                            elif child.dep_ in ["dobj", "pobj"] and not obj:
                                obj = child
                        
                        if subj and obj:
                            relationships.append({
                                "source": subj.text,
                                "target": obj.text,
                                "type": token.lemma_
                            })
            
            return entities, relationships
        except Exception as e:
            print(f"Error extracting graph elements: {str(e)}")
            return [], []
    
    def process_documents(self, ignored_files: List[str] = [], force_reprocess: bool = False) -> List[Document]:
        """
        Process documents from the source directory.
        
        Args:
            ignored_files (List[str]): List of files to ignore
            force_reprocess (bool): Whether to force reprocessing of all files
            
        Returns:
            List[Document]: List of processed document chunks
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        
        # Find all files in source directory that match supported extensions
        all_files = []
        for ext in self.LOADER_MAPPING:
            all_files.extend(glob.glob(os.path.join(self.source_directory, f"**/*{ext}"), recursive=True))
        
        # Get already processed files
        cursor.execute("SELECT id, path FROM documents WHERE chunk_count IS NOT NULL")
        processed_files_rows = cursor.fetchall()
        processed_files_dict = {row[1]: row[0] for row in processed_files_rows}
        
        # Filter files based on what we want to process
        if not force_reprocess:
            # Only process new files
            filtered_files = [f for f in all_files if f not in ignored_files and f not in processed_files_dict]
            print(f"Found {len(filtered_files)} new documents to process")
        else:
            # Process all files (but still handle existing documents properly)
            filtered_files = [f for f in all_files if f not in ignored_files]
            print(f"Found {len(filtered_files)} documents to process, including {len([f for f in filtered_files if f in processed_files_dict])} already processed")
        
        if not filtered_files:
            print("No documents to process")
            conn.close()
            return []
        
        # Process files
        all_docs = []
        total_chunks = 0  # Track total chunks processed
        
        with tqdm(total=len(filtered_files), desc="Processing documents") as pbar:
            for file_path in filtered_files:
                try:
                    # Check if this file was already processed
                    already_processed = file_path in processed_files_dict
                    
                    # If reprocessing, clear existing chunks
                    if already_processed and force_reprocess:
                        doc_id = processed_files_dict[file_path]
                        cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                        conn.commit()
                        print(f"Cleared existing chunks for document ID {doc_id} ({file_path})")
                    
                    # Load document
                    docs = self.load_single_document(file_path)
                    if not docs:
                        pbar.update(1)
                        continue
                    
                    # Combine all text from document for metadata extraction
                    full_text = " ".join([doc.page_content for doc in docs])
                    
                    # Extract metadata
                    metadata = self.extract_document_metadata(full_text, file_path)
                    
                    # Get or create document ID
                    if already_processed:
                        doc_id = processed_files_dict[file_path]
                        # Update metadata
                        cursor.execute(
                            "UPDATE documents SET title = ?, author = ?, file_type = ?, size = ?, indexed_date = ?, themes = ? WHERE id = ?",
                            (metadata["title"], metadata["author"], metadata["file_type"], 
                             metadata["size"], datetime.now(), metadata["themes"], doc_id)
                        )
                    else:
                        try:
                            # Insert document
                            cursor.execute(
                                "INSERT INTO documents (path, title, author, file_type, size, indexed_date, themes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (file_path, metadata["title"], metadata["author"], metadata["file_type"], 
                                 metadata["size"], datetime.now(), metadata["themes"])
                            )
                            doc_id = cursor.lastrowid
                        except sqlite3.IntegrityError:
                            # If there's a race condition, get the existing ID
                            cursor.execute("SELECT id FROM documents WHERE path = ?", (file_path,))
                            doc_id = cursor.fetchone()[0]
                    
                    # Split into chunks with hierarchical approach
                    paragraph_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.config["chunk_size"],
                        chunk_overlap=self.config["chunk_overlap"],
                        separators=["\n\n", "\n", ". ", "! ", "? ", ";", ",", " ", ""]
                    )
                    sentence_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.config["sentence_chunk_size"],
                        chunk_overlap=self.config["sentence_chunk_overlap"],
                        separators=[". ", "! ", "? ", ";", "\n", ",", " ", ""]
                    )
                    
                    processed_chunks = []
                    
                    # First split into paragraphs
                    for i, doc in enumerate(docs):
                        # Update base metadata
                        base_metadata = {
                            "source": file_path,
                            "doc_id": str(doc_id),  # Convert to string for Chroma compatibility
                            "title": metadata["title"],
                            "author": metadata["author"],
                            "chunk_type": "paragraph"
                        }
                        
                        # Split into paragraphs
                        paragraphs = paragraph_splitter.split_text(doc.page_content)
                        
                        for j, para in enumerate(paragraphs):
                            # Find position in original text
                            start_pos = doc.page_content.find(para) if len(para) < 1000 else 0
                            para_metadata = {**base_metadata, "start_pos": str(start_pos)}
                            
                            # Extract entities and relationships
                            entities, relationships = self.extract_graph_elements(para)
                            
                            # Store entities in metadata (as string for Chroma compatibility)
                            entity_names = [e["text"] for e in entities]
                            para_metadata["entities"] = ",".join(entity_names)
                            
                            # Store paragraph chunk with enhanced metadata
                            para_doc = Document(page_content=para, metadata=para_metadata)
                            processed_chunks.append(para_doc)
                            
                            # Add entities and relationships to database and graph
                            for entity in entities:
                                # Add to database
                                cursor.execute(
                                    "INSERT OR IGNORE INTO entities (name, type, frequency, last_seen) VALUES (?, ?, ?, ?)",
                                    (entity["text"], entity["type"], 1, datetime.now())
                                )
                                cursor.execute(
                                    "UPDATE entities SET frequency = frequency + 1, last_seen = ? WHERE name = ?",
                                    (datetime.now(), entity["text"])
                                )
                                
                                # Add to graph
                                if not self.graph.has_node(entity["text"]):
                                    self.graph.add_node(entity["text"], type=entity["type"])
                                else:
                                    # Increment weight if node exists
                                    weight = self.graph.nodes[entity["text"]].get("weight", 0) + 1
                                    self.graph.nodes[entity["text"]]["weight"] = weight
                            
                            # Add relationships to graph
                            for rel in relationships:
                                # Add to database
                                source_id = cursor.execute("SELECT id FROM entities WHERE name = ?", (rel["source"],)).fetchone()
                                target_id = cursor.execute("SELECT id FROM entities WHERE name = ?", (rel["target"],)).fetchone()
                                
                                if source_id and target_id:
                                    cursor.execute(
                                        "INSERT OR IGNORE INTO relationships (source_id, target_id, type, frequency, chunks) VALUES (?, ?, ?, ?, ?)",
                                        (source_id[0], target_id[0], rel["type"], 1, str(doc_id))
                                    )
                                    cursor.execute(
                                        "UPDATE relationships SET frequency = frequency + 1 WHERE source_id = ? AND target_id = ? AND type = ?",
                                        (source_id[0], target_id[0], rel["type"])
                                    )
                                
                                # Add to graph if both entities exist
                                if self.graph.has_node(rel["source"]) and self.graph.has_node(rel["target"]):
                                    if self.graph.has_edge(rel["source"], rel["target"]):
                                        # Increment weight if edge exists
                                        weight = self.graph[rel["source"]][rel["target"]].get("weight", 0) + 1
                                        self.graph[rel["source"]][rel["target"]]["weight"] = weight
                                    else:
                                        self.graph.add_edge(rel["source"], rel["target"], type=rel["type"], weight=1)
                            
                            # Also split into smaller sentence chunks for detailed retrieval
                            sentences = sentence_splitter.split_text(para)
                            for sentence in sentences:
                                if len(sentence) < 30:  # Skip very short sentences
                                    continue
                                    
                                # Create a document chunk for this sentence
                                sent_metadata = {
                                    **base_metadata,
                                    "chunk_type": "sentence",
                                    "start_pos": str(para.find(sentence)),
                                    "parent_chunk": para[:100] + "..." if len(para) > 100 else para
                                }
                                sent_doc = Document(page_content=sentence, metadata=sent_metadata)
                                processed_chunks.append(sent_doc)
                    
                    # Update document with chunk count
                    cursor.execute(
                        "UPDATE documents SET chunk_count = ? WHERE id = ?",
                        (len(processed_chunks), doc_id)
                    )
                    
                    all_docs.extend(processed_chunks)
                    total_chunks += len(processed_chunks)
                    print(f"Document '{metadata['title']}' processed into {len(processed_chunks)} chunks. Total chunks so far: {total_chunks}")
                    
                    # Commit every 10 documents to avoid data loss
                    if len(all_docs) % 10 == 0:
                        conn.commit()
                        # Save graph periodically
                        with open(self.graph_file, 'wb') as f:
                            pickle.dump(self.graph, f)
                
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                pbar.update(1)
        
        # Final commit
        conn.commit()
        conn.close()
        
        # Save final graph
        with open(self.graph_file, 'wb') as f:
            pickle.dump(self.graph, f)
        
        print(f"Processed {len(filtered_files)} documents into {len(all_docs)} chunks")
        return all_docs
    
def create_vector_store(self, chunks, force_new=False):
    """
    Create or update vector store with document chunks.
    
    Args:
        chunks (List[Document]): List of document chunks
        force_new (bool): Whether to create a new vector store
        
    Returns:
        Chroma: Vector store object
    """
    if not chunks:
        print("No chunks to add to vector database")
        return None
    
    # Create embeddings
    print(f"Initializing embeddings model: {self.config['embeddings_model_name']}")
    embeddings = HuggingFaceEmbeddings(model_name=self.config['embeddings_model_name'])
    
    # Set client settings for Chroma
    import chromadb
    print(f"Using ChromaDB version: {chromadb.__version__}")
    
    client_settings = chromadb.config.Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
    
    batch_size = self.config["batch_size"]  # Size of batches for processing
    
    # Check if vectorstore exists and we're not forcing a new one
    if os.path.exists(os.path.join(self.persist_directory, 'index')) and not force_new:
        # Update existing vectorstore in batches
        print(f"Opening existing ChromaDB at {self.persist_directory}")
        try:
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
                client_settings=client_settings
            )
            print("Successfully opened existing Chroma database")
            
            total_added = 0
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                print(f"Adding batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
                
                db.add_documents(batch)
                total_added += len(batch)
                print(f"Batch {i//batch_size + 1} added successfully. Total added: {total_added}")
                
            return db
            
        except Exception as e:
            print(f"Error updating existing vector store: {str(e)}")
            if force_new:
                print("Forcing creation of new vector store as requested")
            else:
                return None
    else:
        # Create new vectorstore
        print(f"Creating new Chroma DB at {self.persist_directory}")
        os.makedirs(self.persist_directory, exist_ok=True)
        
        try:
            # Start with a small batch to establish the database
            first_batch_size = min(500, len(chunks))
            first_batch = chunks[:first_batch_size]
            print(f"Creating initial database with first {first_batch_size} chunks...")
            
            # Create the initial database
            db = Chroma.from_documents(
                first_batch,
                embeddings,
                persist_directory=self.persist_directory,
                client_settings=client_settings
            )
            print(f"Initial database created with {first_batch_size} chunks.")
            
            # Note: removed db.persist() calls as they're not needed with current Chroma version
            # Persistence is handled automatically when persist_directory is specified
            
            # Process remaining documents in batches
            total_added = first_batch_size
            for i in range(first_batch_size, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                print(f"Adding batch {(i-first_batch_size)//batch_size + 1}/{(len(chunks)-first_batch_size-1)//batch_size + 1} ({len(batch)} chunks)...")
                db.add_documents(batch)
                total_added += len(batch)
                print(f"Batch added. Total chunks in database: {total_added}")
                # Removed db.persist() call here as well
                
            return db

        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return None
            
def main():
    """Main function for running document ingestion process."""
    parser = argparse.ArgumentParser(description="Pattern RAG Document Ingestion")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all documents")
    parser.add_argument("--new-vectordb", action="store_true", help="Create a new vector database")
    args = parser.parse_args()
    
    # Initialize ingestion system
    ingest = PatternIngest(config_path=args.config)
    
    # Process documents
    print("Starting document processing...")
    chunks = ingest.process_documents(force_reprocess=args.force)
    
    # Create or update vector store
    print("Creating/updating vector store...")
    db = ingest.create_vector_store(chunks, force_new=args.new_vectordb)
    
    if db:
        print("Vector store creation/update successful!")
    else:
        print("Vector store operation failed or no documents were processed.")


if __name__ == "__main__":
    main()
