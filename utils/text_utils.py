"""Text processing utility functions for PatternRAG.

This module provides functions for text analysis, entity extraction,
relationship extraction, and other text-related utilities.
"""

import re
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
import spacy


def load_spacy_model(model_name: str = "en_core_web_sm") -> Optional[spacy.language.Language]:
    """
    Load a spaCy model, falling back to a smaller model if necessary.
    
    Args:
        model_name (str): Name of the spaCy model to load
        
    Returns:
        Optional[spacy.language.Language]: Loaded spaCy model or None if loading fails
    """
    try:
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        # Try to load a smaller model
        try:
            print(f"Model {model_name} not found. Trying en_core_web_sm...")
            return spacy.load("en_core_web_sm")
        except OSError:
            print("No spaCy model available. You may need to download one:")
            print("python -m spacy download en_core_web_sm")
            return None


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and normalizing Unicode.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    # Normalize Unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()


def extract_entities(text: str, nlp: Optional[spacy.language.Language] = None) -> List[Dict[str, Any]]:
    """
    Extract named entities from text.
    
    Args:
        text (str): Text to analyze
        nlp (spacy.language.Language, optional): Loaded spaCy model
        
    Returns:
        List[Dict[str, Any]]: List of extracted entities with metadata
    """
    if not text:
        return []
    
    # Load model if not provided
    if nlp is None:
        nlp = load_spacy_model()
        if nlp is None:
            return []
    
    # Process text
    doc = nlp(text[:10000] if len(text) > 10000 else text)  # Limit size for processing speed
    
    # Extract entities
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "EVENT", "DATE", "WORK_OF_ART"]:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
    
    return entities


def extract_relationships(text: str, nlp: Optional[spacy.language.Language] = None) -> List[Dict[str, Any]]:
    """
    Extract subject-verb-object relationships from text.
    
    Args:
        text (str): Text to analyze
        nlp (spacy.language.Language, optional): Loaded spaCy model
        
    Returns:
        List[Dict[str, Any]]: List of extracted relationships
    """
    if not text:
        return []
    
    # Load model if not provided
    if nlp is None:
        nlp = load_spacy_model()
        if nlp is None:
            return []
    
    # Process text
    doc = nlp(text[:10000] if len(text) > 10000 else text)
    
    # Extract relationships
    relationships = []
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
                        "type": token.lemma_,
                        "sentence": sent.text
                    })
    
    return relationships


def extract_key_phrases(text: str, nlp: Optional[spacy.language.Language] = None, 
                       top_n: int = 10) -> List[str]:
    """
    Extract key noun phrases from text.
    
    Args:
        text (str): Text to analyze
        nlp (spacy.language.Language, optional): Loaded spaCy model
        top_n (int): Maximum number of phrases to return
        
    Returns:
        List[str]: List of key phrases
    """
    if not text:
        return []
    
    # Load model if not provided
    if nlp is None:
        nlp = load_spacy_model()
        if nlp is None:
            return []
    
    # Process text
    doc = nlp(text[:10000] if len(text) > 10000 else text)
    
    # Extract noun phrases
    phrases = {}
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) > 1 and chunk.root.pos_ == "NOUN":
            normalized_chunk = normalize_text(chunk.text.lower())
            phrases[normalized_chunk] = phrases.get(normalized_chunk, 0) + 1
    
    # Sort by frequency
    sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N phrases
    return [phrase for phrase, _ in sorted_phrases[:top_n]]


def extract_themes(text: str, nlp: Optional[spacy.language.Language] = None) -> List[str]:
    """
    Extract main themes from text based on entities and noun phrases.
    
    Args:
        text (str): Text to analyze
        nlp (spacy.language.Language, optional): Loaded spaCy model
        
    Returns:
        List[str]: List of main themes
    """
    if not text:
        return []
    
    # Load model if not provided
    if nlp is None:
        nlp = load_spacy_model()
        if nlp is None:
            return []
    
    # Process first 5000 chars for efficiency
    doc = nlp(text[:5000] if len(text) > 5000 else text)
    
    # Extract themes from entities and noun phrases
    themes = []
    
    # Add entities
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART"]:
            themes.append(ent.text)
    
    # Add noun phrases
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) > 1 and chunk.root.pos_ == "NOUN":
            themes.append(chunk.text)
    
    # Get most common themes
    from collections import Counter
    top_themes = [item[0] for item in Counter(themes).most_common(10)]
    
    return top_themes


def split_text_into_chunks(text: str, chunk_size: int = 1000, 
                          chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ";", ",", " ", ""]
    )
    
    # Split text into chunks
    return text_splitter.split_text(text)


def extract_keywords(text: str, nlp: Optional[spacy.language.Language] = None, 
                    min_freq: int = 2, max_keywords: int = 15) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text (str): Text to analyze
        nlp (spacy.language.Language, optional): Loaded spaCy model
        min_freq (int): Minimum frequency for keywords
        max_keywords (int): Maximum number of keywords to return
        
    Returns:
        List[str]: List of keywords
    """
    if not text:
        return []
    
    # Load model if not provided
    if nlp is None:
        nlp = load_spacy_model()
        if nlp is None:
            return []
    
    # Process text
    doc = nlp(text)
    
    # Extract keywords (nouns, proper nouns, and adjectives)
    keywords = {}
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and len(token.text) > 2:
            lemma = token.lemma_.lower()
            keywords[lemma] = keywords.get(lemma, 0) + 1
    
    # Filter by minimum frequency
    keywords = {k: v for k, v in keywords.items() if v >= min_freq}
    
    # Sort by frequency
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    
    # Return top keywords
    return [keyword for keyword, _ in sorted_keywords[:max_keywords]]


def extract_dates(text: str, nlp: Optional[spacy.language.Language] = None) -> List[Dict[str, Any]]:
    """
    Extract dates from text.
    
    Args:
        text (str): Text to analyze
        nlp (spacy.language.Language, optional): Loaded spaCy model
        
    Returns:
        List[Dict[str, Any]]: List of dates with position information
    """
    if not text:
        return []
    
    # Load model if not provided
    if nlp is None:
        nlp = load_spacy_model()
        if nlp is None:
            return []
    
    # Process text
    doc = nlp(text)
    
    # Extract dates
    dates = []
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char
            })
    
    return dates


def find_similar_chunks(query: str, chunks: List[str], 
                       nlp: Optional[spacy.language.Language] = None, 
                       top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Find chunks most similar to a query using spaCy similarity.
    
    Args:
        query (str): Query string
        chunks (List[str]): List of text chunks
        nlp (spacy.language.Language, optional): Loaded spaCy model
        top_n (int): Number of results to return
        
    Returns:
        List[Tuple[str, float]]: List of (chunk, similarity_score) pairs
    """
    if not query or not chunks:
        return []
    
    # Load model if not provided
    if nlp is None:
        nlp = load_spacy_model()
        if nlp is None:
            return []
    
    # Process query
    query_doc = nlp(query)
    
    # Calculate similarity for each chunk
    similarities = []
    for chunk in chunks:
        chunk_doc = nlp(chunk)
        similarity = query_doc.similarity(chunk_doc)
        similarities.append((chunk, similarity))
    
    # Sort by similarity score
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Return top N results
    return sorted_similarities[:top_n]