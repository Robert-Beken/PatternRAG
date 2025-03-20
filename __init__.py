"""PatternRAG: A pattern-finding Retrieval-Augmented Generation system.

PatternRAG is a system designed to identify non-obvious connections and
patterns across documents by combining vector search, knowledge graph
analysis, and LLM reasoning.
"""

__version__ = "1.0.0"

# Import key classes for easier access
from patternrag.ingest import PatternIngest
from patternrag.service import PatternRAGService