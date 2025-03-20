"""Knowledge graph utility functions for PatternRAG.

This module provides functions for creating and manipulating the knowledge
graph used for finding connections between entities.
"""

import os
import pickle
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple


def init_knowledge_graph(graph_file: str) -> nx.Graph:
    """
    Initialize or load the knowledge graph.
    
    Args:
        graph_file (str): Path to the graph file
        
    Returns:
        nx.Graph: Knowledge graph
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(graph_file), exist_ok=True)
    
    # Load existing graph if it exists
    if os.path.exists(graph_file):
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)
        return graph
    
    # Create new graph
    return nx.Graph()


def save_knowledge_graph(graph: nx.Graph, graph_file: str) -> None:
    """
    Save the knowledge graph to disk.
    
    Args:
        graph (nx.Graph): Knowledge graph
        graph_file (str): Path to save the graph
    """
    with open(graph_file, 'wb') as f:
        pickle.dump(graph, f)


def add_entity_to_graph(graph: nx.Graph, entity: str, entity_type: str) -> None:
    """
    Add an entity to the knowledge graph or update if it exists.
    
    Args:
        graph (nx.Graph): Knowledge graph
        entity (str): Entity name
        entity_type (str): Entity type (PERSON, ORG, etc.)
    """
    if graph.has_node(entity):
        # Update weight if node exists
        weight = graph.nodes[entity].get("weight", 0) + 1
        graph.nodes[entity]["weight"] = weight
    else:
        # Add new node
        graph.add_node(entity, type=entity_type, weight=1)


def add_relationship_to_graph(graph: nx.Graph, source: str, target: str, 
                             relationship_type: str) -> None:
    """
    Add a relationship between entities in the knowledge graph.
    
    Args:
        graph (nx.Graph): Knowledge graph
        source (str): Source entity name
        target (str): Target entity name
        relationship_type (str): Type of relationship
    """
    # Make sure both entities exist in the graph
    if not graph.has_node(source) or not graph.has_node(target):
        return
    
    # Update edge if it exists
    if graph.has_edge(source, target):
        # Increment weight
        weight = graph[source][target].get("weight", 0) + 1
        graph[source][target]["weight"] = weight
        
        # Add relationship type if not present
        if "types" in graph[source][target]:
            if relationship_type not in graph[source][target]["types"]:
                graph[source][target]["types"].append(relationship_type)
        else:
            graph[source][target]["types"] = [relationship_type]
    else:
        # Add new edge
        graph.add_edge(source, target, type=relationship_type, 
                      types=[relationship_type], weight=1)


def find_related_entities(graph: nx.Graph, query_entities: List[str], 
                         top_n: int = 5) -> List[str]:
    """
    Find related entities in the knowledge graph.
    
    Args:
        graph (nx.Graph): Knowledge graph
        query_entities (List[str]): List of entities from the query
        top_n (int): Number of top related entities to return
        
    Returns:
        List[str]: List of related entities
    """
    if not graph or graph.number_of_nodes() == 0:
        return []

    related_entities = {}

    for entity in query_entities:
        if entity in graph:
            # Get direct connections
            for neighbor in graph.neighbors(entity):
                score = graph[entity][neighbor].get('weight', 1)
                if neighbor in related_entities:
                    related_entities[neighbor] += score
                else:
                    related_entities[neighbor] = score

            # Use graph measures for broader connections
            try:
                # PageRank to find important connected entities
                pr = nx.pagerank(graph, personalization={entity: 1.0})
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


def find_path_between_entities(graph: nx.Graph, source: str, target: str, 
                              max_depth: int = 3) -> List[List[str]]:
    """
    Find paths between two entities in the knowledge graph.
    
    Args:
        graph (nx.Graph): Knowledge graph
        source (str): Source entity
        target (str): Target entity
        max_depth (int): Maximum path length
        
    Returns:
        List[List[str]]: List of paths between the entities
    """
    if not graph.has_node(source) or not graph.has_node(target):
        return []
    
    try:
        # Find all simple paths with limited length
        paths = list(nx.all_simple_paths(graph, source, target, cutoff=max_depth))
        return paths
    except (nx.NetworkXError, nx.NetworkXNoPath):
        return []


def get_entity_centrality(graph: nx.Graph, top_n: int = 20) -> List[Tuple[str, float]]:
    """
    Get the most central entities in the knowledge graph.
    
    Args:
        graph (nx.Graph): Knowledge graph
        top_n (int): Number of top central entities to return
        
    Returns:
        List[Tuple[str, float]]: List of (entity, centrality) pairs
    """
    if not graph or graph.number_of_nodes() == 0:
        return []
    
    try:
        # Calculate eigenvector centrality
        centrality = nx.eigenvector_centrality(graph, max_iter=100, tol=1e-4)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_centrality[:top_n]
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        # Fall back to degree centrality if eigenvector centrality fails
        centrality = nx.degree_centrality(graph)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_centrality[:top_n]


def find_communities(graph: nx.Graph) -> Dict[str, int]:
    """
    Find communities of entities in the knowledge graph.
    
    Args:
        graph (nx.Graph): Knowledge graph
        
    Returns:
        Dict[str, int]: Mapping of entity names to community IDs
    """
    if not graph or graph.number_of_nodes() < 3:
        return {}
    
    try:
        # Use Louvain method for community detection
        from community import best_partition
        partition = best_partition(graph)
        return partition
    except ImportError:
        # Fall back to connected components if python-louvain is not installed
        communities = {}
        for i, component in enumerate(nx.connected_components(graph)):
            for node in component:
                communities[node] = i
        return communities


def prune_knowledge_graph(graph: nx.Graph, min_weight: int = 2, 
                         max_nodes: int = 10000) -> nx.Graph:
    """
    Prune the knowledge graph to reduce size.
    
    Args:
        graph (nx.Graph): Knowledge graph
        min_weight (int): Minimum weight to keep a node
        max_nodes (int): Maximum number of nodes to keep
        
    Returns:
        nx.Graph: Pruned knowledge graph
    """
    if not graph:
        return graph
    
    # Create a copy of the graph
    pruned_graph = graph.copy()
    
    # Remove low-weight nodes
    nodes_to_remove = [node for node, data in pruned_graph.nodes(data=True) 
                      if data.get('weight', 0) < min_weight]
    pruned_graph.remove_nodes_from(nodes_to_remove)
    
    # If still too many nodes, keep only the most connected ones
    if pruned_graph.number_of_nodes() > max_nodes:
        degree = dict(pruned_graph.degree())
        sorted_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)
        nodes_to_keep = [node for node, _ in sorted_nodes[:max_nodes]]
        nodes_to_remove = [node for node in pruned_graph.nodes() if node not in nodes_to_keep]
        pruned_graph.remove_nodes_from(nodes_to_remove)
    
    return pruned_graph


def get_graph_stats(graph: nx.Graph) -> Dict[str, Any]:
    """
    Get statistics about the knowledge graph.
    
    Args:
        graph (nx.Graph): Knowledge graph
        
    Returns:
        Dict[str, Any]: Statistics about the graph
    """
    if not graph:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0,
            "connected_components": 0
        }
    
    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "connected_components": nx.number_connected_components(graph)
    }
    
    # Add degree statistics if graph has nodes
    if graph.number_of_nodes() > 0:
        degrees = [d for _, d in graph.degree()]
        stats["avg_degree"] = sum(degrees) / len(degrees)
        stats["max_degree"] = max(degrees) if degrees else 0
    
    return stats