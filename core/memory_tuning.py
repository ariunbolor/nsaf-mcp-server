"""
Neuro-Symbolic Autonomy Framework (NSAF) - Hyper-Symbolic Memory Tuning
=======================================================================

Author: Bolorerdene Bundgaa
Contact: bolor@ariunbolor.org
Website: https://bolor.me

Hyper-Symbolic Memory implementation with graph-based memory structure,
symbolic reasoning, and dynamic memory consolidation.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass
import yaml
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import networkx as nx
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import RDFS, XSD
import sympy as sp
from datetime import datetime
import heapq

@dataclass
class MemoryNode:
    """Represents a node in the memory graph"""
    id: str
    content: Any
    type: str
    timestamp: float
    importance: float
    connections: Set[str]  # Set of connected node IDs
    symbolic_form: Optional[sp.Expr] = None
    embedding: Optional[np.ndarray] = None

class MemoryGraph:
    """Graph-based memory structure with symbolic reasoning capabilities"""
    def __init__(self):
        self.graph = nx.DiGraph()
        self.knowledge_graph = Graph()
        self.node_embeddings = {}
        
    def add_node(self, node: MemoryNode) -> None:
        """Add a node to both graph representations"""
        self.graph.add_node(node.id, 
                           content=node.content,
                           type=node.type,
                           timestamp=node.timestamp,
                           importance=node.importance)
        
        # Add to RDF knowledge graph
        node_uri = URIRef(f"memory:{node.id}")
        self.knowledge_graph.add((node_uri, RDF.type, URIRef(f"type:{node.type}")))
        self.knowledge_graph.add((node_uri, RDFS.label, Literal(str(node.content))))
        
        if node.embedding is not None:
            self.node_embeddings[node.id] = node.embedding
    
    def add_edge(self, source_id: str, target_id: str, 
                relation_type: str, weight: float) -> None:
        """Add an edge between nodes"""
        self.graph.add_edge(source_id, target_id, 
                           type=relation_type, weight=weight)
        
        # Add to RDF knowledge graph
        source_uri = URIRef(f"memory:{source_id}")
        target_uri = URIRef(f"memory:{target_id}")
        relation_uri = URIRef(f"relation:{relation_type}")
        self.knowledge_graph.add((source_uri, relation_uri, target_uri))

class HyperSymbolicMemory:
    """
    Implements Hyper-Symbolic Memory Tuning for enhanced agent cognition through
    advanced memory management and symbolic reasoning.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['memory_tuning']
        
        # Initialize parameters
        self.memory_size = self.config['memory_size']
        self.retention_rate = self.config['retention_rate']
        self.forgetting_factor = self.config['forgetting_factor']
        self.consolidation_interval = self.config['consolidation_interval']
        
        # Initialize memory structures
        self.memory_graph = MemoryGraph()
        self.importance_queue = []  # Priority queue for memory management
        self.symbolic_cache = {}    # Cache for symbolic computations
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize neural components
        self.encoder = self._build_encoder()
        self.attention = self._build_attention()

    def _build_encoder(self) -> nn.Module:
        """Build neural encoder for memory embeddings"""
        return nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def _build_attention(self) -> nn.Module:
        """Build attention mechanism for memory access"""
        return nn.MultiheadAttention(128, num_heads=4)

    def _compute_importance(self, node: MemoryNode) -> float:
        """Compute importance score for a memory node"""
        # Factors affecting importance:
        # 1. Recency (time decay)
        time_factor = np.exp(-self.forgetting_factor * 
                           (datetime.now().timestamp() - node.timestamp))
        
        # 2. Connectivity (number of connections)
        connectivity_factor = len(node.connections) / 10  # Normalize
        
        # 3. Usage frequency (from access patterns)
        usage_factor = self.graph.degree(node.id) / 10  # Normalize
        
        # 4. Symbolic importance (based on symbolic form complexity)
        symbolic_factor = 0.0
        if node.symbolic_form is not None:
            symbolic_factor = len(str(node.symbolic_form)) / 100  # Normalize
        
        # Combine factors with weights
        importance = (0.3 * time_factor + 
                     0.3 * connectivity_factor +
                     0.2 * usage_factor +
                     0.2 * symbolic_factor)
        
        return importance

    def _consolidate_memories(self) -> None:
        """Consolidate memories through symbolic abstraction and pruning"""
        # Update importance scores
        nodes = list(self.memory_graph.graph.nodes(data=True))
        for node_id, data in nodes:
            node = MemoryNode(
                id=node_id,
                content=data['content'],
                type=data['type'],
                timestamp=data['timestamp'],
                importance=data['importance'],
                connections=set(self.memory_graph.graph.neighbors(node_id))
            )
            importance = self._compute_importance(node)
            self.memory_graph.graph.nodes[node_id]['importance'] = importance
            heapq.heappush(self.importance_queue, (-importance, node_id))
        
        # Prune least important memories if over capacity
        while len(self.memory_graph.graph) > self.memory_size:
            _, node_id = heapq.heappop(self.importance_queue)
            if node_id in self.memory_graph.graph:
                self.memory_graph.graph.remove_node(node_id)
        
        # Perform symbolic abstraction
        self._symbolic_abstraction()

    def _symbolic_abstraction(self) -> None:
        """Create symbolic abstractions of related memories"""
        # Find clusters of related memories
        clusters = list(nx.community.greedy_modularity_communities(
            self.memory_graph.graph.to_undirected()))
        
        for cluster in clusters:
            if len(cluster) > 1:
                # Extract subgraph for the cluster
                subgraph = self.memory_graph.graph.subgraph(cluster)
                
                # Create symbolic representation
                symbols = []
                relations = []
                
                for node_id in subgraph.nodes():
                    node_data = subgraph.nodes[node_id]
                    if isinstance(node_data['content'], (int, float)):
                        symbol = sp.Symbol(f"x_{node_id}")
                        symbols.append((symbol, node_data['content']))
                
                for edge in subgraph.edges(data=True):
                    source, target, data = edge
                    if data['type'] in ['greater_than', 'less_than', 'equals']:
                        source_sym = sp.Symbol(f"x_{source}")
                        target_sym = sp.Symbol(f"x_{target}")
                        if data['type'] == 'greater_than':
                            relations.append(source_sym > target_sym)
                        elif data['type'] == 'less_than':
                            relations.append(source_sym < target_sym)
                        elif data['type'] == 'equals':
                            relations.append(sp.Eq(source_sym, target_sym))
                
                # Create and store symbolic abstraction
                if symbols and relations:
                    abstraction = sp.And(*relations)
                    self.symbolic_cache[frozenset(cluster)] = {
                        'symbols': symbols,
                        'abstraction': abstraction
                    }

    def add_memory(self, content: Any, memory_type: str, 
                  connections: Optional[Set[str]] = None) -> str:
        """
        Add new memory to the system.
        
        Args:
            content: Memory content
            memory_type: Type of memory (e.g., 'fact', 'rule', 'experience')
            connections: Set of connected memory IDs
            
        Returns:
            str: ID of the new memory node
        """
        # Generate memory embedding with proper string handling (768 dims for encoder)
        if isinstance(content, str):
            # Convert string to numeric representation using character codes
            content_vec = [ord(c) for c in content[:768]]  # Limit to 768 chars to match encoder
            # Pad or truncate to fixed size
            if len(content_vec) < 768:
                content_vec.extend([0] * (768 - len(content_vec)))
            else:
                content_vec = content_vec[:768]
            content_tensor = torch.tensor(content_vec, dtype=torch.float32)
        elif isinstance(content, (list, tuple)):
            # Handle list/array content
            content_list = list(content)
            if len(content_list) < 768:
                content_list.extend([0.0] * (768 - len(content_list)))
            else:
                content_list = content_list[:768]
            content_tensor = torch.tensor(content_list, dtype=torch.float32)
        elif isinstance(content, dict):
            # Convert dict to numeric representation
            content_str = str(content)
            content_vec = [ord(c) for c in content_str[:768]]
            if len(content_vec) < 768:
                content_vec.extend([0] * (768 - len(content_vec)))
            else:
                content_vec = content_vec[:768]
            content_tensor = torch.tensor(content_vec, dtype=torch.float32)
        else:
            # Handle other types by converting to string first
            content_str = str(content)
            content_vec = [ord(c) for c in content_str[:768]]
            if len(content_vec) < 768:
                content_vec.extend([0] * (768 - len(content_vec)))
            else:
                content_vec = content_vec[:768]
            content_tensor = torch.tensor(content_vec, dtype=torch.float32)
        
        embedding = self.encoder(content_tensor)
        
        # Create memory node
        node = MemoryNode(
            id=f"memory_{len(self.memory_graph.graph)}",
            content=content,
            type=memory_type,
            timestamp=datetime.now().timestamp(),
            importance=1.0,
            connections=connections or set(),
            embedding=embedding.detach().numpy()
        )
        
        # Add to memory graph
        self.memory_graph.add_node(node)
        
        # Add connections
        if connections:
            for connected_id in connections:
                if connected_id in self.memory_graph.graph:
                    self.memory_graph.add_edge(
                        node.id, connected_id, 
                        relation_type='associated_with',
                        weight=1.0
                    )
        
        # Consolidate if needed
        if len(self.memory_graph.graph) % self.consolidation_interval == 0:
            self._consolidate_memories()
        
        return node.id

    def query_memory(self, query: Any, k: int = 5) -> List[Tuple[str, float]]:
        """
        Query memory system for relevant information.
        
        Args:
            query: Query content
            k: Number of results to return
            
        Returns:
            List[Tuple[str, float]]: List of (memory_id, relevance_score) pairs
        """
        # Encode query with proper string handling (768 dims for encoder)
        if isinstance(query, str):
            # Convert string to numeric representation using character codes
            query_vec = [ord(c) for c in query[:768]]  # Limit to 768 chars to match encoder
            # Pad or truncate to fixed size
            if len(query_vec) < 768:
                query_vec.extend([0] * (768 - len(query_vec)))
            else:
                query_vec = query_vec[:768]
            query_tensor = torch.tensor(query_vec, dtype=torch.float32)
        elif isinstance(query, (list, tuple)):
            # Handle list/array query
            query_list = list(query)
            if len(query_list) < 768:
                query_list.extend([0.0] * (768 - len(query_list)))
            else:
                query_list = query_list[:768]
            query_tensor = torch.tensor(query_list, dtype=torch.float32)
        elif isinstance(query, dict):
            # Convert dict to numeric representation
            query_str = str(query)
            query_vec = [ord(c) for c in query_str[:768]]
            if len(query_vec) < 768:
                query_vec.extend([0] * (768 - len(query_vec)))
            else:
                query_vec = query_vec[:768]
            query_tensor = torch.tensor(query_vec, dtype=torch.float32)
        else:
            # Handle other types by converting to string first
            query_str = str(query)
            query_vec = [ord(c) for c in query_str[:768]]
            if len(query_vec) < 768:
                query_vec.extend([0] * (768 - len(query_vec)))
            else:
                query_vec = query_vec[:768]
            query_tensor = torch.tensor(query_vec, dtype=torch.float32)
        
        query_embedding = self.encoder(query_tensor)
        
        # Compute similarity scores with all memories using cosine similarity
        if len(self.memory_graph.graph.nodes()) == 0:
            return []
            
        memory_embeddings = torch.stack([
            torch.tensor(self.memory_graph.node_embeddings[node_id])
            for node_id in self.memory_graph.graph.nodes()
        ])
        
        # Compute cosine similarity between query and all memory embeddings
        query_norm = query_embedding / query_embedding.norm()
        memory_norms = memory_embeddings / memory_embeddings.norm(dim=1, keepdim=True)
        similarities = torch.mm(query_norm.unsqueeze(0), memory_norms.T)
        
        # Get similarity scores
        scores = similarities.squeeze().detach().numpy()
        
        # Handle single memory case
        if len(self.memory_graph.graph.nodes()) == 1:
            scores = [scores.item()]
        node_ids = list(self.memory_graph.graph.nodes())
        results = [(node_id, score) for node_id, score 
                  in zip(node_ids, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]

    def reason(self, query: Any) -> List[Dict[str, Any]]:
        """
        Perform symbolic reasoning over memories.
        
        Args:
            query: Query to reason about
            
        Returns:
            List[Dict[str, Any]]: List of reasoning results
        """
        # Find relevant memories
        relevant_memories = self.query_memory(query)
        
        results = []
        for memory_id, score in relevant_memories:
            # Get connected memories
            connected = set(self.memory_graph.graph.neighbors(memory_id))
            
            # Check for symbolic abstractions
            for cached_cluster, abstraction in self.symbolic_cache.items():
                if memory_id in cached_cluster:
                    # Apply symbolic reasoning
                    symbols = abstraction['symbols']
                    expr = abstraction['abstraction']
                    
                    try:
                        # Attempt to solve or simplify
                        solution = sp.solve(expr)
                        results.append({
                            'memory_id': memory_id,
                            'relevance': score,
                            'symbolic_result': str(solution),
                            'confidence': score * len(connected) / len(symbols)
                        })
                    except Exception as e:
                        self.logger.warning(f"Symbolic reasoning failed: {e}")
        
        return results

    def update_importance(self, memory_id: str, delta: float) -> None:
        """Update importance score of a memory"""
        if memory_id in self.memory_graph.graph:
            current = self.memory_graph.graph.nodes[memory_id]['importance']
            new_importance = max(0.0, min(1.0, current + delta))
            self.memory_graph.graph.nodes[memory_id]['importance'] = new_importance
            
            # Update priority queue
            heapq.heappush(self.importance_queue, 
                          (-new_importance, memory_id))

    def forget(self, threshold: float) -> None:
        """Remove memories below importance threshold"""
        to_remove = []
        for node_id, data in self.memory_graph.graph.nodes(data=True):
            if data['importance'] < threshold:
                to_remove.append(node_id)
        
        for node_id in to_remove:
            self.memory_graph.graph.remove_node(node_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get memory system metrics.
        
        Returns:
            Dictionary containing memory system metrics
        """
        return {
            'total_memories': len(self.memory_graph.graph.nodes()),
            'total_connections': len(self.memory_graph.graph.edges()),
            'symbolic_cache_size': len(self.symbolic_cache),
            'memory_types': len(set(data.get('memory_type', 'unknown') 
                                  for node_id, data in self.memory_graph.graph.nodes(data=True))),
            'average_importance': sum(data.get('importance', 0.0) 
                                    for node_id, data in self.memory_graph.graph.nodes(data=True)) / 
                                 max(len(self.memory_graph.graph.nodes()), 1)
        }
