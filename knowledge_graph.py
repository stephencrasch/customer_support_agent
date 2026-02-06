"""
Knowledge Graph - DAG structure for adaptive learning
Tracks nodes, prerequisites, proficiency, and learning paths.
"""
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json


class NodeStatus(Enum):
    """Status of a knowledge node in the learning path."""
    LOCKED = "locked"           # Prerequisites not met
    AVAILABLE = "available"     # Ready to learn
    IN_PROGRESS = "in_progress" # Currently studying
    COMPLETED = "completed"     # Mastered (proficiency >= 80)


@dataclass
class KnowledgeNode:
    """A single knowledge node in the curriculum."""
    id: str
    name: str
    description: str
    prerequisites: List[str] = field(default_factory=list)
    proficiency: float = 0.0  # 0-100 scale
    status: NodeStatus = NodeStatus.LOCKED
    attempts: int = 0
    # Lightweight “cumulative” memory.
    # Keep short and bounded: it’s meant to steer probing/teaching, not store a transcript.
    taught_points: List[str] = field(default_factory=list)
    misconceptions: List[str] = field(default_factory=list)
    last_check_question: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "prerequisites": self.prerequisites,
            "proficiency": self.proficiency,
            "status": self.status.value,
            "attempts": self.attempts,
            "taught_points": self.taught_points,
            "misconceptions": self.misconceptions,
            "last_check_question": self.last_check_question,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeNode':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            prerequisites=data.get("prerequisites", []),
            proficiency=data.get("proficiency", 0.0),
            status=NodeStatus(data.get("status", "locked")),
            attempts=data.get("attempts", 0),
            taught_points=data.get("taught_points", []) or [],
            misconceptions=data.get("misconceptions", []) or [],
            last_check_question=data.get("last_check_question", "") or "",
        )


class KnowledgeGraph:
    """
    Manages the DAG of knowledge nodes for a learning topic.
    Handles prerequisites, proficiency tracking, and progression.
    """
    
    def __init__(self, topic: str):
        self.topic = topic
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.current_node_id: Optional[str] = None
    
    def add_node(self, node: KnowledgeNode) -> None:
        """Add a knowledge node to the graph."""
        self.nodes[node.id] = node
        
        # If this is the first node with no prerequisites, make it available
        if not node.prerequisites and not self.current_node_id:
            node.status = NodeStatus.AVAILABLE
            self.current_node_id = node.id
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def update_proficiency(self, node_id: str, score: float) -> None:
        """
        Update proficiency for a node based on test score.
        Uses weighted average: new_prof = 0.7 * old + 0.3 * score
        """
        node = self.get_node(node_id)
        if not node:
            return
        
        node.attempts += 1
        
        # Weighted average favoring recent performance
        if node.attempts == 1:
            node.proficiency = score
        else:
            node.proficiency = 0.7 * node.proficiency + 0.3 * score
        
        # Update status based on proficiency
        if node.proficiency >= 80:
            node.status = NodeStatus.COMPLETED
            self._unlock_next_nodes(node_id)
        elif node.proficiency >= 50:
            node.status = NodeStatus.IN_PROGRESS
    
    def _unlock_next_nodes(self, completed_node_id: str) -> None:
        """Unlock nodes that had this node as a prerequisite."""
        for node in self.nodes.values():
            if completed_node_id in node.prerequisites:
                # Check if all prerequisites are completed
                if self._all_prerequisites_met(node.id):
                    node.status = NodeStatus.AVAILABLE
    
    def _all_prerequisites_met(self, node_id: str) -> bool:
        """Check if all prerequisites for a node are completed."""
        node = self.get_node(node_id)
        if not node:
            return False
        
        for prereq_id in node.prerequisites:
            prereq_node = self.get_node(prereq_id)
            if not prereq_node or prereq_node.status != NodeStatus.COMPLETED:
                return False
        
        return True
    
    def get_available_nodes(self) -> List[KnowledgeNode]:
        """Get all nodes available for learning."""
        return [
            node for node in self.nodes.values()
            if node.status == NodeStatus.AVAILABLE
        ]
    
    def get_current_node(self) -> Optional[KnowledgeNode]:
        """Get the current node being studied."""
        if self.current_node_id:
            return self.get_node(self.current_node_id)
        return None
    
    def set_current_node(self, node_id: str) -> bool:
        """Set the current node to study."""
        node = self.get_node(node_id)
        if node and node.status in [NodeStatus.AVAILABLE, NodeStatus.IN_PROGRESS]:
            self.current_node_id = node_id
            if node.status == NodeStatus.AVAILABLE:
                node.status = NodeStatus.IN_PROGRESS
            return True
        return False
    
    def get_progress_summary(self) -> Dict:
        """Get overall learning progress statistics."""
        total = len(self.nodes)
        completed = sum(1 for n in self.nodes.values() if n.status == NodeStatus.COMPLETED)
        in_progress = sum(1 for n in self.nodes.values() if n.status == NodeStatus.IN_PROGRESS)
        avg_proficiency = sum(n.proficiency for n in self.nodes.values()) / total if total > 0 else 0
        
        return {
            "topic": self.topic,
            "total_nodes": total,
            "completed": completed,
            "in_progress": in_progress,
            "locked": total - completed - in_progress,
            "avg_proficiency": round(avg_proficiency, 1),
            "completion_percentage": round((completed / total) * 100, 1) if total > 0 else 0
        }
    
    def get_learning_path(self) -> List[KnowledgeNode]:
        """Get suggested learning path (topological sort of available/unlocked nodes)."""
        path = []
        visited = set()
        
        def dfs(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            
            node = self.get_node(node_id)
            if not node:
                return
            
            # Visit prerequisites first
            for prereq_id in node.prerequisites:
                dfs(prereq_id)
            
            path.append(node)
        
        # Start with nodes that have no prerequisites
        for node in self.nodes.values():
            if not node.prerequisites:
                dfs(node.id)
        
        return path
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "topic": self.topic,
            "current_node_id": self.current_node_id,
            "nodes": [node.to_dict() for node in self.nodes.values()]
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeGraph':
        """Deserialize from dictionary."""
        graph = cls(topic=data["topic"])
        graph.current_node_id = data.get("current_node_id")
        
        for node_data in data.get("nodes", []):
            node = KnowledgeNode.from_dict(node_data)
            graph.nodes[node.id] = node
        
        return graph
    
    @classmethod
    def from_json(cls, json_str: str) -> 'KnowledgeGraph':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
