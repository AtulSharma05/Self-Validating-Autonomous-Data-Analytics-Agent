"""
Provenance DAG Tracking Module (Feature 10).

Maintains a cryptographic Directed Acyclic Graph (DAG) using NetworkX, recording
every analytical step, execution SQL/Python, result summary, validation timestamp,
and SHA-256 audit fingerprint.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import networkx as nx


@dataclass
class EvidenceArtifact:
    """
    Immutable artifact node in the provenance graph.
    """
    step_id: str
    action_type: str
    query_sql: Optional[str]
    result_summary: str
    checksum_hash: str
    validation_status: str
    execution_time_ms: float
    parent_step_ids: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def compute_checksum(cls, query: str, data_snapshot: str = "") -> str:
        """Computes SHA-256 hash over query and input slice."""
        payload = f"{query.strip()}|{data_snapshot.strip()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


class ProvenanceDAG:
    """
    Directed Acyclic Graph tracking lineage and dependencies between analytical steps.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.graph = nx.DiGraph()
        self.artifacts: Dict[str, EvidenceArtifact] = {}

    def add_artifact(self, artifact: EvidenceArtifact) -> None:
        """
        Inserts an artifact into the provenance graph and creates edges to its parents.
        """
        self.artifacts[artifact.step_id] = artifact
        self.graph.add_node(artifact.step_id, **artifact.to_dict())

        for parent_id in artifact.parent_step_ids:
            if parent_id in self.graph:
                self.graph.add_edge(parent_id, artifact.step_id)

    def add_dependency(self, child_step_id: str, parent_step_id: str) -> None:
        """Explicitly registers a dependency edge between two steps."""
        if child_step_id in self.graph and parent_step_id in self.graph:
            self.graph.add_edge(parent_step_id, child_step_id)
            if parent_step_id not in self.artifacts[child_step_id].parent_step_ids:
                self.artifacts[child_step_id].parent_step_ids.append(parent_step_id)

    def get_artifact(self, step_id: str) -> Optional[EvidenceArtifact]:
        """Retrieves artifact by step_id."""
        return self.artifacts.get(step_id)

    def get_lineage(self, step_id: str) -> List[EvidenceArtifact]:
        """
        Returns full upstream ancestor lineage of a step in topological order.
        """
        if step_id not in self.graph:
            return []

        ancestors = nx.ancestors(self.graph, step_id)
        # Add the target step itself
        sub_nodes = set(ancestors) | {step_id}
        subgraph = self.graph.subgraph(sub_nodes)

        try:
            ordered_ids = list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            # Fallback if cyclic anomaly
            ordered_ids = list(sub_nodes)

        return [self.artifacts[nid] for nid in ordered_ids if nid in self.artifacts]

    def verify_acyclic(self) -> bool:
        """Checks if the dependency structure forms a valid DAG (no cycles)."""
        return nx.is_directed_acyclic_graph(self.graph)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the DAG into nodes and edges suitable for API responses and visual graphs.
        """
        nodes = []
        for nid in self.graph.nodes():
            art = self.artifacts.get(nid)
            if art:
                nodes.append(art.to_dict())
            else:
                nodes.append({"step_id": nid})

        edges = [
            {"from": u, "to": v}
            for u, v in self.graph.edges()
        ]

        return {
            "session_id": self.session_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "is_acyclic": self.verify_acyclic(),
            "nodes": nodes,
            "edges": edges,
        }

    def to_json(self) -> str:
        """Returns JSON string of the graph structure."""
        return json.dumps(self.to_dict(), indent=2)

