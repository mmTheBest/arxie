"""Cross-artifact dependency graph and edit propagation for proposal artifacts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class ProposalArtifact(str, Enum):
    """Canonical artifact identifiers for synchronized proposal surfaces."""

    LOGICAL_TREE = "logical_tree"
    EVIDENCE_MAP = "evidence_map"
    HYPOTHESIS_TREE = "hypothesis_tree"
    DATA_OPTIONS_TABLE = "data_options_table"
    FEASIBILITY_SCORECARD = "feasibility_scorecard"
    EXPERIMENT_FLOW_DIAGRAM = "experiment_flow_diagram"
    ANALYSIS_PLAN_TREE = "analysis_plan_tree"
    OUTCOME_COMPARISON_MATRIX = "outcome_comparison_matrix"


@dataclass(frozen=True, slots=True)
class ArtifactNode:
    """Single artifact node tracked inside a proposal session."""

    artifact: ProposalArtifact
    node_id: str
    content: str
    provenance_link: str | None = None
    stale: bool = False


class ArtifactNodeNotFoundError(KeyError):
    """Raised when a requested artifact node does not exist."""

    def __init__(self, *, session_id: str, artifact: ProposalArtifact, node_id: str) -> None:
        super().__init__(
            f"Artifact node '{artifact.value}:{node_id}' was not found in session '{session_id}'."
        )
        self.session_id = session_id
        self.artifact = artifact
        self.node_id = node_id


class ProvenanceNotFoundError(KeyError):
    """Raised when a node has no provenance link to click through."""

    def __init__(self, *, session_id: str, artifact: ProposalArtifact, node_id: str) -> None:
        super().__init__(
            f"Provenance link not found for artifact node '{artifact.value}:{node_id}' in "
            f"session '{session_id}'."
        )
        self.session_id = session_id
        self.artifact = artifact
        self.node_id = node_id


@dataclass(slots=True)
class _SessionArtifactGraph:
    nodes: dict[tuple[ProposalArtifact, str], ArtifactNode] = field(default_factory=dict)
    edges: dict[tuple[ProposalArtifact, str], set[tuple[ProposalArtifact, str]]] = field(
        default_factory=dict
    )


class ArtifactSyncManager:
    """In-memory artifact graph manager with deterministic propagation behavior."""

    def __init__(self) -> None:
        self._graphs: dict[str, _SessionArtifactGraph] = {}

    def upsert_node(
        self,
        *,
        session_id: str,
        artifact: ProposalArtifact,
        node_id: str,
        content: str,
        provenance_link: str | None = None,
    ) -> ArtifactNode:
        key = _normalize_session_id(session_id)
        normalized_node_id = _normalize_node_id(node_id)
        normalized_content = _normalize_content(content)
        normalized_provenance = _normalize_optional_string(
            provenance_link,
            field_name="provenance_link",
            max_length=2048,
        )

        graph = self._graphs.setdefault(key, _SessionArtifactGraph())
        node_key = (artifact, normalized_node_id)
        existing = graph.nodes.get(node_key)
        stale = existing.stale if existing is not None else False

        node = ArtifactNode(
            artifact=artifact,
            node_id=normalized_node_id,
            content=normalized_content,
            provenance_link=normalized_provenance,
            stale=stale,
        )
        graph.nodes[node_key] = node
        graph.edges.setdefault(node_key, set())
        return node

    def add_dependency(
        self,
        *,
        session_id: str,
        upstream_artifact: ProposalArtifact,
        upstream_node_id: str,
        downstream_artifact: ProposalArtifact,
        downstream_node_id: str,
    ) -> None:
        key = _normalize_session_id(session_id)
        graph = self._graphs.setdefault(key, _SessionArtifactGraph())

        upstream_key = (upstream_artifact, _normalize_node_id(upstream_node_id))
        downstream_key = (downstream_artifact, _normalize_node_id(downstream_node_id))

        if upstream_key == downstream_key:
            raise ValueError("dependency edge must connect different nodes")

        self._require_node(graph, session_id=key, key=upstream_key)
        self._require_node(graph, session_id=key, key=downstream_key)

        if self._would_create_cycle(
            graph,
            upstream_key=upstream_key,
            downstream_key=downstream_key,
        ):
            raise ValueError("dependency edge would create a cycle")

        graph.edges.setdefault(upstream_key, set()).add(downstream_key)
        graph.edges.setdefault(downstream_key, set())

    def record_edit(
        self,
        *,
        session_id: str,
        artifact: ProposalArtifact,
        node_id: str,
        content: str,
    ) -> tuple[ArtifactNode, tuple[ArtifactNode, ...]]:
        key = _normalize_session_id(session_id)
        normalized_node_id = _normalize_node_id(node_id)
        normalized_content = _normalize_content(content)
        graph = self._graphs.setdefault(key, _SessionArtifactGraph())

        source_key = (artifact, normalized_node_id)
        source = self._require_node(graph, session_id=key, key=source_key)

        updated_source = ArtifactNode(
            artifact=source.artifact,
            node_id=source.node_id,
            content=normalized_content,
            provenance_link=source.provenance_link,
            stale=False,
        )
        graph.nodes[source_key] = updated_source

        impacted_keys = self._collect_downstream(graph, source_key)
        for impacted_key in impacted_keys:
            impacted = self._require_node(graph, session_id=key, key=impacted_key)
            graph.nodes[impacted_key] = ArtifactNode(
                artifact=impacted.artifact,
                node_id=impacted.node_id,
                content=impacted.content,
                provenance_link=impacted.provenance_link,
                stale=True,
            )

        impacted_nodes = tuple(
            graph.nodes[item_key]
            for item_key in sorted(impacted_keys, key=lambda item: (item[0].value, item[1]))
        )
        return updated_source, impacted_nodes

    def downstream_nodes(
        self,
        *,
        session_id: str,
        artifact: ProposalArtifact,
        node_id: str,
    ) -> tuple[ArtifactNode, ...]:
        key = _normalize_session_id(session_id)
        normalized_node_id = _normalize_node_id(node_id)
        graph = self._graphs.setdefault(key, _SessionArtifactGraph())
        source_key = (artifact, normalized_node_id)
        _ = self._require_node(graph, session_id=key, key=source_key)

        downstream_keys = self._collect_downstream(graph, source_key)
        return tuple(
            graph.nodes[item_key]
            for item_key in sorted(downstream_keys, key=lambda item: (item[0].value, item[1]))
        )

    def provenance_link(
        self,
        *,
        session_id: str,
        artifact: ProposalArtifact,
        node_id: str,
    ) -> str:
        key = _normalize_session_id(session_id)
        normalized_node_id = _normalize_node_id(node_id)
        graph = self._graphs.setdefault(key, _SessionArtifactGraph())
        node = self._require_node(graph, session_id=key, key=(artifact, normalized_node_id))
        if node.provenance_link is None:
            raise ProvenanceNotFoundError(
                session_id=key,
                artifact=artifact,
                node_id=normalized_node_id,
            )
        return node.provenance_link

    @staticmethod
    def _collect_downstream(
        graph: _SessionArtifactGraph,
        source_key: tuple[ProposalArtifact, str],
    ) -> set[tuple[ProposalArtifact, str]]:
        discovered: set[tuple[ProposalArtifact, str]] = set()
        queue: deque[tuple[ProposalArtifact, str]] = deque(graph.edges.get(source_key, ()))
        while queue:
            candidate = queue.popleft()
            if candidate == source_key:
                continue
            if candidate in discovered:
                continue
            discovered.add(candidate)
            queue.extend(graph.edges.get(candidate, ()))
        return discovered

    @classmethod
    def _would_create_cycle(
        cls,
        graph: _SessionArtifactGraph,
        *,
        upstream_key: tuple[ProposalArtifact, str],
        downstream_key: tuple[ProposalArtifact, str],
    ) -> bool:
        reachable_from_downstream = cls._collect_downstream(graph, downstream_key)
        return upstream_key in reachable_from_downstream

    @staticmethod
    def _require_node(
        graph: _SessionArtifactGraph,
        *,
        session_id: str,
        key: tuple[ProposalArtifact, str],
    ) -> ArtifactNode:
        node = graph.nodes.get(key)
        if node is None:
            artifact, node_id = key
            raise ArtifactNodeNotFoundError(
                session_id=session_id,
                artifact=artifact,
                node_id=node_id,
            )
        return node


def _normalize_session_id(session_id: str) -> str:
    return _normalize_required_string(session_id, field_name="session_id", max_length=128)


def _normalize_node_id(node_id: str) -> str:
    return _normalize_required_string(node_id, field_name="node_id", max_length=128)


def _normalize_content(content: str) -> str:
    return _normalize_required_string(content, field_name="content", max_length=20000)


def _normalize_required_string(value: str, *, field_name: str, max_length: int) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be <= {max_length} characters")
    return normalized


def _normalize_optional_string(
    value: str | None,
    *,
    field_name: str,
    max_length: int,
) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be <= {max_length} characters")
    return normalized
