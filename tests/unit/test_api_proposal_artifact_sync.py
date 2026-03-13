from __future__ import annotations

from fastapi.testclient import TestClient

from ra.api import create_app


class _StubRetriever:
    async def search(self, query: str, limit: int, sources: tuple[str, ...]):  # noqa: ARG002
        return []

    async def get_paper(self, identifier: str):  # noqa: ARG002
        return None

    async def close(self) -> None:
        return None


def _mk_app():
    return create_app(retriever_factory=lambda: _StubRetriever())


def _upsert_node(
    client: TestClient,
    *,
    session_id: str,
    artifact: str,
    node_id: str,
    content: str,
    provenance_link: str | None = None,
) -> dict[str, object]:
    response = client.put(
        f"/api/proposal/artifacts/{session_id}/nodes/{artifact}/{node_id}",
        json={"content": content, "provenance_link": provenance_link},
    )
    assert response.status_code == 200
    return response.json()


def test_cross_artifact_edit_propagation_sets_downstream_stale_markers() -> None:
    client = TestClient(_mk_app())

    _ = _upsert_node(
        client,
        session_id="session-1",
        artifact="logical_tree",
        node_id="logic-1",
        content="Problem -> mechanism chain",
    )
    _ = _upsert_node(
        client,
        session_id="session-1",
        artifact="evidence_map",
        node_id="evidence-1",
        content="Evidence bucket baseline",
        provenance_link="https://doi.org/10.1000/xyz123",
    )
    _ = _upsert_node(
        client,
        session_id="session-1",
        artifact="hypothesis_tree",
        node_id="hypothesis-1",
        content="Primary hypothesis",
    )

    dep_1 = client.post(
        "/api/proposal/artifacts/session-1/dependencies",
        json={
            "upstream_artifact": "logical_tree",
            "upstream_node_id": "logic-1",
            "downstream_artifact": "evidence_map",
            "downstream_node_id": "evidence-1",
        },
    )
    assert dep_1.status_code == 201

    dep_2 = client.post(
        "/api/proposal/artifacts/session-1/dependencies",
        json={
            "upstream_artifact": "evidence_map",
            "upstream_node_id": "evidence-1",
            "downstream_artifact": "hypothesis_tree",
            "downstream_node_id": "hypothesis-1",
        },
    )
    assert dep_2.status_code == 201

    edited = client.post(
        "/api/proposal/artifacts/session-1/edits",
        json={
            "artifact": "logical_tree",
            "node_id": "logic-1",
            "content": "Problem -> mechanism chain (revised)",
        },
    )
    assert edited.status_code == 200
    payload = edited.json()
    assert payload["session_id"] == "session-1"
    assert payload["source"]["artifact"] == "logical_tree"
    assert payload["source"]["node_id"] == "logic-1"
    assert payload["impacted_count"] == 2
    impacted = {(item["artifact"], item["node_id"]): item["stale"] for item in payload["impacted"]}
    assert impacted[("evidence_map", "evidence-1")] is True
    assert impacted[("hypothesis_tree", "hypothesis-1")] is True

    snapshot = client.get("/api/proposal/artifacts/session-1/dependencies/logical_tree/logic-1")
    assert snapshot.status_code == 200
    snapshot_payload = snapshot.json()
    assert snapshot_payload["count"] == 2
    downstream = {
        (item["artifact"], item["node_id"]): item["stale"]
        for item in snapshot_payload["downstream"]
    }
    assert downstream[("evidence_map", "evidence-1")] is True
    assert downstream[("hypothesis_tree", "hypothesis-1")] is True


def test_provenance_clickthrough_redirects_to_source_link() -> None:
    client = TestClient(_mk_app())
    _ = _upsert_node(
        client,
        session_id="session-1",
        artifact="evidence_map",
        node_id="paper-1",
        content="Pinned paper",
        provenance_link="https://doi.org/10.1000/xyz123",
    )

    response = client.get(
        "/api/proposal/artifacts/session-1/provenance/evidence_map/paper-1",
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert response.headers["location"] == "https://doi.org/10.1000/xyz123"


def test_provenance_clickthrough_returns_not_found_when_node_has_no_link() -> None:
    client = TestClient(_mk_app())
    _ = _upsert_node(
        client,
        session_id="session-1",
        artifact="hypothesis_tree",
        node_id="hypothesis-1",
        content="No citation yet",
    )

    response = client.get(
        "/api/proposal/artifacts/session-1/provenance/hypothesis_tree/hypothesis-1",
        follow_redirects=False,
    )

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"] == "provenance_not_found"


def test_planning_artifacts_support_dependency_propagation() -> None:
    client = TestClient(_mk_app())

    _ = _upsert_node(
        client,
        session_id="session-1",
        artifact="data_options_table",
        node_id="data-option-1",
        content="Public cohort option with demographics and outcomes.",
    )
    _ = _upsert_node(
        client,
        session_id="session-1",
        artifact="experiment_flow_diagram",
        node_id="experiment-1",
        content="Enroll -> randomize -> monitor -> evaluate.",
    )
    _ = _upsert_node(
        client,
        session_id="session-1",
        artifact="analysis_plan_tree",
        node_id="analysis-1",
        content="Primary model -> sensitivity checks.",
    )
    _ = _upsert_node(
        client,
        session_id="session-1",
        artifact="outcome_comparison_matrix",
        node_id="outcome-1",
        content="Expected and adverse outcomes by branch.",
    )

    dep_1 = client.post(
        "/api/proposal/artifacts/session-1/dependencies",
        json={
            "upstream_artifact": "data_options_table",
            "upstream_node_id": "data-option-1",
            "downstream_artifact": "experiment_flow_diagram",
            "downstream_node_id": "experiment-1",
        },
    )
    assert dep_1.status_code == 201

    dep_2 = client.post(
        "/api/proposal/artifacts/session-1/dependencies",
        json={
            "upstream_artifact": "experiment_flow_diagram",
            "upstream_node_id": "experiment-1",
            "downstream_artifact": "analysis_plan_tree",
            "downstream_node_id": "analysis-1",
        },
    )
    assert dep_2.status_code == 201

    dep_3 = client.post(
        "/api/proposal/artifacts/session-1/dependencies",
        json={
            "upstream_artifact": "analysis_plan_tree",
            "upstream_node_id": "analysis-1",
            "downstream_artifact": "outcome_comparison_matrix",
            "downstream_node_id": "outcome-1",
        },
    )
    assert dep_3.status_code == 201

    edited = client.post(
        "/api/proposal/artifacts/session-1/edits",
        json={
            "artifact": "data_options_table",
            "node_id": "data-option-1",
            "content": "Public cohort option (revised for coverage gaps).",
        },
    )
    assert edited.status_code == 200
    payload = edited.json()
    assert payload["impacted_count"] == 3

    impacted = {(item["artifact"], item["node_id"]): item["stale"] for item in payload["impacted"]}
    assert impacted[("experiment_flow_diagram", "experiment-1")] is True
    assert impacted[("analysis_plan_tree", "analysis-1")] is True
    assert impacted[("outcome_comparison_matrix", "outcome-1")] is True


def test_dependency_creation_rejects_self_edge() -> None:
    client = TestClient(_mk_app())
    _ = _upsert_node(
        client,
        session_id="session-1",
        artifact="logical_tree",
        node_id="logic-1",
        content="Problem statement",
    )

    response = client.post(
        "/api/proposal/artifacts/session-1/dependencies",
        json={
            "upstream_artifact": "logical_tree",
            "upstream_node_id": "logic-1",
            "downstream_artifact": "logical_tree",
            "downstream_node_id": "logic-1",
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "invalid_input"
    assert "must connect different nodes" in payload["message"]


def test_dependency_creation_rejects_cycle_edge() -> None:
    client = TestClient(_mk_app())
    _ = _upsert_node(
        client,
        session_id="session-1",
        artifact="logical_tree",
        node_id="logic-1",
        content="Problem statement",
    )
    _ = _upsert_node(
        client,
        session_id="session-1",
        artifact="evidence_map",
        node_id="evidence-1",
        content="Evidence bucket",
    )

    first = client.post(
        "/api/proposal/artifacts/session-1/dependencies",
        json={
            "upstream_artifact": "logical_tree",
            "upstream_node_id": "logic-1",
            "downstream_artifact": "evidence_map",
            "downstream_node_id": "evidence-1",
        },
    )
    assert first.status_code == 201

    cycle = client.post(
        "/api/proposal/artifacts/session-1/dependencies",
        json={
            "upstream_artifact": "evidence_map",
            "upstream_node_id": "evidence-1",
            "downstream_artifact": "logical_tree",
            "downstream_node_id": "logic-1",
        },
    )

    assert cycle.status_code == 400
    payload = cycle.json()
    assert payload["error"] == "invalid_input"
    assert "would create a cycle" in payload["message"]
