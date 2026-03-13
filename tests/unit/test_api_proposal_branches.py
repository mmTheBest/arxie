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


def _create_branch(
    client: TestClient,
    *,
    session_id: str,
    branch_id: str,
    name: str,
    hypothesis: str,
    parent_branch_id: str | None = None,
    scorecard: dict[str, float] | None = None,
    metadata: dict[str, str] | None = None,
):
    payload: dict[str, object] = {
        "session_id": session_id,
        "branch_id": branch_id,
        "name": name,
        "hypothesis": hypothesis,
        "scorecard": scorecard
        or {
            "evidence_support": 0.7,
            "feasibility": 0.8,
            "risk": 0.3,
            "impact": 0.9,
        },
    }
    if parent_branch_id is not None:
        payload["parent_branch_id"] = parent_branch_id
    if metadata is not None:
        payload["metadata"] = metadata
    return client.post("/api/proposal/branches", json=payload)


def test_create_branch_endpoint_and_list_endpoint_round_trip() -> None:
    client = TestClient(_mk_app())
    root = _create_branch(
        client,
        session_id="session-1",
        branch_id="root",
        name="Root hypothesis",
        hypothesis="Primary causal hypothesis.",
    )

    assert root.status_code == 201
    root_payload = root.json()
    assert root_payload["branch_id"] == "root"
    assert root_payload["is_primary"] is True

    alt = _create_branch(
        client,
        session_id="session-1",
        branch_id="alt",
        name="Alternative hypothesis",
        hypothesis="Alternative causal explanation.",
        parent_branch_id="root",
    )
    assert alt.status_code == 201

    listed = client.get("/api/proposal/branches/session-1")
    assert listed.status_code == 200
    listed_payload = listed.json()
    assert listed_payload["count"] == 2
    assert [branch["branch_id"] for branch in listed_payload["branches"]] == ["root", "alt"]
    assert listed_payload["branches"][1]["parent_branch_id"] == "root"
    assert listed_payload["branches"][1]["lineage"] == ["root"]


def test_get_branch_endpoint_returns_metadata_and_confidence_label() -> None:
    client = TestClient(_mk_app())
    created = _create_branch(
        client,
        session_id="session-1",
        branch_id="root",
        name="Root hypothesis",
        hypothesis="Primary causal hypothesis.",
        scorecard={
            "evidence_support": 0.92,
            "feasibility": 0.88,
            "risk": 0.1,
            "impact": 0.9,
        },
        metadata={"owner": "planner", "stage": "m3"},
    )
    assert created.status_code == 201
    created_payload = created.json()
    assert created_payload["confidence_label"] == "high"
    assert created_payload["metadata"]["owner"] == "planner"

    fetched = client.get("/api/proposal/branches/session-1/root")
    assert fetched.status_code == 200
    payload = fetched.json()
    assert payload["branch_id"] == "root"
    assert payload["confidence_label"] == "high"
    assert payload["metadata"] == {"owner": "planner", "stage": "m3"}


def test_compare_branches_endpoint_returns_winner_and_scores() -> None:
    client = TestClient(_mk_app())
    _ = _create_branch(
        client,
        session_id="session-1",
        branch_id="branch-a",
        name="Branch A",
        hypothesis="Hypothesis A",
        scorecard={
            "evidence_support": 0.9,
            "feasibility": 0.8,
            "risk": 0.1,
            "impact": 0.9,
        },
    )
    _ = _create_branch(
        client,
        session_id="session-1",
        branch_id="branch-b",
        name="Branch B",
        hypothesis="Hypothesis B",
        scorecard={
            "evidence_support": 0.6,
            "feasibility": 0.7,
            "risk": 0.4,
            "impact": 0.7,
        },
    )

    compared = client.post(
        "/api/proposal/branches/compare",
        json={
            "session_id": "session-1",
            "branch_ids": ["branch-a", "branch-b"],
        },
    )

    assert compared.status_code == 200
    payload = compared.json()
    assert payload["winner_branch_id"] == "branch-a"
    assert len(payload["comparisons"]) == 2
    assert (
        payload["comparisons"][0]["aggregate_score"]
        >= payload["comparisons"][1]["aggregate_score"]
    )
    assert payload["comparisons"][0]["confidence_label"] in {"high", "medium", "low"}


def test_promote_branch_endpoint_enforces_single_primary_branch() -> None:
    client = TestClient(_mk_app())
    _ = _create_branch(
        client,
        session_id="session-1",
        branch_id="root",
        name="Root",
        hypothesis="Hypothesis root.",
    )
    _ = _create_branch(
        client,
        session_id="session-1",
        branch_id="alt",
        name="Alt",
        hypothesis="Hypothesis alt.",
    )

    promoted = client.post("/api/proposal/branches/session-1/alt/promote")
    assert promoted.status_code == 200
    promoted_payload = promoted.json()
    assert promoted_payload["branch_id"] == "alt"
    assert promoted_payload["is_primary"] is True

    listed = client.get("/api/proposal/branches/session-1")
    assert listed.status_code == 200
    primary_ids = [
        branch["branch_id"]
        for branch in listed.json()["branches"]
        if branch["is_primary"]
    ]
    assert primary_ids == ["alt"]


def test_create_branch_endpoint_rejects_duplicate_branch_id() -> None:
    client = TestClient(_mk_app())
    first = _create_branch(
        client,
        session_id="session-1",
        branch_id="root",
        name="Root",
        hypothesis="Hypothesis root.",
    )
    duplicate = _create_branch(
        client,
        session_id="session-1",
        branch_id="root",
        name="Duplicate",
        hypothesis="Hypothesis duplicate.",
    )

    assert first.status_code == 201
    assert duplicate.status_code == 409
    payload = duplicate.json()
    assert payload["error"] == "branch_exists"


def test_compare_branches_endpoint_returns_not_found_for_missing_branch() -> None:
    client = TestClient(_mk_app())
    _ = _create_branch(
        client,
        session_id="session-1",
        branch_id="branch-a",
        name="Branch A",
        hypothesis="Hypothesis A",
    )

    compared = client.post(
        "/api/proposal/branches/compare",
        json={"session_id": "session-1", "branch_ids": ["branch-a", "missing"]},
    )

    assert compared.status_code == 404
    payload = compared.json()
    assert payload["error"] == "branch_not_found"
