from __future__ import annotations

from fastapi.testclient import TestClient

from ra.api import create_app
from ra.proposal import ProposalStage


class _StubRetriever:
    async def search(self, query: str, limit: int, sources: tuple[str, ...]):  # noqa: ARG002
        return []

    async def get_paper(self, identifier: str):  # noqa: ARG002
        return None

    async def close(self) -> None:
        return None


def _mk_app():
    return create_app(retriever_factory=lambda: _StubRetriever())


def _create_session(client: TestClient, *, session_id: str = "session-1") -> dict[str, object]:
    resp = client.post("/api/proposal/sessions", json={"session_id": session_id})
    assert resp.status_code == 201
    return resp.json()


def test_create_session_endpoint_returns_initial_snapshot() -> None:
    client = TestClient(_mk_app())

    resp = client.post("/api/proposal/sessions", json={"session_id": "session-1"})

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["session_id"] == "session-1"
    assert payload["version"] == 0
    assert payload["state"]["current_stage"] == ProposalStage.IDEA_INTAKE.value

    stage_states = payload["state"]["stage_states"]
    assert stage_states["idea_intake"]["payload"] == {}
    assert stage_states["idea_intake"]["confirmed"] is False


def test_get_session_endpoint_returns_existing_snapshot() -> None:
    client = TestClient(_mk_app())
    _ = _create_session(client, session_id="session-1")

    resp = client.get("/api/proposal/sessions/session-1")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["session_id"] == "session-1"
    assert payload["version"] == 0
    assert payload["state"]["current_stage"] == ProposalStage.IDEA_INTAKE.value


def test_update_stage_endpoint_autosaves_and_confirms_when_complete() -> None:
    client = TestClient(_mk_app())
    _ = _create_session(client, session_id="session-1")

    partial = client.patch(
        "/api/proposal/sessions/session-1/stages/idea_intake",
        json={
            "expected_version": 0,
            "payload": {
                "problem": "Unsupported claims in generated summaries.",
                "target_population": "Clinical NLP researchers.",
            },
        },
    )

    assert partial.status_code == 200
    partial_payload = partial.json()
    assert partial_payload["version"] == 1
    assert partial_payload["state"]["stage_states"]["idea_intake"]["confirmed"] is False

    complete = client.patch(
        "/api/proposal/sessions/session-1/stages/idea_intake",
        json={
            "expected_version": 1,
            "payload": {
                "mechanism": "Retrieval-grounded synthesis with citation checks.",
                "expected_outcome": "Higher precision in citation-backed claims.",
            },
        },
    )

    assert complete.status_code == 200
    complete_payload = complete.json()
    assert complete_payload["version"] == 2
    assert complete_payload["state"]["stage_states"]["idea_intake"]["confirmed"] is True


def test_advance_stage_endpoint_moves_to_next_stage_when_current_is_complete() -> None:
    client = TestClient(_mk_app())
    _ = _create_session(client, session_id="session-1")
    updated = client.patch(
        "/api/proposal/sessions/session-1/stages/idea_intake",
        json={
            "expected_version": 0,
            "payload": {
                "problem": "Unsupported claims in generated summaries.",
                "target_population": "Clinical NLP researchers.",
                "mechanism": "Retrieval-grounded synthesis with citation checks.",
                "expected_outcome": "Higher precision in citation-backed claims.",
            },
        },
    )
    assert updated.status_code == 200

    resp = client.post("/api/proposal/sessions/session-1/advance", json={"expected_version": 1})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["version"] == 2
    assert payload["state"]["current_stage"] == ProposalStage.LOGIC_REFINEMENT.value


def test_advance_stage_endpoint_rejects_incomplete_stage_with_structured_error() -> None:
    client = TestClient(_mk_app())
    _ = _create_session(client, session_id="session-1")

    resp = client.post("/api/proposal/sessions/session-1/advance", json={"expected_version": 0})

    assert resp.status_code == 409
    payload = resp.json()
    assert payload["error"] == "stage_transition_rejected"
    assert payload["details"][0]["reason"] == "incomplete_stage"
    assert "problem" in payload["details"][0]["missing_fields"]


def test_update_stage_endpoint_rejects_stale_version_with_conflict_error() -> None:
    client = TestClient(_mk_app())
    _ = _create_session(client, session_id="session-1")
    first = client.patch(
        "/api/proposal/sessions/session-1/stages/idea_intake",
        json={
            "expected_version": 0,
            "payload": {"problem": "P", "target_population": "T"},
        },
    )
    assert first.status_code == 200

    stale = client.patch(
        "/api/proposal/sessions/session-1/stages/idea_intake",
        json={
            "expected_version": 0,
            "payload": {"mechanism": "M"},
        },
    )

    assert stale.status_code == 409
    payload = stale.json()
    assert payload["error"] == "version_conflict"
    assert payload["details"][0]["session_id"] == "session-1"
    assert payload["details"][0]["expected_version"] == 0
    assert payload["details"][0]["current_version"] == 1


def test_get_session_endpoint_returns_not_found_for_unknown_session() -> None:
    client = TestClient(_mk_app())

    resp = client.get("/api/proposal/sessions/unknown-session")

    assert resp.status_code == 404
    payload = resp.json()
    assert payload["error"] == "session_not_found"


def test_create_session_endpoint_rejects_duplicate_session_id() -> None:
    client = TestClient(_mk_app())
    first = client.post("/api/proposal/sessions", json={"session_id": "session-1"})
    second = client.post("/api/proposal/sessions", json={"session_id": "session-1"})

    assert first.status_code == 201
    assert second.status_code == 409
    payload = second.json()
    assert payload["error"] == "session_exists"


def test_openapi_documents_proposal_workflow_examples() -> None:
    client = TestClient(_mk_app())

    resp = client.get("/openapi.json")

    assert resp.status_code == 200
    schema = resp.json()

    create_operation = schema["paths"]["/api/proposal/sessions"]["post"]
    create_examples = create_operation["requestBody"]["content"]["application/json"]["examples"]
    assert create_examples["new_session"]["value"]["session_id"] == "proposal-session-1"
    assert (
        create_operation["responses"]["409"]["content"]["application/json"]["example"]["error"]
        == "session_exists"
    )

    update_operation = schema["paths"]["/api/proposal/sessions/{session_id}/stages/{stage}"][
        "patch"
    ]
    update_examples = update_operation["requestBody"]["content"]["application/json"]["examples"]
    assert update_examples["stage_payload_patch"]["value"]["expected_version"] == 2

    advance_operation = schema["paths"]["/api/proposal/sessions/{session_id}/advance"]["post"]
    advance_examples = advance_operation["requestBody"]["content"]["application/json"]["examples"]
    assert advance_examples["next_stage"]["value"]["expected_version"] == 3
