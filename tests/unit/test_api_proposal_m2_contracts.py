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


def test_proposal_conversation_contract_supports_write_then_read() -> None:
    client = TestClient(_mk_app())

    posted = client.post(
        "/api/proposal/conversations/session-1/messages",
        json={
            "role": "user",
            "content": "Draft a stage summary for idea intake.",
            "metadata": {"ui_source": "dashboard"},
        },
    )

    assert posted.status_code == 201
    message = posted.json()
    assert message["session_id"] == "session-1"
    assert message["role"] == "user"
    assert message["content"] == "Draft a stage summary for idea intake."
    assert message["metadata"]["ui_source"] == "dashboard"

    listed = client.get("/api/proposal/conversations/session-1/messages")
    assert listed.status_code == 200
    payload = listed.json()
    assert payload["session_id"] == "session-1"
    assert payload["count"] == 1
    assert payload["messages"][0]["content"] == "Draft a stage summary for idea intake."


def test_proposal_evidence_inspector_contract_returns_placeholder_schema() -> None:
    client = TestClient(_mk_app())

    response = client.get("/api/proposal/evidence/session-1/inspector")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "session-1"
    assert payload["count"] == 0
    assert payload["items"] == []
