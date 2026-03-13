"""Application service for proposal workflow sessions."""

from __future__ import annotations

from typing import Any, Mapping

from ra.proposal.models import ProposalStage
from ra.proposal.stage_engine import ProposalStageEngine
from ra.proposal.store import (
    ProposalSessionSnapshot,
    ProposalSessionStore,
    SessionNotFoundError,
    SessionVersionConflictError,
)


class ProposalSessionService:
    """Session-level operations with autosave on every mutation."""

    def __init__(
        self,
        *,
        store: ProposalSessionStore,
        stage_engine: ProposalStageEngine | None = None,
    ) -> None:
        self._store = store
        self._stage_engine = stage_engine or ProposalStageEngine()

    def create_session(self, session_id: str) -> ProposalSessionSnapshot:
        key = _normalize_session_id(session_id)
        state = self._stage_engine.create_initial_state()
        return self._store.create_snapshot(key, state)

    def get_session(self, session_id: str) -> ProposalSessionSnapshot:
        key = _normalize_session_id(session_id)
        snapshot = self._store.get_snapshot(key)
        if snapshot is None:
            raise SessionNotFoundError(key)
        return snapshot

    def update_stage_payload(
        self,
        session_id: str,
        stage: ProposalStage,
        payload: Mapping[str, Any],
        *,
        expected_version: int,
    ) -> ProposalSessionSnapshot:
        current = self.get_session(session_id)
        _assert_expected_version(current, expected_version)
        updated_state = self._stage_engine.update_stage_payload(current.state, stage, payload)
        return self._store.save_snapshot(
            current.session_id,
            updated_state,
            expected_version=expected_version,
        )

    def advance_stage(
        self,
        session_id: str,
        *,
        expected_version: int,
    ) -> ProposalSessionSnapshot:
        current = self.get_session(session_id)
        _assert_expected_version(current, expected_version)
        advanced_state = self._stage_engine.advance_stage(current.state)
        return self._store.save_snapshot(
            current.session_id,
            advanced_state,
            expected_version=expected_version,
        )


def _normalize_session_id(session_id: str) -> str:
    key = str(session_id).strip()
    if not key:
        raise ValueError("session_id must not be empty")
    return key


def _assert_expected_version(
    snapshot: ProposalSessionSnapshot,
    expected_version: int,
) -> None:
    if snapshot.version != expected_version:
        raise SessionVersionConflictError(
            snapshot.session_id,
            expected_version=expected_version,
            current_version=snapshot.version,
        )
