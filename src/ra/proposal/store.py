"""Persistence primitives for proposal workflow sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ra.proposal.models import STAGE_SEQUENCE, ProposalStage, ProposalWorkflowState, StageState


@dataclass(frozen=True, slots=True)
class ProposalSessionSnapshot:
    """Persisted workflow state for a proposal session."""

    session_id: str
    version: int
    state: ProposalWorkflowState


class SessionNotFoundError(KeyError):
    """Raised when a session snapshot cannot be found in the store."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session '{session_id}' was not found.")
        self.session_id = session_id


class SessionAlreadyExistsError(ValueError):
    """Raised when attempting to create a session that already exists."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session '{session_id}' already exists.")
        self.session_id = session_id


class SessionVersionConflictError(ValueError):
    """Raised when an optimistic concurrency write uses a stale version."""

    def __init__(self, session_id: str, expected_version: int, current_version: int) -> None:
        super().__init__(
            f"Version conflict for session '{session_id}': "
            f"expected {expected_version}, current {current_version}."
        )
        self.session_id = session_id
        self.expected_version = expected_version
        self.current_version = current_version


class ProposalSessionStore(Protocol):
    """Store protocol supporting versioned snapshot persistence."""

    def create_snapshot(
        self,
        session_id: str,
        state: ProposalWorkflowState,
    ) -> ProposalSessionSnapshot: ...

    def get_snapshot(self, session_id: str) -> ProposalSessionSnapshot | None: ...

    def save_snapshot(
        self,
        session_id: str,
        state: ProposalWorkflowState,
        *,
        expected_version: int,
    ) -> ProposalSessionSnapshot: ...

    def list_snapshots(self) -> tuple[ProposalSessionSnapshot, ...]: ...


class InMemoryProposalSessionStore:
    """In-memory snapshot store with optimistic concurrency checks."""

    def __init__(
        self,
        snapshots: Mapping[str, ProposalSessionSnapshot] | None = None,
    ) -> None:
        self._snapshots: dict[str, ProposalSessionSnapshot] = {}
        if snapshots:
            for session_id, snapshot in snapshots.items():
                self._snapshots[str(session_id)] = _clone_snapshot(snapshot)

    def create_snapshot(
        self,
        session_id: str,
        state: ProposalWorkflowState,
    ) -> ProposalSessionSnapshot:
        key = str(session_id)
        if key in self._snapshots:
            raise SessionAlreadyExistsError(key)
        snapshot = ProposalSessionSnapshot(session_id=key, version=0, state=_clone_state(state))
        self._snapshots[key] = snapshot
        return _clone_snapshot(snapshot)

    def get_snapshot(self, session_id: str) -> ProposalSessionSnapshot | None:
        snapshot = self._snapshots.get(str(session_id))
        if snapshot is None:
            return None
        return _clone_snapshot(snapshot)

    def save_snapshot(
        self,
        session_id: str,
        state: ProposalWorkflowState,
        *,
        expected_version: int,
    ) -> ProposalSessionSnapshot:
        key = str(session_id)
        current = self._snapshots.get(key)
        if current is None:
            raise SessionNotFoundError(key)
        if current.version != expected_version:
            raise SessionVersionConflictError(
                key,
                expected_version=expected_version,
                current_version=current.version,
            )
        snapshot = ProposalSessionSnapshot(
            session_id=key,
            version=current.version + 1,
            state=_clone_state(state),
        )
        self._snapshots[key] = snapshot
        return _clone_snapshot(snapshot)

    def list_snapshots(self) -> tuple[ProposalSessionSnapshot, ...]:
        return tuple(_clone_snapshot(snapshot) for _, snapshot in sorted(self._snapshots.items()))

    def dump_snapshots(self) -> dict[str, ProposalSessionSnapshot]:
        """Export snapshots for checkpointing/recovery."""

        return {session_id: _clone_snapshot(snapshot) for session_id, snapshot in self._snapshots.items()}

    @classmethod
    def from_snapshots(
        cls,
        snapshots: Mapping[str, ProposalSessionSnapshot],
    ) -> InMemoryProposalSessionStore:
        """Reconstruct store state from exported snapshots."""

        return cls(snapshots=snapshots)


def _clone_snapshot(snapshot: ProposalSessionSnapshot) -> ProposalSessionSnapshot:
    return ProposalSessionSnapshot(
        session_id=snapshot.session_id,
        version=snapshot.version,
        state=_clone_state(snapshot.state),
    )


def _clone_state(state: ProposalWorkflowState) -> ProposalWorkflowState:
    current_stage = _normalize_stage(state.current_stage) or STAGE_SEQUENCE[0]

    incoming = state.stage_states if isinstance(state.stage_states, Mapping) else {}
    normalized_stage_states: dict[ProposalStage, StageState] = {}
    for raw_stage, raw_stage_state in incoming.items():
        stage = _normalize_stage(raw_stage)
        if stage is None:
            continue
        normalized_stage_states[stage] = _coerce_stage_state(raw_stage_state)

    canonical_stage_states: dict[ProposalStage, StageState] = {}
    for stage in STAGE_SEQUENCE:
        canonical_stage_states[stage] = normalized_stage_states.get(stage, StageState())

    return ProposalWorkflowState(current_stage=current_stage, stage_states=canonical_stage_states)


def _normalize_stage(value: object) -> ProposalStage | None:
    if isinstance(value, ProposalStage):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return ProposalStage(candidate)
        except ValueError:
            return None
    return None


def _coerce_stage_state(value: object) -> StageState:
    if isinstance(value, StageState):
        return StageState(
            payload=_coerce_payload(value.payload),
            confirmed=bool(value.confirmed),
        )

    if isinstance(value, Mapping):
        payload = _coerce_payload(value.get("payload"))
        confirmed = bool(value.get("confirmed", False))
        return StageState(payload=payload, confirmed=confirmed)

    return StageState()


def _coerce_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): payload_value for key, payload_value in value.items()}
