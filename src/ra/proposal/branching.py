"""Hypothesis branch operations for proposal workflow exploration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum


class BranchConfidenceLabel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class BranchScorecard:
    evidence_support: float
    feasibility: float
    risk: float
    impact: float

    def __post_init__(self) -> None:
        for field_name in ("evidence_support", "feasibility", "risk", "impact"):
            value = float(getattr(self, field_name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class HypothesisBranch:
    session_id: str
    branch_id: str
    name: str
    hypothesis: str
    scorecard: BranchScorecard
    confidence_label: BranchConfidenceLabel
    metadata: dict[str, str] = field(default_factory=dict)
    parent_branch_id: str | None = None
    lineage: tuple[str, ...] = ()
    is_primary: bool = False


@dataclass(frozen=True, slots=True)
class BranchComparisonItem:
    branch_id: str
    name: str
    scorecard: BranchScorecard
    confidence_label: BranchConfidenceLabel
    aggregate_score: float
    is_primary: bool


@dataclass(frozen=True, slots=True)
class BranchComparisonResult:
    session_id: str
    comparisons: tuple[BranchComparisonItem, ...]
    winner_branch_id: str


class BranchNotFoundError(KeyError):
    def __init__(self, session_id: str, branch_id: str) -> None:
        super().__init__(f"Branch '{branch_id}' was not found for session '{session_id}'.")
        self.session_id = session_id
        self.branch_id = branch_id


class BranchAlreadyExistsError(ValueError):
    def __init__(self, session_id: str, branch_id: str) -> None:
        super().__init__(f"Branch '{branch_id}' already exists for session '{session_id}'.")
        self.session_id = session_id
        self.branch_id = branch_id


class HypothesisBranchManager:
    """In-memory branch manager with deterministic compare/promote behavior."""

    def __init__(self) -> None:
        self._branches_by_session: dict[str, dict[str, HypothesisBranch]] = {}
        self._branch_order_by_session: dict[str, list[str]] = {}

    def create_branch(
        self,
        *,
        session_id: str,
        branch_id: str,
        name: str,
        hypothesis: str,
        scorecard: BranchScorecard | None = None,
        metadata: dict[str, str] | None = None,
        parent_branch_id: str | None = None,
    ) -> HypothesisBranch:
        sid = _normalize_key(session_id, field_name="session_id")
        bid = _normalize_key(branch_id, field_name="branch_id")
        branch_name = _normalize_key(name, field_name="name")
        branch_hypothesis = _normalize_key(hypothesis, field_name="hypothesis")
        session_branches = self._branches_by_session.setdefault(sid, {})
        session_order = self._branch_order_by_session.setdefault(sid, [])
        if bid in session_branches:
            raise BranchAlreadyExistsError(sid, bid)

        parent: HypothesisBranch | None = None
        if parent_branch_id is not None:
            pid = _normalize_key(parent_branch_id, field_name="parent_branch_id")
            parent = session_branches.get(pid)
            if parent is None:
                raise BranchNotFoundError(sid, pid)

        lineage = () if parent is None else parent.lineage + (parent.branch_id,)
        has_primary = any(branch.is_primary for branch in session_branches.values())
        branch_scorecard = scorecard or _default_scorecard()
        branch = HypothesisBranch(
            session_id=sid,
            branch_id=bid,
            name=branch_name,
            hypothesis=branch_hypothesis,
            scorecard=branch_scorecard,
            confidence_label=_confidence_label_for_scorecard(branch_scorecard),
            metadata=_normalize_metadata(metadata if metadata is not None else (parent.metadata if parent else {})),
            parent_branch_id=parent.branch_id if parent else None,
            lineage=lineage,
            is_primary=not has_primary,
        )
        session_branches[bid] = branch
        session_order.append(bid)
        return branch

    def fork_branch(
        self,
        *,
        session_id: str,
        source_branch_id: str,
        new_branch_id: str,
        name: str,
        hypothesis: str | None = None,
        scorecard: BranchScorecard | None = None,
    ) -> HypothesisBranch:
        source = self.get_branch(session_id=session_id, branch_id=source_branch_id)
        return self.create_branch(
            session_id=session_id,
            branch_id=new_branch_id,
            name=name,
            hypothesis=hypothesis or source.hypothesis,
            scorecard=scorecard or source.scorecard,
            metadata=dict(source.metadata),
            parent_branch_id=source.branch_id,
        )

    def list_branches(self, session_id: str) -> tuple[HypothesisBranch, ...]:
        sid = _normalize_key(session_id, field_name="session_id")
        session_branches = self._branches_by_session.get(sid, {})
        order = self._branch_order_by_session.get(sid, [])
        return tuple(session_branches[bid] for bid in order if bid in session_branches)

    def get_branch(self, *, session_id: str, branch_id: str) -> HypothesisBranch:
        sid = _normalize_key(session_id, field_name="session_id")
        bid = _normalize_key(branch_id, field_name="branch_id")
        session_branches = self._branches_by_session.get(sid, {})
        branch = session_branches.get(bid)
        if branch is None:
            raise BranchNotFoundError(sid, bid)
        return branch

    def promote_branch(self, *, session_id: str, branch_id: str) -> HypothesisBranch:
        sid = _normalize_key(session_id, field_name="session_id")
        bid = _normalize_key(branch_id, field_name="branch_id")
        session_branches = self._branches_by_session.get(sid, {})
        if bid not in session_branches:
            raise BranchNotFoundError(sid, bid)

        updated: dict[str, HypothesisBranch] = {}
        for existing_id, branch in session_branches.items():
            updated[existing_id] = replace(branch, is_primary=(existing_id == bid))
        self._branches_by_session[sid] = updated
        return updated[bid]

    def compare_branches(
        self,
        *,
        session_id: str,
        branch_ids: tuple[str, ...],
    ) -> BranchComparisonResult:
        sid = _normalize_key(session_id, field_name="session_id")
        ordered_ids = tuple(
            dict.fromkeys(
                _normalize_key(branch_id, field_name="branch_id")
                for branch_id in branch_ids
            )
        )
        if len(ordered_ids) < 2:
            raise ValueError("branch_ids must include at least two unique branch IDs")

        items: list[BranchComparisonItem] = []
        for bid in ordered_ids:
            branch = self.get_branch(session_id=sid, branch_id=bid)
            aggregate = _aggregate_score(branch.scorecard)
            items.append(
                BranchComparisonItem(
                    branch_id=branch.branch_id,
                    name=branch.name,
                    scorecard=branch.scorecard,
                    confidence_label=branch.confidence_label,
                    aggregate_score=aggregate,
                    is_primary=branch.is_primary,
                )
            )

        ranked = tuple(
            sorted(
                items,
                key=lambda item: (-item.aggregate_score, item.branch_id),
            )
        )
        winner_branch_id = ranked[0].branch_id
        return BranchComparisonResult(
            session_id=sid,
            comparisons=ranked,
            winner_branch_id=winner_branch_id,
        )


def _normalize_key(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _default_scorecard() -> BranchScorecard:
    return BranchScorecard(
        evidence_support=0.5,
        feasibility=0.5,
        risk=0.5,
        impact=0.5,
    )


def _normalize_metadata(metadata: dict[str, str] | None) -> dict[str, str]:
    if metadata is None:
        return {}
    return {str(key).strip(): str(value).strip() for key, value in metadata.items() if str(key).strip()}


def _confidence_label_for_scorecard(scorecard: BranchScorecard) -> BranchConfidenceLabel:
    aggregate = _aggregate_score(scorecard)
    if aggregate >= 0.75:
        return BranchConfidenceLabel.HIGH
    if aggregate >= 0.5:
        return BranchConfidenceLabel.MEDIUM
    return BranchConfidenceLabel.LOW


def _aggregate_score(scorecard: BranchScorecard) -> float:
    score = (
        scorecard.evidence_support
        + scorecard.feasibility
        + (1.0 - scorecard.risk)
        + scorecard.impact
    ) / 4.0
    return round(max(0.0, min(1.0, score)), 6)
