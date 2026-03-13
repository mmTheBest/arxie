from __future__ import annotations

import pytest

from ra.proposal import (
    BranchAlreadyExistsError,
    BranchNotFoundError,
    BranchScorecard,
    HypothesisBranchManager,
)


def _score(
    *,
    evidence_support: float,
    feasibility: float,
    risk: float,
    impact: float,
) -> BranchScorecard:
    return BranchScorecard(
        evidence_support=evidence_support,
        feasibility=feasibility,
        risk=risk,
        impact=impact,
    )


def test_create_and_fork_branch_preserves_lineage_integrity() -> None:
    manager = HypothesisBranchManager()
    root = manager.create_branch(
        session_id="session-1",
        branch_id="root",
        name="Root hypothesis",
        hypothesis="Primary causal hypothesis.",
        scorecard=_score(evidence_support=0.7, feasibility=0.8, risk=0.3, impact=0.9),
        metadata={"owner": "m3-backend"},
    )

    fork = manager.fork_branch(
        session_id="session-1",
        source_branch_id="root",
        new_branch_id="alt",
        name="Alternative mechanism",
        hypothesis="Alternative causal explanation.",
        scorecard=_score(evidence_support=0.6, feasibility=0.7, risk=0.4, impact=0.8),
    )

    assert root.parent_branch_id is None
    assert root.lineage == ()
    assert root.metadata["owner"] == "m3-backend"
    assert fork.parent_branch_id == "root"
    assert fork.lineage == ("root",)
    assert fork.metadata["owner"] == "m3-backend"


def test_branch_confidence_label_derives_from_scorecard() -> None:
    manager = HypothesisBranchManager()
    high = manager.create_branch(
        session_id="session-1",
        branch_id="high",
        name="High confidence",
        hypothesis="High confidence hypothesis.",
        scorecard=_score(evidence_support=0.95, feasibility=0.9, risk=0.1, impact=0.9),
    )
    low = manager.create_branch(
        session_id="session-1",
        branch_id="low",
        name="Low confidence",
        hypothesis="Low confidence hypothesis.",
        scorecard=_score(evidence_support=0.2, feasibility=0.2, risk=0.9, impact=0.1),
    )

    assert high.confidence_label.value == "high"
    assert low.confidence_label.value == "low"


def test_promote_branch_marks_only_one_primary_branch() -> None:
    manager = HypothesisBranchManager()
    _ = manager.create_branch(
        session_id="session-1",
        branch_id="root",
        name="Root hypothesis",
        hypothesis="Primary causal hypothesis.",
        scorecard=_score(evidence_support=0.7, feasibility=0.8, risk=0.3, impact=0.9),
    )
    _ = manager.create_branch(
        session_id="session-1",
        branch_id="alt",
        name="Alternative hypothesis",
        hypothesis="Alternative explanation.",
        scorecard=_score(evidence_support=0.6, feasibility=0.7, risk=0.4, impact=0.8),
    )

    promoted = manager.promote_branch(session_id="session-1", branch_id="alt")
    branches = manager.list_branches("session-1")
    primary_ids = [branch.branch_id for branch in branches if branch.is_primary]

    assert promoted.branch_id == "alt"
    assert primary_ids == ["alt"]


def test_compare_branches_returns_expected_scoring_shape_and_winner() -> None:
    manager = HypothesisBranchManager()
    _ = manager.create_branch(
        session_id="session-1",
        branch_id="branch-a",
        name="Branch A",
        hypothesis="Hypothesis A",
        scorecard=_score(evidence_support=0.9, feasibility=0.8, risk=0.1, impact=0.9),
    )
    _ = manager.create_branch(
        session_id="session-1",
        branch_id="branch-b",
        name="Branch B",
        hypothesis="Hypothesis B",
        scorecard=_score(evidence_support=0.6, feasibility=0.6, risk=0.4, impact=0.7),
    )

    comparison = manager.compare_branches(
        session_id="session-1",
        branch_ids=("branch-a", "branch-b"),
    )

    assert comparison.winner_branch_id == "branch-a"
    assert len(comparison.comparisons) == 2
    assert all(0.0 <= item.aggregate_score <= 1.0 for item in comparison.comparisons)
    assert comparison.comparisons[0].scorecard.evidence_support >= 0.0


def test_fork_branch_rejects_unknown_source_branch() -> None:
    manager = HypothesisBranchManager()

    with pytest.raises(BranchNotFoundError):
        _ = manager.fork_branch(
            session_id="session-1",
            source_branch_id="missing",
            new_branch_id="alt",
            name="Alternative mechanism",
        )


def test_create_branch_rejects_duplicate_branch_id() -> None:
    manager = HypothesisBranchManager()
    _ = manager.create_branch(
        session_id="session-1",
        branch_id="root",
        name="Root hypothesis",
        hypothesis="Primary causal hypothesis.",
        scorecard=_score(evidence_support=0.7, feasibility=0.8, risk=0.3, impact=0.9),
    )

    with pytest.raises(BranchAlreadyExistsError):
        _ = manager.create_branch(
            session_id="session-1",
            branch_id="root",
            name="Duplicate",
            hypothesis="Duplicate hypothesis.",
            scorecard=_score(evidence_support=0.1, feasibility=0.2, risk=0.9, impact=0.3),
        )
