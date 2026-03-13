"""Pydantic request/response models for proposal workflow endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ra.proposal import (
    BranchComparisonItem,
    BranchComparisonResult,
    BranchConfidenceLabel,
    BranchScorecard,
    HypothesisBranch,
    ProposalSessionSnapshot,
    ProposalStage,
)


class ProposalSessionCreateRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"example": {"session_id": "proposal-session-1"}},
    )

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Client-controlled identifier for proposal workflow state.",
        examples=["proposal-session-1"],
    )


class ProposalStageUpdateRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "expected_version": 2,
                "payload": {
                    "supporting_evidence": ["paper-1", "paper-2"],
                    "contradicting_evidence": ["paper-3"],
                    "landscape_summary": "Consensus exists, but methods are heterogeneous.",
                },
            }
        },
    )

    expected_version: int = Field(
        ...,
        ge=0,
        description="Expected current session version for optimistic concurrency control.",
        examples=[2],
    )
    payload: dict[str, Any] = Field(
        ...,
        min_length=1,
        description="Partial key/value payload to merge into the specified stage snapshot.",
    )


class ProposalStageAdvanceRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"expected_version": 3}},
    )

    expected_version: int = Field(
        ...,
        ge=0,
        description="Expected current session version before attempting stage advancement.",
        examples=[3],
    )


class ProposalStageStateResponse(BaseModel):
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Captured artifact payload for this stage.",
    )
    confirmed: bool = Field(
        ...,
        description=(
            "Whether this stage currently satisfies all required fields "
            "according to the stage engine."
        ),
    )


class ProposalWorkflowStateResponse(BaseModel):
    current_stage: ProposalStage = Field(..., description="Current active stage in the workflow.")
    stage_states: dict[ProposalStage, ProposalStageStateResponse] = Field(
        ...,
        description="Per-stage payload snapshots keyed by canonical stage name.",
    )


class ProposalSessionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "proposal-session-1",
                "version": 0,
                "state": {
                    "current_stage": "idea_intake",
                    "stage_states": {
                        "idea_intake": {"payload": {}, "confirmed": False},
                        "logic_refinement": {"payload": {}, "confirmed": False},
                    },
                },
            }
        }
    )

    session_id: str = Field(..., description="Proposal session identifier.")
    version: int = Field(..., ge=0, description="Monotonic session version after each write.")
    state: ProposalWorkflowStateResponse = Field(
        ...,
        description="Current workflow state snapshot.",
    )

    @classmethod
    def from_snapshot(cls, snapshot: ProposalSessionSnapshot) -> ProposalSessionResponse:
        stage_states = {
            stage: ProposalStageStateResponse(
                payload=dict(stage_state.payload),
                confirmed=stage_state.confirmed,
            )
            for stage, stage_state in snapshot.state.stage_states.items()
        }
        return cls(
            session_id=snapshot.session_id,
            version=snapshot.version,
            state=ProposalWorkflowStateResponse(
                current_stage=snapshot.state.current_stage,
                stage_states=stage_states,
            ),
        )


class ProposalBranchScorecard(BaseModel):
    evidence_support: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Evidence support strength for this branch.",
    )
    feasibility: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated feasibility score for this branch.",
    )
    risk: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated risk score where higher means more risk.",
    )
    impact: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Expected contribution/impact score for this branch.",
    )

    def to_domain(self) -> BranchScorecard:
        return BranchScorecard(
            evidence_support=self.evidence_support,
            feasibility=self.feasibility,
            risk=self.risk,
            impact=self.impact,
        )

    @classmethod
    def from_domain(cls, scorecard: BranchScorecard) -> ProposalBranchScorecard:
        return cls(
            evidence_support=scorecard.evidence_support,
            feasibility=scorecard.feasibility,
            risk=scorecard.risk,
            impact=scorecard.impact,
        )


class ProposalBranchCreateRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "session_id": "proposal-session-1",
                "branch_id": "branch-a",
                "name": "Primary mechanism",
                "hypothesis": "Mechanism A drives outcome B.",
                "scorecard": {
                    "evidence_support": 0.8,
                    "feasibility": 0.7,
                    "risk": 0.3,
                    "impact": 0.9,
                },
            }
        },
    )

    session_id: str = Field(..., min_length=1, max_length=128, description="Proposal session ID.")
    branch_id: str = Field(..., min_length=1, max_length=128, description="Unique branch ID.")
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable branch name.")
    hypothesis: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Branch hypothesis text.",
    )
    scorecard: ProposalBranchScorecard = Field(..., description="Branch scoring dimensions.")
    parent_branch_id: str | None = Field(
        None,
        min_length=1,
        max_length=128,
        description="Optional parent branch ID for forked branches.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata fields attached to the hypothesis node.",
    )


class ProposalBranchResponse(BaseModel):
    session_id: str
    branch_id: str
    name: str
    hypothesis: str
    scorecard: ProposalBranchScorecard
    confidence_label: BranchConfidenceLabel
    metadata: dict[str, str] = Field(default_factory=dict)
    parent_branch_id: str | None = None
    lineage: list[str] = Field(default_factory=list)
    is_primary: bool

    @classmethod
    def from_domain(cls, branch: HypothesisBranch) -> ProposalBranchResponse:
        return cls(
            session_id=branch.session_id,
            branch_id=branch.branch_id,
            name=branch.name,
            hypothesis=branch.hypothesis,
            scorecard=ProposalBranchScorecard.from_domain(branch.scorecard),
            confidence_label=branch.confidence_label,
            metadata=dict(branch.metadata),
            parent_branch_id=branch.parent_branch_id,
            lineage=list(branch.lineage),
            is_primary=branch.is_primary,
        )


class ProposalBranchListResponse(BaseModel):
    session_id: str
    count: int
    branches: list[ProposalBranchResponse]


class ProposalBranchCompareRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "session_id": "proposal-session-1",
                "branch_ids": ["branch-a", "branch-b"],
            }
        },
    )

    session_id: str = Field(..., min_length=1, max_length=128, description="Proposal session ID.")
    branch_ids: list[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Branch IDs to compare.",
    )


class ProposalBranchComparisonItemResponse(BaseModel):
    branch_id: str
    name: str
    scorecard: ProposalBranchScorecard
    confidence_label: BranchConfidenceLabel
    aggregate_score: float = Field(..., ge=0.0, le=1.0)
    is_primary: bool

    @classmethod
    def from_domain(
        cls,
        item: BranchComparisonItem,
    ) -> ProposalBranchComparisonItemResponse:
        return cls(
            branch_id=item.branch_id,
            name=item.name,
            scorecard=ProposalBranchScorecard.from_domain(item.scorecard),
            confidence_label=item.confidence_label,
            aggregate_score=item.aggregate_score,
            is_primary=item.is_primary,
        )


class ProposalBranchCompareResponse(BaseModel):
    session_id: str
    winner_branch_id: str
    comparisons: list[ProposalBranchComparisonItemResponse]

    @classmethod
    def from_domain(
        cls,
        result: BranchComparisonResult,
    ) -> ProposalBranchCompareResponse:
        return cls(
            session_id=result.session_id,
            winner_branch_id=result.winner_branch_id,
            comparisons=[
                ProposalBranchComparisonItemResponse.from_domain(item)
                for item in result.comparisons
            ],
        )
