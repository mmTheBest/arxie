"""Domain models for the v0.2 proposal workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProposalStage(str, Enum):
    """Canonical, ordered stages for proposal co-creation."""

    IDEA_INTAKE = "idea_intake"
    LOGIC_REFINEMENT = "logic_refinement"
    EVIDENCE_MAPPING = "evidence_mapping"
    HYPOTHESIS_RESHAPING = "hypothesis_reshaping"
    DATA_FEASIBILITY_PLANNING = "data_feasibility_planning"
    EXPERIMENT_ANALYSIS_DESIGN = "experiment_analysis_design"
    PROPOSAL_ASSEMBLY = "proposal_assembly"


STAGE_SEQUENCE: tuple[ProposalStage, ...] = (
    ProposalStage.IDEA_INTAKE,
    ProposalStage.LOGIC_REFINEMENT,
    ProposalStage.EVIDENCE_MAPPING,
    ProposalStage.HYPOTHESIS_RESHAPING,
    ProposalStage.DATA_FEASIBILITY_PLANNING,
    ProposalStage.EXPERIMENT_ANALYSIS_DESIGN,
    ProposalStage.PROPOSAL_ASSEMBLY,
)


@dataclass(frozen=True, slots=True)
class StageState:
    """Mutable stage artifacts captured as immutable snapshots per write."""

    payload: dict[str, Any] = field(default_factory=dict)
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class ProposalWorkflowState:
    """Top-level workflow state for stage-gated proposal progression."""

    current_stage: ProposalStage
    stage_states: dict[ProposalStage, StageState]


def create_empty_workflow_state() -> ProposalWorkflowState:
    """Create deterministic initial workflow state."""

    stage_states = {stage: StageState() for stage in STAGE_SEQUENCE}
    return ProposalWorkflowState(current_stage=ProposalStage.IDEA_INTAKE, stage_states=stage_states)
