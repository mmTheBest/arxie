"""Deterministic stage transitions for proposal workflow state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from ra.proposal.models import (
    STAGE_SEQUENCE,
    ProposalStage,
    ProposalWorkflowState,
    StageState,
    create_empty_workflow_state,
)


class StageTransitionReason(str, Enum):
    """Machine-readable transition rejection reasons."""

    INVALID_TRANSITION = "invalid_transition"
    INCOMPLETE_STAGE = "incomplete_stage"
    FINAL_STAGE_REACHED = "final_stage_reached"


class StageTransitionError(ValueError):
    """Typed stage transition error with actionable metadata."""

    def __init__(
        self,
        *,
        reason: StageTransitionReason,
        from_stage: ProposalStage,
        to_stage: ProposalStage,
        message: str,
        missing_fields: tuple[str, ...] = (),
        allowed_next_stage: ProposalStage | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.from_stage = from_stage
        self.to_stage = to_stage
        self.missing_fields = missing_fields
        self.allowed_next_stage = allowed_next_stage


@dataclass(frozen=True, slots=True)
class StageDiagnostics:
    """Completeness details for a specific stage snapshot."""

    stage: ProposalStage
    required_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    is_complete: bool


_REQUIRED_FIELDS: dict[ProposalStage, tuple[str, ...]] = {
    ProposalStage.IDEA_INTAKE: (
        "problem",
        "target_population",
        "mechanism",
        "expected_outcome",
    ),
    ProposalStage.LOGIC_REFINEMENT: (
        "problem_gap_chain",
        "core_assumptions",
        "testable_hypothesis",
    ),
    ProposalStage.EVIDENCE_MAPPING: (
        "supporting_evidence",
        "contradicting_evidence",
        "landscape_summary",
    ),
    ProposalStage.HYPOTHESIS_RESHAPING: (
        "candidate_hypotheses",
        "falsification_criteria",
        "selected_primary_hypothesis",
    ),
    ProposalStage.DATA_FEASIBILITY_PLANNING: (
        "candidate_datasets",
        "feasibility_constraints",
        "selected_data_strategy",
    ),
    ProposalStage.EXPERIMENT_ANALYSIS_DESIGN: (
        "experiment_design",
        "analysis_plan",
        "controls_and_confounders",
    ),
    ProposalStage.PROPOSAL_ASSEMBLY: (
        "background_and_gap",
        "hypothesis_statement",
        "method_summary",
        "analysis_summary",
        "expected_outcomes",
        "risks_and_limitations",
    ),
}


class ProposalStageEngine:
    """Pure stage engine for deterministic state transitions."""

    stage_sequence: tuple[ProposalStage, ...] = STAGE_SEQUENCE

    def create_initial_state(self) -> ProposalWorkflowState:
        return create_empty_workflow_state()

    def required_fields(self, stage: ProposalStage) -> tuple[str, ...]:
        return _REQUIRED_FIELDS[stage]

    def required_fields_payload(self, stage: ProposalStage) -> dict[str, str]:
        """Build a minimal valid payload for tests and fixtures."""

        return {field_name: f"{stage.value}:{field_name}" for field_name in self.required_fields(stage)}

    def diagnose_stage(
        self,
        state: ProposalWorkflowState,
        stage: ProposalStage,
    ) -> StageDiagnostics:
        stage_state = state.stage_states[stage]
        required = self.required_fields(stage)
        missing = self._missing_fields(stage_state.payload, required)
        return StageDiagnostics(
            stage=stage,
            required_fields=required,
            missing_fields=missing,
            is_complete=stage_state.confirmed and not missing,
        )

    def is_stage_complete(
        self,
        state: ProposalWorkflowState,
        stage: ProposalStage | None = None,
    ) -> bool:
        target_stage = stage or state.current_stage
        return self.diagnose_stage(state, target_stage).is_complete

    def update_stage_payload(
        self,
        state: ProposalWorkflowState,
        stage: ProposalStage,
        payload: Mapping[str, Any],
    ) -> ProposalWorkflowState:
        current = state.stage_states[stage]
        merged_payload = dict(current.payload)
        merged_payload.update(dict(payload))

        missing = self._missing_fields(merged_payload, self.required_fields(stage))
        updated_stage = StageState(payload=merged_payload, confirmed=not missing)

        updated_states = dict(state.stage_states)
        updated_states[stage] = updated_stage
        return ProposalWorkflowState(current_stage=state.current_stage, stage_states=updated_states)

    def transition_to(
        self,
        state: ProposalWorkflowState,
        target_stage: ProposalStage,
    ) -> ProposalWorkflowState:
        current_stage = state.current_stage
        if target_stage is current_stage:
            return state

        allowed_next_stage = self._next_stage(current_stage)
        if allowed_next_stage is None:
            raise StageTransitionError(
                reason=StageTransitionReason.FINAL_STAGE_REACHED,
                from_stage=current_stage,
                to_stage=current_stage,
                message="Final stage already reached; no further transitions are allowed.",
            )

        if target_stage is not allowed_next_stage:
            raise StageTransitionError(
                reason=StageTransitionReason.INVALID_TRANSITION,
                from_stage=current_stage,
                to_stage=target_stage,
                allowed_next_stage=allowed_next_stage,
                message=(
                    f"Invalid transition from '{current_stage.value}' to '{target_stage.value}'. "
                    f"Only '{allowed_next_stage.value}' is allowed next."
                ),
            )

        diagnostics = self.diagnose_stage(state, current_stage)
        if not diagnostics.is_complete:
            missing = diagnostics.missing_fields
            missing_str = ", ".join(missing) if missing else "none"
            raise StageTransitionError(
                reason=StageTransitionReason.INCOMPLETE_STAGE,
                from_stage=current_stage,
                to_stage=allowed_next_stage,
                missing_fields=missing,
                message=(
                    f"Stage '{current_stage.value}' is incomplete. "
                    f"Fill missing fields: {missing_str}."
                ),
            )

        return ProposalWorkflowState(current_stage=target_stage, stage_states=dict(state.stage_states))

    def advance_stage(self, state: ProposalWorkflowState) -> ProposalWorkflowState:
        current_stage = state.current_stage
        next_stage = self._next_stage(current_stage)
        if next_stage is None:
            raise StageTransitionError(
                reason=StageTransitionReason.FINAL_STAGE_REACHED,
                from_stage=current_stage,
                to_stage=current_stage,
                message="Final stage already reached; no further transitions are allowed.",
            )
        return self.transition_to(state, next_stage)

    def _next_stage(self, stage: ProposalStage) -> ProposalStage | None:
        index = self.stage_sequence.index(stage)
        next_index = index + 1
        if next_index >= len(self.stage_sequence):
            return None
        return self.stage_sequence[next_index]

    @staticmethod
    def _missing_fields(
        payload: Mapping[str, Any],
        required_fields: tuple[str, ...],
    ) -> tuple[str, ...]:
        missing: list[str] = []
        for field_name in required_fields:
            value = payload.get(field_name)
            if value is None:
                missing.append(field_name)
                continue
            if isinstance(value, str) and not value.strip():
                missing.append(field_name)
        return tuple(missing)
