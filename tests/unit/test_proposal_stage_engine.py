from __future__ import annotations

from ra.proposal import (
    ProposalStage,
    ProposalStageEngine,
    StageTransitionError,
    StageTransitionReason,
)


def test_stage_sequence_matches_v02_workflow_order() -> None:
    engine = ProposalStageEngine()

    assert engine.stage_sequence == (
        ProposalStage.IDEA_INTAKE,
        ProposalStage.LOGIC_REFINEMENT,
        ProposalStage.EVIDENCE_MAPPING,
        ProposalStage.HYPOTHESIS_RESHAPING,
        ProposalStage.DATA_FEASIBILITY_PLANNING,
        ProposalStage.EXPERIMENT_ANALYSIS_DESIGN,
        ProposalStage.PROPOSAL_ASSEMBLY,
    )


def test_create_initial_state_sets_first_stage_and_empty_payloads() -> None:
    engine = ProposalStageEngine()

    state = engine.create_initial_state()

    assert state.current_stage is ProposalStage.IDEA_INTAKE
    assert tuple(state.stage_states.keys()) == engine.stage_sequence
    assert all(not stage_state.payload for stage_state in state.stage_states.values())


def test_diagnose_stage_reports_required_and_missing_fields() -> None:
    engine = ProposalStageEngine()
    state = engine.create_initial_state()
    state = engine.update_stage_payload(
        state,
        ProposalStage.IDEA_INTAKE,
        {
            "problem": "LLM citations are often incorrect.",
            "target_population": "Graduate researchers.",
        },
    )

    diagnostics = engine.diagnose_stage(state, ProposalStage.IDEA_INTAKE)

    assert diagnostics.stage is ProposalStage.IDEA_INTAKE
    assert "mechanism" in diagnostics.missing_fields
    assert "expected_outcome" in diagnostics.missing_fields
    assert diagnostics.is_complete is False


def test_advance_stage_requires_current_stage_completion() -> None:
    engine = ProposalStageEngine()
    state = engine.create_initial_state()

    try:
        engine.advance_stage(state)
    except StageTransitionError as exc:
        assert exc.reason is StageTransitionReason.INCOMPLETE_STAGE
        assert exc.from_stage is ProposalStage.IDEA_INTAKE
        assert exc.to_stage is ProposalStage.LOGIC_REFINEMENT
        assert "problem" in exc.missing_fields
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected StageTransitionError for incomplete current stage")


def test_transition_to_rejects_non_adjacent_stage_jump() -> None:
    engine = ProposalStageEngine()
    state = engine.create_initial_state()

    try:
        engine.transition_to(state, ProposalStage.EVIDENCE_MAPPING)
    except StageTransitionError as exc:
        assert exc.reason is StageTransitionReason.INVALID_TRANSITION
        assert exc.from_stage is ProposalStage.IDEA_INTAKE
        assert exc.to_stage is ProposalStage.EVIDENCE_MAPPING
        assert exc.allowed_next_stage is ProposalStage.LOGIC_REFINEMENT
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected StageTransitionError for stage jump")


def test_transition_to_next_stage_after_completion_succeeds() -> None:
    engine = ProposalStageEngine()
    state = engine.create_initial_state()
    state = engine.update_stage_payload(
        state,
        ProposalStage.IDEA_INTAKE,
        {
            "problem": "LLM citations are often incorrect.",
            "target_population": "Graduate researchers.",
            "mechanism": "Grounding via retrieval and citation checks.",
            "expected_outcome": "Higher citation precision.",
        },
    )

    next_state = engine.transition_to(state, ProposalStage.LOGIC_REFINEMENT)

    assert next_state.current_stage is ProposalStage.LOGIC_REFINEMENT
    assert engine.is_stage_complete(next_state, ProposalStage.IDEA_INTAKE) is True


def test_advance_from_final_stage_raises_final_stage_error() -> None:
    engine = ProposalStageEngine()
    state = engine.create_initial_state()

    for stage in engine.stage_sequence[:-1]:
        state = engine.update_stage_payload(state, stage, engine.required_fields_payload(stage))
        state = engine.advance_stage(state)

    state = engine.update_stage_payload(
        state,
        ProposalStage.PROPOSAL_ASSEMBLY,
        engine.required_fields_payload(ProposalStage.PROPOSAL_ASSEMBLY),
    )

    try:
        engine.advance_stage(state)
    except StageTransitionError as exc:
        assert exc.reason is StageTransitionReason.FINAL_STAGE_REACHED
        assert exc.from_stage is ProposalStage.PROPOSAL_ASSEMBLY
        assert exc.to_stage is ProposalStage.PROPOSAL_ASSEMBLY
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected StageTransitionError for final stage")


def test_transition_outputs_are_deterministic_for_identical_inputs() -> None:
    engine = ProposalStageEngine()
    base_state = engine.create_initial_state()
    base_state = engine.update_stage_payload(
        base_state,
        ProposalStage.IDEA_INTAKE,
        {
            "problem": "Citations in generated reports are fragile.",
            "target_population": "Academic lab teams.",
            "mechanism": "Add deterministic citation verification.",
            "expected_outcome": "Lower unsupported-claim rate.",
        },
    )

    out_a = engine.transition_to(base_state, ProposalStage.LOGIC_REFINEMENT)
    out_b = engine.transition_to(base_state, ProposalStage.LOGIC_REFINEMENT)

    assert out_a == out_b


def test_stage_four_and_five_required_fields_match_planning_artifact_contract() -> None:
    engine = ProposalStageEngine()

    assert engine.required_fields(ProposalStage.DATA_FEASIBILITY_PLANNING) == (
        "data_options_table",
        "feasibility_scorecard",
        "selected_data_strategy",
    )
    assert engine.required_fields(ProposalStage.EXPERIMENT_ANALYSIS_DESIGN) == (
        "experiment_flow_diagram",
        "analysis_plan_tree",
        "outcome_comparison_matrix",
    )
