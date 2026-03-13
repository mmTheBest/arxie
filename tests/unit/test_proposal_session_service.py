from __future__ import annotations

from ra.proposal import (
    InMemoryProposalSessionStore,
    ProposalSessionService,
    ProposalStage,
    ProposalStageEngine,
    SessionVersionConflictError,
)


def test_create_session_persists_initial_snapshot_with_version_zero() -> None:
    store = InMemoryProposalSessionStore()
    service = ProposalSessionService(store=store, stage_engine=ProposalStageEngine())

    snapshot = service.create_session("session-1")

    assert snapshot.session_id == "session-1"
    assert snapshot.version == 0
    assert snapshot.state.current_stage is ProposalStage.IDEA_INTAKE
    assert store.get_snapshot("session-1") == snapshot


def test_update_and_advance_autosave_state_and_increment_version() -> None:
    engine = ProposalStageEngine()
    store = InMemoryProposalSessionStore()
    service = ProposalSessionService(store=store, stage_engine=engine)
    _ = service.create_session("session-1")

    partial = service.update_stage_payload(
        "session-1",
        ProposalStage.IDEA_INTAKE,
        {
            "problem": "Unsupported claims in generated summaries.",
            "target_population": "Clinical NLP researchers.",
        },
        expected_version=0,
    )

    assert partial.version == 1
    assert store.get_snapshot("session-1") == partial
    assert partial.state.stage_states[ProposalStage.IDEA_INTAKE].confirmed is False

    complete = service.update_stage_payload(
        "session-1",
        ProposalStage.IDEA_INTAKE,
        {
            "mechanism": "Retrieval-grounded synthesis with citation checks.",
            "expected_outcome": "Higher precision in citation-backed claims.",
        },
        expected_version=1,
    )
    assert complete.version == 2
    assert complete.state.stage_states[ProposalStage.IDEA_INTAKE].confirmed is True

    advanced = service.advance_stage("session-1", expected_version=2)
    assert advanced.version == 3
    assert advanced.state.current_stage is ProposalStage.LOGIC_REFINEMENT
    assert store.get_snapshot("session-1") == advanced


def test_update_rejects_stale_version_with_predictable_conflict_error() -> None:
    engine = ProposalStageEngine()
    service = ProposalSessionService(
        store=InMemoryProposalSessionStore(),
        stage_engine=engine,
    )
    _ = service.create_session("session-1")
    _ = service.update_stage_payload(
        "session-1",
        ProposalStage.IDEA_INTAKE,
        {"problem": "P", "target_population": "T"},
        expected_version=0,
    )

    try:
        _ = service.update_stage_payload(
            "session-1",
            ProposalStage.IDEA_INTAKE,
            {"mechanism": "M"},
            expected_version=0,
        )
    except SessionVersionConflictError as exc:
        assert exc.session_id == "session-1"
        assert exc.expected_version == 0
        assert exc.current_version == 1
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected SessionVersionConflictError for stale version")


def test_recover_latest_snapshot_after_store_rehydration() -> None:
    engine = ProposalStageEngine()
    store = InMemoryProposalSessionStore()
    service = ProposalSessionService(store=store, stage_engine=engine)
    _ = service.create_session("session-1")
    _ = service.update_stage_payload(
        "session-1",
        ProposalStage.IDEA_INTAKE,
        engine.required_fields_payload(ProposalStage.IDEA_INTAKE),
        expected_version=0,
    )
    advanced = service.advance_stage("session-1", expected_version=1)

    persisted = store.dump_snapshots()
    recovered_store = InMemoryProposalSessionStore.from_snapshots(persisted)
    recovered_service = ProposalSessionService(store=recovered_store, stage_engine=ProposalStageEngine())

    recovered = recovered_service.get_session("session-1")
    assert recovered == advanced
    assert recovered.version == 2
    assert recovered.state.current_stage is ProposalStage.LOGIC_REFINEMENT

    updated = recovered_service.update_stage_payload(
        "session-1",
        ProposalStage.LOGIC_REFINEMENT,
        {"problem_gap_chain": "Gap -> theory mismatch."},
        expected_version=recovered.version,
    )
    assert updated.version == 3
