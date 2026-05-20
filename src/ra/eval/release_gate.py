"""Release-gate evaluation for v0.2 proposal workflow milestones."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ra.proposal import (
    ClaimEvidenceLabel,
    ProposalAssembler,
    ProposalDraft,
    ProposalStage,
    ProposalStageEngine,
)

_GATE_STAGE_SEQUENCE: tuple[ProposalStage, ...] = ProposalStageEngine().stage_sequence


@dataclass(frozen=True, slots=True)
class ReleaseGateCase:
    """Single release-gate case with thresholds."""

    case_id: str
    session_id: str
    stage_payloads: dict[ProposalStage, dict[str, Any]]
    min_stage_completion_ratio: float
    min_evidence_link_coverage: float


def load_release_gate_cases(source: str | Path | list[dict[str, Any]]) -> list[ReleaseGateCase]:
    """Load and validate release-gate cases from a file path or inline rows."""

    payload: Any
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Release-gate dataset not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = source

    if not isinstance(payload, list):
        raise ValueError("Release-gate dataset must be a JSON list.")

    if len(payload) == 0:
        raise ValueError("Release-gate dataset must contain at least one case.")

    cases: list[ReleaseGateCase] = []
    for idx, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Case #{idx} must be a JSON object.")

        case_id = _required_text(row.get("id"), field_name=f"Case #{idx} id")
        session_id = _required_text(
            row.get("session_id"),
            field_name=f"Case #{idx} session_id",
        )
        stage_payloads = _parse_stage_payloads(
            row.get("stage_payloads"),
            row_index=idx,
        )
        min_stage_completion_ratio = _validate_ratio(
            row.get("min_stage_completion_ratio"),
            field_name=f"Case #{idx} min_stage_completion_ratio",
        )
        min_evidence_link_coverage = _validate_ratio(
            row.get("min_evidence_link_coverage"),
            field_name=f"Case #{idx} min_evidence_link_coverage",
        )

        cases.append(
            ReleaseGateCase(
                case_id=case_id,
                session_id=session_id,
                stage_payloads=stage_payloads,
                min_stage_completion_ratio=min_stage_completion_ratio,
                min_evidence_link_coverage=min_evidence_link_coverage,
            )
        )

    return cases


def evaluate_release_gate_cases(cases: list[ReleaseGateCase]) -> dict[str, Any]:
    """Evaluate cases against stage completion and evidence-link thresholds."""

    if len(cases) == 0:
        raise ValueError("Release-gate evaluation requires at least one case.")

    engine = ProposalStageEngine()
    assembler = ProposalAssembler()
    results: list[dict[str, Any]] = []

    stage_passes = 0
    evidence_link_passes = 0
    overall_passes = 0

    for case in cases:
        state = engine.create_initial_state()
        for stage, payload in case.stage_payloads.items():
            state = engine.update_stage_payload(state, stage, payload)

        completed_stage_count = sum(
            1
            for stage in _GATE_STAGE_SEQUENCE
            if engine.diagnose_stage(state, stage).is_complete
        )
        stage_completion_ratio = completed_stage_count / len(_GATE_STAGE_SEQUENCE)

        draft = assembler.assemble(session_id=case.session_id, state=state)
        evidence_link_coverage = _evidence_link_coverage(draft)

        stage_completion_pass = stage_completion_ratio >= case.min_stage_completion_ratio
        evidence_link_pass = evidence_link_coverage >= case.min_evidence_link_coverage
        gate_pass = stage_completion_pass and evidence_link_pass

        stage_passes += int(stage_completion_pass)
        evidence_link_passes += int(evidence_link_pass)
        overall_passes += int(gate_pass)

        results.append(
            {
                "id": case.case_id,
                "session_id": case.session_id,
                "completed_stage_count": completed_stage_count,
                "expected_stage_count": len(_GATE_STAGE_SEQUENCE),
                "stage_completion_ratio": round(stage_completion_ratio, 6),
                "min_stage_completion_ratio": case.min_stage_completion_ratio,
                "evidence_link_coverage": round(evidence_link_coverage, 6),
                "min_evidence_link_coverage": case.min_evidence_link_coverage,
                "stage_completion_pass": stage_completion_pass,
                "evidence_link_pass": evidence_link_pass,
                "pass": gate_pass,
            }
        )

    total_cases = len(cases)
    metrics = {
        "stage_completion_pass_rate": _safe_rate(stage_passes, total_cases),
        "evidence_link_pass_rate": _safe_rate(evidence_link_passes, total_cases),
        "gate_pass_rate": _safe_rate(overall_passes, total_cases),
    }
    return {
        "total_cases": total_cases,
        "metrics": metrics,
        "results": results,
        "overall_pass": overall_passes == total_cases,
    }


def _parse_stage_payloads(
    raw_stage_payloads: Any,
    *,
    row_index: int,
) -> dict[ProposalStage, dict[str, Any]]:
    if not isinstance(raw_stage_payloads, dict):
        raise ValueError(f"Case #{row_index} stage_payloads must be an object.")

    parsed: dict[ProposalStage, dict[str, Any]] = {}
    for stage_name, payload in raw_stage_payloads.items():
        if not isinstance(stage_name, str):
            raise ValueError(f"Case #{row_index} stage name must be a string.")
        try:
            stage = ProposalStage(stage_name.strip())
        except ValueError as exc:
            raise ValueError(f"Case #{row_index} has unknown stage: {stage_name!r}") from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Case #{row_index} payload for stage {stage_name!r} must be an object."
            )
        parsed[stage] = dict(payload)
    return parsed


def _evidence_link_coverage(draft: ProposalDraft) -> float:
    linked_refs = 0
    total_refs = 0

    for section in draft.sections:
        for claim in section.claims:
            if claim.label is not ClaimEvidenceLabel.EVIDENCE_BACKED:
                continue
            links = claim.provenance_links or {}
            for reference_id in claim.reference_ids:
                total_refs += 1
                if _required_text(links.get(reference_id), field_name="link", allow_empty=True):
                    linked_refs += 1

    return _safe_rate(linked_refs, total_refs)


def _required_text(value: Any, *, field_name: str, allow_empty: bool = False) -> str:
    text = str(value).strip() if value is not None else ""
    if not text and not allow_empty:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _validate_ratio(value: Any, *, field_name: str) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number in [0, 1].") from exc
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1].")
    return ratio


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)
