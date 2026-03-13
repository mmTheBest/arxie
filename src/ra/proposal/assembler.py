"""Proposal draft assembly from stage-gated workflow state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ra.proposal.models import ProposalStage, ProposalWorkflowState


class ClaimEvidenceLabel(str, Enum):
    """Evidence confidence label for a proposal claim."""

    EVIDENCE_BACKED = "evidence_backed"
    SPECULATIVE = "speculative"


@dataclass(frozen=True, slots=True)
class ProposalClaim:
    """Single proposal claim with provenance metadata."""

    text: str
    label: ClaimEvidenceLabel
    reference_ids: tuple[str, ...] = ()
    provenance_links: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ProposalDraftSection:
    """One section inside the assembled proposal draft."""

    section_id: str
    title: str
    summary: str
    claims: tuple[ProposalClaim, ...]


@dataclass(frozen=True, slots=True)
class ProposalReference:
    """Canonical proposal reference record."""

    reference_id: str
    title: str
    provenance_link: str | None = None


@dataclass(frozen=True, slots=True)
class ProposalDraft:
    """Deterministic proposal draft assembled from workflow state."""

    session_id: str
    sections: tuple[ProposalDraftSection, ...]
    references: tuple[ProposalReference, ...]


class ProposalAssembler:
    """Assemble a structured proposal draft from stage payload snapshots."""

    def assemble(
        self,
        *,
        session_id: str,
        state: ProposalWorkflowState,
    ) -> ProposalDraft:
        normalized_session_id = _normalize_required_string(session_id, field_name="session_id")

        idea_payload = _stage_payload(state, ProposalStage.IDEA_INTAKE)
        logic_payload = _stage_payload(state, ProposalStage.LOGIC_REFINEMENT)
        evidence_payload = _stage_payload(state, ProposalStage.EVIDENCE_MAPPING)
        hypothesis_payload = _stage_payload(state, ProposalStage.HYPOTHESIS_RESHAPING)
        data_payload = _stage_payload(state, ProposalStage.DATA_FEASIBILITY_PLANNING)
        experiment_payload = _stage_payload(state, ProposalStage.EXPERIMENT_ANALYSIS_DESIGN)
        assembly_payload = _stage_payload(state, ProposalStage.PROPOSAL_ASSEMBLY)

        supporting_refs = _extract_references(evidence_payload.get("supporting_evidence"))
        contradicting_refs = _extract_references(evidence_payload.get("contradicting_evidence"))
        explicit_refs = _extract_references(assembly_payload.get("references"))

        all_refs = dict(supporting_refs)
        all_refs.update(contradicting_refs)
        all_refs.update(explicit_refs)

        framing_summary = _first_non_empty(
            assembly_payload.get("background_and_gap"),
            logic_payload.get("problem_gap_chain"),
            idea_payload.get("problem"),
            fallback="Background and research gap were not provided.",
        )
        hypothesis_summary = _first_non_empty(
            assembly_payload.get("hypothesis_statement"),
            hypothesis_payload.get("selected_primary_hypothesis"),
            logic_payload.get("testable_hypothesis"),
            fallback="Hypothesis statement was not provided.",
        )
        method_summary = _merge_sentences(
            _first_non_empty(
                assembly_payload.get("method_summary"),
                fallback="Method summary was not provided.",
            ),
            _first_non_empty(
                data_payload.get("selected_data_strategy"),
                fallback="Data strategy was not provided.",
            ),
            _first_non_empty(
                experiment_payload.get("experiment_design"),
                fallback="Experiment design was not provided.",
            ),
            _first_non_empty(
                experiment_payload.get("analysis_plan"),
                fallback="Analysis plan was not provided.",
            ),
        )

        outcomes_summary = _first_non_empty(
            assembly_payload.get("expected_outcomes"),
            idea_payload.get("expected_outcome"),
            fallback="Expected outcomes were not provided.",
        )

        risks_summary = _first_non_empty(
            assembly_payload.get("risks_and_limitations"),
            experiment_payload.get("controls_and_confounders"),
            fallback="Risks and limitations were not provided.",
        )

        evidence_summary = _first_non_empty(
            evidence_payload.get("landscape_summary"),
            fallback="Evidence landscape summary was not provided.",
        )

        evidence_claims = (
            _build_claim(
                _claim_sentence_from_references(
                    prefix="Supporting evidence",
                    references=supporting_refs,
                    empty_fallback="No supporting evidence references were provided.",
                ),
                references=supporting_refs,
            ),
            _build_claim(
                _claim_sentence_from_references(
                    prefix="Contradicting evidence",
                    references=contradicting_refs,
                    empty_fallback="No contradicting evidence references were provided.",
                ),
                references=contradicting_refs,
            ),
        )

        supporting_bundle = dict(supporting_refs)
        supporting_bundle.update(explicit_refs)

        sections = (
            ProposalDraftSection(
                section_id="framing",
                title="Framing",
                summary=framing_summary,
                claims=(
                    _build_claim(framing_summary, references=supporting_bundle),
                ),
            ),
            ProposalDraftSection(
                section_id="evidence",
                title="Evidence",
                summary=evidence_summary,
                claims=evidence_claims,
            ),
            ProposalDraftSection(
                section_id="hypothesis",
                title="Hypothesis",
                summary=hypothesis_summary,
                claims=(
                    _build_claim(hypothesis_summary, references=supporting_bundle),
                ),
            ),
            ProposalDraftSection(
                section_id="method",
                title="Method",
                summary=method_summary,
                claims=(
                    _build_claim(method_summary),
                ),
            ),
            ProposalDraftSection(
                section_id="outcomes",
                title="Outcomes",
                summary=outcomes_summary,
                claims=(
                    _build_claim(outcomes_summary),
                ),
            ),
            ProposalDraftSection(
                section_id="risks",
                title="Risks",
                summary=risks_summary,
                claims=(
                    _build_claim(risks_summary),
                ),
            ),
        )

        ordered_refs = tuple(all_refs[reference_id] for reference_id in sorted(all_refs))

        return ProposalDraft(
            session_id=normalized_session_id,
            sections=sections,
            references=ordered_refs,
        )


def _stage_payload(
    state: ProposalWorkflowState,
    stage: ProposalStage,
) -> Mapping[str, Any]:
    stage_state = state.stage_states.get(stage)
    if stage_state is None:
        return {}
    payload = stage_state.payload
    if not isinstance(payload, Mapping):
        return {}
    return payload


def _extract_references(value: Any) -> dict[str, ProposalReference]:
    raw_items = _as_sequence(value)
    references: dict[str, ProposalReference] = {}

    for item in raw_items:
        ref = _to_reference(item)
        if ref is None:
            continue
        existing = references.get(ref.reference_id)
        if existing is None:
            references[ref.reference_id] = ref
            continue

        # Preserve earliest title; allow later item to fill missing provenance.
        if existing.provenance_link is None and ref.provenance_link is not None:
            references[ref.reference_id] = ProposalReference(
                reference_id=existing.reference_id,
                title=existing.title,
                provenance_link=ref.provenance_link,
            )

    return references


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _to_reference(value: Any) -> ProposalReference | None:
    if isinstance(value, str):
        reference_id = value.strip()
        if not reference_id:
            return None
        return ProposalReference(reference_id=reference_id, title=reference_id)

    if not isinstance(value, Mapping):
        return None

    reference_id = _first_non_empty(
        value.get("paper_id"),
        value.get("id"),
        value.get("reference_id"),
        fallback="",
    )
    if not reference_id:
        return None

    title = _first_non_empty(
        value.get("title"),
        value.get("name"),
        fallback=reference_id,
    )
    provenance_link = _optional_string(
        value.get("provenance_link")
        or value.get("url")
        or value.get("link"),
    )
    return ProposalReference(
        reference_id=reference_id,
        title=title,
        provenance_link=provenance_link,
    )


def _claim_sentence_from_references(
    *,
    prefix: str,
    references: Mapping[str, ProposalReference],
    empty_fallback: str,
) -> str:
    if not references:
        return empty_fallback
    titles = [reference.title for reference in references.values()]
    return f"{prefix}: {_join_with_commas(titles)}."


def _build_claim(
    text: str,
    *,
    references: Mapping[str, ProposalReference] | None = None,
) -> ProposalClaim:
    normalized_text = _first_non_empty(text, fallback="Claim not provided.")
    ref_map = dict(references or {})

    if not ref_map:
        return ProposalClaim(
            text=normalized_text,
            label=ClaimEvidenceLabel.SPECULATIVE,
            reference_ids=(),
            provenance_links=None,
        )

    sorted_ids = tuple(sorted(ref_map))
    provenance_links = {
        reference_id: ref_map[reference_id].provenance_link
        for reference_id in sorted_ids
        if ref_map[reference_id].provenance_link is not None
    }

    return ProposalClaim(
        text=normalized_text,
        label=ClaimEvidenceLabel.EVIDENCE_BACKED,
        reference_ids=sorted_ids,
        provenance_links=provenance_links or None,
    )


def _first_non_empty(*values: Any, fallback: str) -> str:
    for value in values:
        text = _optional_string(value)
        if text:
            return text
    return fallback


def _merge_sentences(*sentences: str) -> str:
    merged: list[str] = []
    for sentence in sentences:
        text = _optional_string(sentence)
        if text:
            merged.append(text)
    if not merged:
        return ""
    return " ".join(merged)


def _join_with_commas(values: Sequence[str]) -> str:
    cleaned = [value.strip() for value in values if value.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized


def _normalize_required_string(value: str, *, field_name: str) -> str:
    normalized = _optional_string(value)
    if normalized is None:
        raise ValueError(f"{field_name} must not be empty")
    return normalized
