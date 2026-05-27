"""Execution policies for internal Paperbase research skills."""

from __future__ import annotations

from dataclasses import dataclass

from paperbase.research.skills import UnsupportedResearchSkillError


@dataclass(frozen=True, slots=True)
class ResearchSkillPolicy:
    skill_id: str
    artifact_type: str
    model_policy: str
    allow_deterministic_fallback: bool


_POLICIES: dict[str, ResearchSkillPolicy] = {
    "quality_harness": ResearchSkillPolicy(
        skill_id="quality_harness",
        artifact_type="critique",
        model_policy="deterministic",
        allow_deterministic_fallback=True,
    ),
    "comparison": ResearchSkillPolicy(
        skill_id="comparison",
        artifact_type="comparison",
        model_policy="optional",
        allow_deterministic_fallback=True,
    ),
    "literature_review": ResearchSkillPolicy(
        skill_id="literature_review",
        artifact_type="literature_review",
        model_policy="preferred",
        allow_deterministic_fallback=True,
    ),
    "benchmark_planning": ResearchSkillPolicy(
        skill_id="benchmark_planning",
        artifact_type="benchmark_plan",
        model_policy="preferred",
        allow_deterministic_fallback=True,
    ),
    "experiment_planning": ResearchSkillPolicy(
        skill_id="experiment_planning",
        artifact_type="experiment_plan",
        model_policy="required",
        allow_deterministic_fallback=False,
    ),
    "field_pattern_analysis": ResearchSkillPolicy(
        skill_id="field_pattern_analysis",
        artifact_type="field_patterns",
        model_policy="deterministic",
        allow_deterministic_fallback=True,
    ),
    "experiment_backlog": ResearchSkillPolicy(
        skill_id="experiment_backlog",
        artifact_type="experiment_backlog",
        model_policy="deterministic",
        allow_deterministic_fallback=True,
    ),
    "assumption_mapping": ResearchSkillPolicy(
        skill_id="assumption_mapping",
        artifact_type="assumption_map",
        model_policy="deterministic",
        allow_deterministic_fallback=True,
    ),
    "hypothesis_generation": ResearchSkillPolicy(
        skill_id="hypothesis_generation",
        artifact_type="hypotheses",
        model_policy="required",
        allow_deterministic_fallback=False,
    ),
    "revision_planning": ResearchSkillPolicy(
        skill_id="revision_planning",
        artifact_type="revision_plan",
        model_policy="required",
        allow_deterministic_fallback=False,
    ),
}


def policy_for_skill(skill_id: str) -> ResearchSkillPolicy:
    if skill_id not in _POLICIES:
        raise UnsupportedResearchSkillError(f"Unsupported research skill: {skill_id}")
    return _POLICIES[skill_id]
