"""Model-backed semantic entailment grading for Paperbase validation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

MODEL_ENTAILMENT_GRADER_ENV = "PAPERBASE_MODEL_ENTAILMENT_GRADER"
ENTAILMENT_VERDICTS = ("entailed", "not_entailed", "unknown")
PUBLIC_NOT_ENTAILED_REASON = (
    "Model entailment grader judged cited evidence insufficient."
)
_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_MAX_ENTAILMENT_TEXT_CHARS = 4000


class ModelEntailmentGrader:
    """Strict JSON adapter from model responses to harness entailment verdicts."""

    def __init__(
        self,
        model_client: object,
        *,
        max_text_chars: int = _MAX_ENTAILMENT_TEXT_CHARS,
    ) -> None:
        self.model_client = model_client
        self.max_text_chars = max(256, max_text_chars)

    def __call__(
        self,
        *,
        recommendation_label: str,
        support_label: str,
        recommendation_text: str,
        support_text: str,
        evidence_family: str,
    ) -> dict[str, str]:
        complete_json = getattr(self.model_client, "complete_json", None)
        if not callable(complete_json):
            return {"verdict": "unknown"}

        try:
            payload = complete_json(
                system_prompt=_model_entailment_system_prompt(),
                user_payload={
                    "task": "semantic_entailment_verdict",
                    "verdict_options": list(ENTAILMENT_VERDICTS),
                    "evidence_family": _bounded_text(evidence_family, limit=128),
                    "recommendation_label": _bounded_text(
                        recommendation_label,
                        limit=512,
                    ),
                    "support_label": _bounded_text(support_label, limit=512),
                    "recommendation_text": _bounded_text(
                        recommendation_text,
                        limit=self.max_text_chars,
                    ),
                    "support_text": _bounded_text(
                        support_text,
                        limit=self.max_text_chars,
                    ),
                    "instructions": (
                        "Return exactly one JSON object with verdict set to "
                        "entailed, not_entailed, or unknown. Use unknown when "
                        "the relationship is ambiguous, underspecified, or "
                        "requires scientific judgment beyond the cited text."
                    ),
                },
            )
        except Exception:
            return {"verdict": "unknown"}

        verdict = _model_verdict(payload)
        if verdict == "not_entailed":
            return {
                "verdict": verdict,
                "reason": PUBLIC_NOT_ENTAILED_REASON,
            }
        return {"verdict": verdict}


class UnavailableModelEntailmentGrader:
    """Fail-closed grader used when opt-in model grading cannot be constructed."""

    def __call__(self, **_ignored: Any) -> dict[str, str]:
        return {"verdict": "unknown"}


def model_entailment_grader_from_env(
    model_client: object | None,
    *,
    env: Mapping[str, str] | None = None,
) -> ModelEntailmentGrader | UnavailableModelEntailmentGrader | None:
    """Build the optional model entailment grader from runtime configuration."""

    if not model_entailment_grader_enabled(env):
        return None
    if model_client is None:
        return UnavailableModelEntailmentGrader()
    if not callable(getattr(model_client, "complete_json", None)):
        return UnavailableModelEntailmentGrader()
    return ModelEntailmentGrader(model_client)


def model_entailment_grader_enabled(
    env: Mapping[str, str] | None = None,
) -> bool:
    resolved_env = os.environ if env is None else env
    return (
        str(resolved_env.get(MODEL_ENTAILMENT_GRADER_ENV) or "")
        .strip()
        .lower()
        in _TRUTHY_VALUES
    )


def _model_entailment_system_prompt() -> str:
    return (
        "You are a semantic entailment verifier for Arxie validation. "
        "Judge only whether the cited support text entails the recommendation. "
        "Return JSON only and do not include hidden reasoning, quotes, markdown, "
        "or extra fields."
    )


def _model_verdict(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    verdict = payload.get("verdict")
    if not isinstance(verdict, str):
        return "unknown"
    normalized = verdict.strip().lower()
    if normalized not in ENTAILMENT_VERDICTS:
        return "unknown"
    return normalized


def _bounded_text(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
