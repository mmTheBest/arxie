"""Model-backed synthesis client for Paperbase research-agent runs."""

from __future__ import annotations

import json
from typing import Any, Mapping

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ra.utils.config import load_config


class ResearchModelClientError(RuntimeError):
    """Raised when model-backed research synthesis fails."""


class OpenAIResearchModelClient:
    """Small JSON-oriented wrapper around the configured OpenAI chat model."""

    def __init__(self, *, model: str, api_key: str, llm: Any | None = None) -> None:
        self.model_name = model
        self.llm = llm or ChatOpenAI(model=model, api_key=api_key, temperature=0)

    def synthesize(self, *, skill_id: str, artifact_type: str, prompt_payload: dict[str, Any]) -> dict[str, Any]:
        messages = self._messages(skill_id=skill_id, artifact_type=artifact_type, prompt_payload=prompt_payload)
        response = self.llm.invoke(messages)
        try:
            return self._parse_response(response)
        except ResearchModelClientError:
            repair_response = self.llm.invoke(
                [
                    SystemMessage(content="Return only valid JSON for the requested Arxie artifact schema."),
                    HumanMessage(content=str(getattr(response, "content", response))),
                ]
            )
            return self._parse_response(repair_response)

    def _messages(self, *, skill_id: str, artifact_type: str, prompt_payload: dict[str, Any]) -> list[Any]:
        return [
            SystemMessage(
                content=(
                    "You are Arxie, a research agent. Produce concise, valid JSON only. "
                    "Ground recommendations in the supplied paper and user-source context. "
                    "Separate paper evidence from user-provided context and mark inferences."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "skill_id": skill_id,
                        "artifact_type": artifact_type,
                        "required_fields": [
                            "title",
                            "summary",
                            "recommendations",
                            "evidence_references",
                            "assumptions_or_inferences",
                            "next_actions",
                            "limitations",
                        ],
                        **prompt_payload,
                    },
                    ensure_ascii=True,
                )
            ),
        ]

    def _parse_response(self, response: Any) -> dict[str, Any]:
        content = str(getattr(response, "content", response)).strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ResearchModelClientError("Model did not return valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ResearchModelClientError("Model JSON payload must be an object.")
        return payload


def default_research_model_client(env: Mapping[str, str] | None = None) -> OpenAIResearchModelClient | None:
    try:
        config = load_config(env)
    except ValueError:
        return None
    return OpenAIResearchModelClient(model=config.ra_model, api_key=config.openai_api_key)
