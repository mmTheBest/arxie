"""Model-backed synthesis client for Paperbase research-agent runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from paperbase.model_providers import (
    SubscriptionCLIJsonClient,
    build_subscription_cli_json_client,
    load_model_provider_config,
)


class ResearchModelClientError(RuntimeError):
    """Raised when model-backed research synthesis fails."""


class OpenAIResearchModelClient:
    """Small JSON-oriented wrapper around the configured OpenAI chat model."""

    def __init__(self, *, model: str, api_key: str, llm: Any | None = None) -> None:
        self.model_name = model
        self.llm = llm or ChatOpenAI(model=model, api_key=api_key, temperature=0)

    def synthesize(
        self,
        *,
        skill_id: str,
        artifact_type: str,
        prompt_payload: dict[str, Any],
    ) -> dict[str, Any]:
        messages = self._messages(
            skill_id=skill_id,
            artifact_type=artifact_type,
            prompt_payload=prompt_payload,
        )
        response = self.llm.invoke(messages)
        try:
            return self._parse_response(response)
        except ResearchModelClientError:
            repair_response = self.llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Return only valid JSON for the requested "
                            "Arxie artifact schema."
                        )
                    ),
                    HumanMessage(content=str(getattr(response, "content", response))),
                ]
            )
            return self._parse_response(repair_response)

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(user_payload, ensure_ascii=True)),
            ]
        )
        return self._parse_response(response)

    def _messages(
        self,
        *,
        skill_id: str,
        artifact_type: str,
        prompt_payload: dict[str, Any],
    ) -> list[Any]:
        return [
            SystemMessage(content=_research_system_prompt()),
            HumanMessage(
                content=json.dumps(
                    _research_user_payload(
                        skill_id=skill_id,
                        artifact_type=artifact_type,
                        prompt_payload=prompt_payload,
                    ),
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


class SubscriptionCLIResearchModelClient:
    """JSON-oriented research synthesis client backed by a local subscription CLI."""

    def __init__(self, *, json_client: SubscriptionCLIJsonClient) -> None:
        self.json_client = json_client
        self.model_name = json_client.model_name

    def synthesize(
        self,
        *,
        skill_id: str,
        artifact_type: str,
        prompt_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self.json_client.complete_json(
            system_prompt=_research_system_prompt(),
            user_payload=_research_user_payload(
                skill_id=skill_id,
                artifact_type=artifact_type,
                prompt_payload=prompt_payload,
            ),
        )

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self.json_client.complete_json(
            system_prompt=system_prompt,
            user_payload=user_payload,
        )


def _research_system_prompt() -> str:
    return (
        "You are Arxie, a research agent. Produce concise, valid JSON only. "
        "Ground recommendations in the supplied paper and user-source context. "
        "Separate paper evidence from user-provided context and mark inferences."
    )


def _research_user_payload(
    *,
    skill_id: str,
    artifact_type: str,
    prompt_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
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
    }


def default_research_model_client(
    env: Mapping[str, str] | None = None,
) -> OpenAIResearchModelClient | SubscriptionCLIResearchModelClient | None:
    config = load_model_provider_config(env)
    if config.provider == "none":
        return None
    if config.provider == "openai":
        if not config.openai_api_key:
            return None
        return OpenAIResearchModelClient(
            model=config.model_name or "gpt-4o-mini",
            api_key=config.openai_api_key,
        )
    if config.provider in {"codex_cli", "claude_cli"}:
        if config.provider == "codex_cli" and not config.allow_agentic_cli:
            return None
        return SubscriptionCLIResearchModelClient(
            json_client=build_subscription_cli_json_client(config),
        )
    return None
