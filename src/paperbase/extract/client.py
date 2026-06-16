"""OpenAI-backed structured extraction client for Paperbase."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_openai import ChatOpenAI

from paperbase.extract.contracts import StructuredExtractionBundle
from paperbase.extract.prompts import build_extraction_messages
from paperbase.model_providers import (
    SubscriptionCLIJsonClient,
    build_subscription_cli_json_client,
    load_model_provider_config,
)
from ra.utils.logging_config import configure_logging_from_env


class OpenAIExtractionClient:
    """Run schema-constrained extraction with the configured OpenAI chat model."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        llm: Any | None = None,
    ) -> None:
        if llm is None:
            configure_logging_from_env()
            self.model_name = model or "gpt-4o-mini"
            resolved_api_key = api_key
            if resolved_api_key is None:
                config = load_model_provider_config()
                self.model_name = model or config.model_name or "gpt-4o-mini"
                resolved_api_key = config.openai_api_key
            if not resolved_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for OpenAI extraction. "
                    "Set PAPERBASE_MODEL_PROVIDER=claude_cli to use Claude Code, "
                    "or codex_cli plus PAPERBASE_ALLOW_AGENTIC_CLI=true for "
                    "trusted local corpora."
                )
            self.llm = ChatOpenAI(
                model=self.model_name,
                api_key=resolved_api_key,
                temperature=0,
            )
        else:
            self.model_name = model or getattr(
                llm,
                "model_name",
                getattr(llm, "model", "custom"),
            )
            self.llm = llm

    def extract(
        self,
        *,
        paper_text: str,
        schema_payload: dict[str, object],
    ) -> StructuredExtractionBundle:
        messages = build_extraction_messages(
            paper_text=paper_text,
            schema_payload=schema_payload,
        )
        runnable = self.llm.with_structured_output(
            StructuredExtractionBundle,
            method="function_calling",
        )
        response = runnable.invoke(messages)
        if isinstance(response, StructuredExtractionBundle):
            return response
        return StructuredExtractionBundle.model_validate(response)


class SubscriptionCLIExtractionClient:
    """Run structured extraction through a local authenticated subscription CLI."""

    def __init__(self, *, json_client: SubscriptionCLIJsonClient) -> None:
        self.json_client = json_client
        self.model_name = json_client.model_name

    def extract(
        self,
        *,
        paper_text: str,
        schema_payload: dict[str, object],
    ) -> StructuredExtractionBundle:
        messages = build_extraction_messages(
            paper_text=paper_text,
            schema_payload=schema_payload,
        )
        response = self.json_client.complete_json(
            system_prompt=str(messages[0].content),
            user_payload={
                "task": (
                    "Extract structured evidence from paper_text according to "
                    "schema_payload and required_schema."
                ),
                "paper_text": paper_text,
                "schema_payload": schema_payload,
                "required_schema": StructuredExtractionBundle.model_json_schema(),
            },
        )
        return StructuredExtractionBundle.model_validate(response)


def default_extraction_client(env: Mapping[str, str] | None = None) -> object:
    config = load_model_provider_config(env)
    if config.provider == "none":
        raise ValueError("PAPERBASE_MODEL_PROVIDER=none disables model-backed extraction.")
    if config.provider == "openai":
        return OpenAIExtractionClient(
            model=config.model_name,
            api_key=config.openai_api_key,
        )
    if config.provider == "codex_cli" and not config.allow_agentic_cli:
        raise ValueError(
            "PAPERBASE_MODEL_PROVIDER=codex_cli requires "
            "PAPERBASE_ALLOW_AGENTIC_CLI=true on trusted local corpora."
        )
    return SubscriptionCLIExtractionClient(
        json_client=build_subscription_cli_json_client(config),
    )
