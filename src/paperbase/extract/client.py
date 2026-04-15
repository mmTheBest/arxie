"""OpenAI-backed structured extraction client for Paperbase."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from paperbase.extract.contracts import StructuredExtractionBundle
from paperbase.extract.prompts import build_extraction_messages
from ra.utils.config import load_config
from ra.utils.logging_config import configure_logging_from_env


class OpenAIExtractionClient:
    """Run schema-constrained extraction with the configured OpenAI chat model."""

    def __init__(
        self,
        *,
        model: str | None = None,
        llm: Any | None = None,
    ) -> None:
        if llm is None:
            configure_logging_from_env()
            config = load_config()
            self.model_name = model or config.ra_model
            self.llm = ChatOpenAI(
                model=self.model_name,
                api_key=config.openai_api_key,
                temperature=0,
            )
        else:
            self.model_name = model or getattr(llm, "model_name", getattr(llm, "model", "custom"))
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
