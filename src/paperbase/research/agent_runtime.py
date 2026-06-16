"""Traceable Paperbase research-agent runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from paperbase.db.models import ResearchAgentRun
from paperbase.db.repositories import ResearchAgentRunRepository, ResearchRepository
from paperbase.research.context_builder import (
    PaperbaseResearchContextBuilder,
    ResearchContextPack,
)
from paperbase.research.entailment import model_entailment_grader_from_env
from paperbase.research.harness import validate_research_output
from paperbase.research.output_contracts import (
    ResearchModelOutputContractError,
    normalize_model_output_payload,
    output_contract_prompt,
)
from paperbase.research.skill_policies import ResearchSkillPolicy, policy_for_skill
from paperbase.research.skills import (
    ResearchSkillContext,
    UnsupportedResearchSkillError,
    default_research_skill_registry,
    select_research_skill,
    with_study_brief_update,
)
from paperbase.research.tool_adapters import default_study_agent_read_tool_executor
from paperbase.research.tool_registry import default_study_agent_tool_registry

TERMINAL_RUN_STATUSES = {"completed", "blocked"}
MODEL_OUTPUT_SCHEMA_ATTEMPTS = 2


def _context_cache_trace(
    *,
    cache_hit: bool,
    cache_key: str | None,
    cached_context_pack: Any,
) -> dict[str, Any]:
    return {
        "status": "hit" if cache_hit else "miss",
        "cache_key_present": bool(cache_key),
        "source_context_pack_id": (
            cached_context_pack.id if cached_context_pack is not None else None
        ),
        "source_run_id": (
            cached_context_pack.run_id if cached_context_pack is not None else None
        ),
        "source_attempt_number": (
            cached_context_pack.attempt_number
            if cached_context_pack is not None
            else None
        ),
    }


class PaperbaseResearchAgentRuntime:
    """Execute one research-agent run with context, synthesis, and validation traces."""

    def __init__(
        self,
        *,
        model_client: object | None = None,
        search_backend: object | None = None,
        embedding_provider: object | None = None,
        project_id: str | None = None,
        entailment_grader: Callable[..., Any] | None = None,
    ) -> None:
        self.model_client = model_client
        self.search_backend = search_backend
        self.embedding_provider = embedding_provider
        self.project_id = project_id
        self.entailment_grader = entailment_grader

    def execute(self, session: Session, payload: dict[str, Any]) -> dict[str, Any]:
        research_repository = ResearchRepository(session)
        run_repository = ResearchAgentRunRepository(session)
        artifact_id = str(payload["artifact_id"])
        artifact = research_repository.get_artifact(artifact_id)
        if artifact is None:
            raise ValueError(f"No research artifact found for id: {artifact_id}")

        skill_id = str(
            payload.get("skill_id")
            or select_research_skill(
                str(payload.get("user_message") or ""),
                suggestion_id=(
                    str(payload["suggestion_id"])
                    if payload.get("suggestion_id")
                    else None
                ),
                artifact_type=(
                    str(payload["artifact_type"])
                    if payload.get("artifact_type")
                    else None
                ),
            )
        )
        try:
            policy = policy_for_skill(skill_id)
        except UnsupportedResearchSkillError as exc:
            self._mark_payload_failed(
                run_repository,
                research_repository,
                payload=payload,
                error_message=str(exc),
            )
            raise
        run = self._ensure_run(
            run_repository,
            payload=payload,
            policy=policy,
            artifact_type=str(payload.get("artifact_type") or policy.artifact_type),
        )
        if run.status in TERMINAL_RUN_STATUSES:
            return self._terminal_run_result(
                run_repository,
                artifact=artifact,
                run=run,
                fallback_artifact_type=policy.artifact_type,
            )
        attempt_number = run_repository.next_attempt_number(run_id=run.id)
        run_repository.mark_running(run.id)

        workspace_id = str(payload["workspace_id"]) if payload.get("workspace_id") else None
        selected_paper_ids = [
            str(item) for item in list(payload.get("selected_paper_ids") or [])
        ]
        source_ids = [str(item) for item in list(payload.get("source_ids") or [])]
        context_builder = PaperbaseResearchContextBuilder(
            session,
            search_backend=self.search_backend,
            embedding_provider=self.embedding_provider,
            project_id=self.project_id,
        )
        context_cache_key = context_builder.cache_lookup_key(
            collection_id=str(payload["collection_id"]),
            task_type=skill_id,
            message=str(payload.get("user_message") or ""),
            prompt_version=str(payload.get("prompt_version") or ""),
            selected_paper_ids=selected_paper_ids,
            workspace_id=workspace_id,
            source_ids=source_ids,
        )
        cached_context_pack = run_repository.get_latest_context_pack_by_cache_key(
            collection_id=str(payload["collection_id"]),
            workspace_id=workspace_id,
            task_type=skill_id,
            cache_key=context_cache_key,
        )
        cache_hit = cached_context_pack is not None
        if cached_context_pack is not None:
            context_pack = ResearchContextPack(
                context=dict(cached_context_pack.context_json or {}),
                selected_item_counts=dict(
                    cached_context_pack.selected_item_counts_json or {}
                ),
                readiness_warnings=list(
                    cached_context_pack.readiness_warnings_json or []
                ),
                cache_key=context_cache_key,
            )
        else:
            context_pack = context_builder.build(
                collection_id=str(payload["collection_id"]),
                task_type=skill_id,
                message=str(payload.get("user_message") or ""),
                selected_paper_ids=selected_paper_ids,
                workspace_id=workspace_id,
                source_ids=source_ids,
                cache_key_override=context_cache_key,
            )
        run_repository.create_context_pack(
            run_id=run.id,
            attempt_number=attempt_number,
            collection_id=str(payload["collection_id"]),
            workspace_id=workspace_id,
            task_type=skill_id,
            context_json=context_pack.context,
            selected_item_counts=context_pack.selected_item_counts,
            readiness_warnings=context_pack.readiness_warnings,
            cache_key=context_pack.cache_key,
        )
        intelligence_layers_summary = self._intelligence_layers_summary(
            context=context_pack.context,
            selected_item_counts=context_pack.selected_item_counts,
        )
        selected_context_summary = self._selected_context_summary(
            context=context_pack.context,
            selected_item_counts=context_pack.selected_item_counts,
            intelligence_layers_summary=intelligence_layers_summary,
        )
        read_tool_summary = default_study_agent_tool_registry().trace_summary(
            task_type=skill_id,
            selected_item_counts=context_pack.selected_item_counts,
        )
        read_tool_observations = default_study_agent_read_tool_executor().execute(
            context=context_pack.context,
            read_tools=read_tool_summary,
            task_type=skill_id,
        )
        context_cache_trace = _context_cache_trace(
            cache_hit=cache_hit,
            cache_key=context_pack.cache_key,
            cached_context_pack=cached_context_pack,
        )
        run_repository.append_step(
            run_id=run.id,
            attempt_number=attempt_number,
            step_type="context",
            label="Build study context",
            output_json={
                "attempt_number": attempt_number,
                "selected_item_counts": context_pack.selected_item_counts,
                "readiness_warnings": context_pack.readiness_warnings,
                "intelligence_layers": intelligence_layers_summary,
                "cache_hit": cache_hit,
                "cache_key": context_pack.cache_key,
                "cached_context_pack_id": (
                    cached_context_pack.id if cached_context_pack is not None else None
                ),
                "context_cache": context_cache_trace,
            },
        )
        run_repository.append_step(
            run_id=run.id,
            attempt_number=attempt_number,
            step_type="plan",
            label="Plan research synthesis",
            input_json={
                "attempt_number": attempt_number,
                "skill_id": skill_id,
                "artifact_type": policy.artifact_type,
                "model_policy": policy.model_policy,
            },
            output_json={
                "attempt_number": attempt_number,
                "planned_action": "synthesize_research_artifact",
                "selected_context": selected_context_summary,
                "missing_prerequisites": list(context_pack.readiness_warnings),
                "read_tools": read_tool_summary,
            },
        )
        run_repository.append_step(
            run_id=run.id,
            attempt_number=attempt_number,
            step_type="tool_call",
            label="Prepare synthesis inputs",
            input_json={
                "attempt_number": attempt_number,
                "tool_name": "research_synthesis",
                "skill_id": skill_id,
                "artifact_type": policy.artifact_type,
                "model_policy": policy.model_policy,
            },
            output_json={
                "attempt_number": attempt_number,
                "tool_name": "research_synthesis",
                "selected_context": selected_context_summary,
                "missing_prerequisites": list(context_pack.readiness_warnings),
                "read_tools": read_tool_summary,
                "tool_observations": read_tool_observations,
                "autonomous_retry": False,
            },
        )

        deterministic_output = self._deterministic_output(
            skill_id=skill_id,
            artifact_type=policy.artifact_type,
            message=str(payload.get("user_message") or ""),
            context=context_pack.context,
        )
        try:
            output_payload, artifact_status, run_status, synthesis_status = self._synthesize(
                policy=policy,
                message=str(payload.get("user_message") or ""),
                context=context_pack.context,
                deterministic_output=deterministic_output,
            )
        except Exception as exc:
            error_message = str(exc)
            run_repository.append_step(
                run_id=run.id,
                attempt_number=attempt_number,
                step_type="synthesis",
                label="Synthesize research artifact",
                status="failed",
                input_json={
                    "attempt_number": attempt_number,
                    "model_policy": policy.model_policy,
                },
                error_message=error_message,
            )
            run_repository.mark_finished(
                run.id,
                status="failed",
                model_name=self._model_name(),
                error_message=error_message,
            )
            research_repository.update_artifact(
                artifact_id,
                status="failed",
                model_name=self._model_name(),
                prompt_version=str(payload.get("prompt_version") or "research-agent-v2"),
                error_message=error_message,
            )
            raise
        run_repository.append_step(
            run_id=run.id,
            attempt_number=attempt_number,
            step_type="synthesis",
            label="Synthesize research artifact",
            status=synthesis_status,
            input_json={
                "attempt_number": attempt_number,
                "model_policy": policy.model_policy,
            },
            output_json={
                "attempt_number": attempt_number,
                "artifact_status": artifact_status,
                "model_backed": bool(output_payload.get("model_backed")),
                "model_required": bool(output_payload.get("model_required")),
                "schema_validation": output_payload.get("schema_validation"),
            },
            error_message=output_payload.get("setup_error"),
        )

        validation = validate_research_output(
            context=context_pack.context,
            output_payload=output_payload,
            readiness_warnings=context_pack.readiness_warnings,
            forced_status=run_status if run_status == "blocked" else None,
            entailment_grader=self._validation_entailment_grader(),
        )
        run_repository.create_validation_report(
            run_id=run.id,
            attempt_number=attempt_number,
            artifact_id=artifact_id,
            harness_status=str(validation["harness_status"]),
            missing_evidence=list(validation["missing_evidence"]),
            unsupported_claims=list(validation["unsupported_claims"]),
            readiness_blockers=list(validation["readiness_blockers"]),
            report_json=validation,
        )
        run_repository.append_step(
            run_id=run.id,
            attempt_number=attempt_number,
            step_type="validation",
            label="Validate evidence and readiness",
            status=str(validation["harness_status"]),
            output_json={**validation, "attempt_number": attempt_number},
        )

        evidence_payload = {
            **context_pack.context,
            "run_id": run.id,
            "attempt_number": attempt_number,
            "skill_id": policy.skill_id,
            "artifact_type": str(output_payload.get("artifact_type") or policy.artifact_type),
            "model_policy": policy.model_policy,
            "validation_report": validation,
        }
        saved_metadata = self._saved_artifact_metadata(
            dict(artifact.output_payload_json or {})
        )
        if saved_metadata:
            output_payload = {**output_payload, **saved_metadata}
        research_repository.update_artifact(
            artifact_id,
            title=str(output_payload.get("title") or artifact.title),
            status=artifact_status,
            output_payload=output_payload,
            evidence_payload=evidence_payload,
            model_name=self._model_name(),
            prompt_version=str(payload.get("prompt_version") or "research-agent-v2"),
            error_message=(
                output_payload.get("setup_error")
                if artifact_status in {"blocked", "failed"}
                else None
            ),
        )
        research_repository.create_message(
            thread_id=str(payload["thread_id"]),
            role="assistant",
            content=self._assistant_summary(
                output_payload=output_payload,
                artifact_status=artifact_status,
            ),
            artifact_id=artifact_id,
            metadata={
                "artifact_type": str(output_payload.get("artifact_type") or policy.artifact_type),
                "run_id": run.id,
                "attempt_number": attempt_number,
                "skill_id": skill_id,
                "harness_status": validation["harness_status"],
            },
        )
        run_repository.mark_finished(
            run.id,
            status=run_status,
            model_name=self._model_name(),
            error_message=(
                output_payload.get("setup_error")
                if run_status in {"blocked", "failed"}
                else None
            ),
        )
        return {
            "run_id": run.id,
            "attempt_number": attempt_number,
            "artifact_type": str(output_payload.get("artifact_type") or policy.artifact_type),
            "artifact_status": artifact_status,
            "evidence_paper_count": len(context_pack.context["papers"]),
        }

    def _saved_artifact_metadata(self, output_payload: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key in ("is_saved", "saved_format", "saved_title"):
            value = output_payload.get(key)
            if value is not None:
                metadata[key] = value
        return metadata

    def _mark_payload_failed(
        self,
        run_repository: ResearchAgentRunRepository,
        research_repository: ResearchRepository,
        *,
        payload: dict[str, Any],
        error_message: str,
    ) -> None:
        run_id = payload.get("run_id")
        if isinstance(run_id, str) and run_repository.get_run(run_id) is not None:
            run_repository.mark_finished(
                run_id,
                status="failed",
                model_name=self._model_name(),
                error_message=error_message,
            )
        artifact_id = payload.get("artifact_id")
        if (
            isinstance(artifact_id, str)
            and research_repository.get_artifact(artifact_id) is not None
        ):
            research_repository.update_artifact(
                artifact_id,
                status="failed",
                error_message=error_message,
            )

    def _terminal_run_result(
        self,
        run_repository: ResearchAgentRunRepository,
        *,
        artifact,  # noqa: ANN001
        run: ResearchAgentRun,
        fallback_artifact_type: str,
    ) -> dict[str, Any]:
        context_pack = run_repository.get_context_pack(run_id=run.id)
        context = dict(context_pack.context_json or {}) if context_pack is not None else {}
        output_payload = dict(artifact.output_payload_json or {})
        return {
            "run_id": run.id,
            "artifact_type": str(
                output_payload.get("artifact_type")
                or artifact.artifact_type
                or fallback_artifact_type
            ),
            "artifact_status": artifact.status,
            "evidence_paper_count": len(
                [paper for paper in context.get("papers", []) if isinstance(paper, dict)]
            ),
        }

    def _selected_context_summary(
        self,
        *,
        context: dict[str, Any],
        selected_item_counts: dict[str, int],
        intelligence_layers_summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "papers": self._summary_count(
                selected_item_counts,
                "papers",
                self._dict_list_count(context.get("papers")),
            ),
            "sources": self._summary_count(
                selected_item_counts,
                "sources",
                self._dict_list_count(context.get("sources")),
            ),
            "chunks": self._summary_count(
                selected_item_counts,
                "chunks",
                self._dict_list_count(context.get("chunks")),
            ),
            "evidence_spans": self._summary_count(
                selected_item_counts,
                "evidence_spans",
                self._dict_list_count(context.get("evidence_spans")),
            ),
            "figures": self._summary_count(
                selected_item_counts,
                "figures",
                self._dict_list_count(context.get("figures")),
            ),
            "tables": self._summary_count(
                selected_item_counts,
                "tables",
                self._dict_list_count(context.get("tables")),
            ),
            "structured_entities": self._summary_count(
                selected_item_counts,
                "structured_entities",
                self._dict_list_count(context.get("structured_entities")),
            ),
            "result_evidence": self._summary_count(
                selected_item_counts,
                "result_evidence",
                self._dict_list_count(context.get("result_evidence")),
            ),
            "sections": self._summary_count(selected_item_counts, "sections", 0),
            "structured_evidence": self._summary_count(
                selected_item_counts,
                "structured_evidence",
                0,
            ),
            "intelligence_layers": intelligence_layers_summary,
        }

    def _intelligence_layers_summary(
        self,
        *,
        context: dict[str, Any],
        selected_item_counts: dict[str, int],
    ) -> dict[str, Any]:
        layers = context.get("intelligence_layers")
        if not isinstance(layers, dict):
            layers = {}
        field_graph = layers.get("field_graph")
        if not isinstance(field_graph, dict):
            field_graph = {}
        study_brief = layers.get("study_brief")
        study_brief_version = (
            study_brief.get("version")
            if isinstance(study_brief, dict)
            else None
        )
        study_brief_count = self._summary_count(
            selected_item_counts,
            "study_brief",
            1 if isinstance(study_brief, dict) else 0,
        )
        return {
            "evidence_memory": self._summary_count(
                selected_item_counts,
                "evidence_memory",
                self._dict_list_count(layers.get("evidence_memory")),
            ),
            "pattern_memory": self._summary_count(
                selected_item_counts,
                "pattern_memory",
                self._dict_list_count(layers.get("pattern_memory")),
            ),
            "source_fact_memory": self._summary_count(
                selected_item_counts,
                "source_fact_memory",
                self._dict_list_count(layers.get("source_fact_memory")),
            ),
            "field_graph": {
                "nodes": self._summary_count(
                    selected_item_counts,
                    "graph_nodes",
                    self._dict_list_count(field_graph.get("nodes")),
                ),
                "edges": self._summary_count(
                    selected_item_counts,
                    "graph_edges",
                    self._dict_list_count(field_graph.get("edges")),
                ),
            },
            "study_brief": {
                "included": study_brief_count > 0,
                "version": study_brief_version,
            },
        }

    def _summary_count(
        self,
        selected_item_counts: dict[str, int],
        key: str,
        fallback: int,
    ) -> int:
        value = selected_item_counts.get(key)
        if isinstance(value, int):
            return value
        return fallback

    def _dict_list_count(self, value: Any) -> int:
        if not isinstance(value, list):
            return 0
        return len([item for item in value if isinstance(item, dict)])

    def _ensure_run(
        self,
        run_repository: ResearchAgentRunRepository,
        *,
        payload: dict[str, Any],
        policy: ResearchSkillPolicy,
        artifact_type: str,
    ) -> ResearchAgentRun:
        run_id = payload.get("run_id")
        if isinstance(run_id, str):
            existing = run_repository.get_run(run_id)
            if existing is not None:
                return existing
        return run_repository.create_run(
            thread_id=str(payload["thread_id"]) if payload.get("thread_id") else None,
            artifact_id=str(payload["artifact_id"]),
            collection_id=str(payload["collection_id"]),
            workspace_id=str(payload["workspace_id"]) if payload.get("workspace_id") else None,
            skill_id=policy.skill_id,
            artifact_type=artifact_type,
            model_policy=policy.model_policy,
            input_json={
                "message": payload.get("user_message"),
                "selected_paper_ids": list(payload.get("selected_paper_ids") or []),
                "source_ids": list(payload.get("source_ids") or []),
            },
        )

    def _deterministic_output(
        self,
        *,
        skill_id: str,
        artifact_type: str,
        message: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        result = default_research_skill_registry().run(
            skill_id,
            ResearchSkillContext(
                message=message,
                artifact_type=artifact_type,
                evidence_payload=context,
            ),
        )
        return result.output_payload

    def _synthesize(
        self,
        *,
        policy: ResearchSkillPolicy,
        message: str,
        context: dict[str, Any],
        deterministic_output: dict[str, Any],
    ) -> tuple[dict[str, Any], str, str, str]:
        if policy.model_policy == "deterministic":
            return (
                self._with_common_output(
                    deterministic_output,
                    context=context,
                    skill_id=policy.skill_id,
                    artifact_type=policy.artifact_type,
                    model_backed=False,
                ),
                "completed",
                "completed",
                "completed",
            )
        if self.model_client is None:
            if policy.allow_deterministic_fallback:
                return (
                    self._with_common_output(
                        {
                            **deterministic_output,
                            "model_warning": (
                                "Model-backed synthesis was unavailable; "
                                "deterministic fallback was used."
                            ),
                        },
                        context=context,
                        skill_id=policy.skill_id,
                        artifact_type=policy.artifact_type,
                        model_backed=False,
                    ),
                    "completed",
                    "completed",
                    "completed",
                )
            return (
                self._blocked_output(
                    policy=policy,
                    message=message,
                    context=context,
                    deterministic_output=deterministic_output,
                ),
                "blocked",
                "blocked",
                "blocked",
            )

        model_payload = self._validated_model_payload(
            policy=policy,
            message=message,
            context=context,
            deterministic_output=deterministic_output,
        )
        return (
            self._with_common_output(
                model_payload,
                context=context,
                skill_id=policy.skill_id,
                artifact_type=policy.artifact_type,
                model_backed=True,
            ),
            "completed",
            "completed",
            "completed",
        )

    def _validated_model_payload(
        self,
        *,
        policy: ResearchSkillPolicy,
        message: str,
        context: dict[str, Any],
        deterministic_output: dict[str, Any],
    ) -> dict[str, Any]:
        if self.model_client is None:
            raise ResearchModelOutputContractError("Model client is not configured.")

        contract = output_contract_prompt(
            skill_id=policy.skill_id,
            artifact_type=policy.artifact_type,
        )
        base_prompt_payload = {
            "message": message,
            "context": context,
            "deterministic_preview": deterministic_output,
            "output_contract": contract,
        }
        last_error = ""
        invalid_payload_keys: list[str] = []
        for attempt in range(1, MODEL_OUTPUT_SCHEMA_ATTEMPTS + 1):
            prompt_payload = dict(base_prompt_payload)
            if attempt > 1:
                prompt_payload["schema_repair"] = {
                    "attempt": attempt,
                    "previous_error": last_error,
                    "invalid_payload_keys": invalid_payload_keys,
                }
            raw_payload = self.model_client.synthesize(
                skill_id=policy.skill_id,
                artifact_type=policy.artifact_type,
                prompt_payload=prompt_payload,
            )
            invalid_payload_keys = (
                sorted(str(key) for key in raw_payload.keys())
                if isinstance(raw_payload, dict)
                else []
            )
            try:
                payload = normalize_model_output_payload(
                    skill_id=policy.skill_id,
                    artifact_type=policy.artifact_type,
                    payload=raw_payload,
                )
            except ResearchModelOutputContractError as exc:
                last_error = str(exc)
                continue
            payload["schema_validation"] = {
                "attempts": attempt,
                "schema_name": str(contract["schema_name"]),
                "status": "passed" if attempt == 1 else "repaired",
            }
            return payload

        raise ResearchModelOutputContractError(
            f"{policy.artifact_type} schema validation failed after "
            f"{MODEL_OUTPUT_SCHEMA_ATTEMPTS} attempt(s): {last_error}"
        )

    def _with_common_output(
        self,
        payload: dict[str, Any],
        *,
        context: dict[str, Any],
        skill_id: str,
        artifact_type: str,
        model_backed: bool,
    ) -> dict[str, Any]:
        output_payload = {
            **payload,
            "artifact_type": artifact_type,
            "skill_id": skill_id,
            "model_backed": model_backed,
            "model_name": self._model_name(),
        }
        return with_study_brief_update(
            ResearchSkillContext(
                message="",
                artifact_type=artifact_type,
                evidence_payload=context,
            ),
            output_payload,
            artifact_type=artifact_type,
            skill_id=skill_id,
        )

    def _blocked_output(
        self,
        *,
        policy: ResearchSkillPolicy,
        message: str,
        context: dict[str, Any],
        deterministic_output: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "artifact_type": policy.artifact_type,
            "skill_id": policy.skill_id,
            "title": f"{deterministic_output.get('title') or 'Research artifact'} blocked",
            "summary": "Model-backed synthesis is required for this research skill.",
            "request": message,
            "paper_count": len(context.get("papers", [])),
            "evidence_basis": [
                str(paper.get("title") or "Untitled paper")
                for paper in context.get("papers", [])
                if isinstance(paper, dict)
            ],
            "model_backed": False,
            "model_required": True,
            "setup_error": (
                "A Paperbase model provider is required for this model-backed research skill."
            ),
            "readiness_blockers": ["model_unavailable"],
            "deterministic_preview": self._without_study_brief_update(deterministic_output),
            "evidence_references": self._paper_references(context),
            "next_actions": [
                "Configure OPENAI_API_KEY for the OpenAI provider, or set "
                "PAPERBASE_MODEL_PROVIDER=claude_cli on a trusted local host. "
                "Codex CLI also requires PAPERBASE_ALLOW_AGENTIC_CLI=true.",
                "Rerun this instruction after model setup is available.",
            ],
        }

    def _without_study_brief_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        blocked_keys = {
            "study_brief_update",
            "study_brief_updates",
            "study_brief_update_source",
        }
        return {
            key: value
            for key, value in payload.items()
            if key not in blocked_keys
        }

    def _paper_references(self, context: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {
                "reference_type": "paper",
                "paper_id": str(paper.get("paper_id")),
                "label": str(paper.get("title") or "Untitled paper"),
            }
            for paper in context.get("papers", [])
            if isinstance(paper, dict) and paper.get("paper_id")
        ]

    def _model_name(self) -> str | None:
        if self.model_client is None:
            return None
        return str(getattr(self.model_client, "model_name", "custom-research-model"))

    def _validation_entailment_grader(self) -> Callable[..., Any] | None:
        if self.entailment_grader is not None:
            return self.entailment_grader
        return model_entailment_grader_from_env(self.model_client)

    def _assistant_summary(self, *, output_payload: dict[str, Any], artifact_status: str) -> str:
        title = str(output_payload.get("title") or "Research artifact")
        if artifact_status == "blocked":
            return f"{title}: model setup is required before this skill can run."
        return f"{title} generated with {output_payload.get('paper_count', 0)} paper(s) in context."
