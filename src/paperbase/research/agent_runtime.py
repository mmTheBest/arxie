"""Traceable Paperbase research-agent runtime."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from paperbase.db.models import ResearchAgentRun
from paperbase.db.repositories import ResearchAgentRunRepository, ResearchRepository
from paperbase.research.context_builder import PaperbaseResearchContextBuilder
from paperbase.research.harness import validate_research_output
from paperbase.research.skill_policies import ResearchSkillPolicy, policy_for_skill
from paperbase.research.skills import (
    ResearchSkillContext,
    UnsupportedResearchSkillError,
    default_research_skill_registry,
    select_research_skill,
)

TERMINAL_RUN_STATUSES = {"completed", "blocked", "failed"}


class PaperbaseResearchAgentRuntime:
    """Execute one research-agent run with context, synthesis, and validation traces."""

    def __init__(self, *, model_client: object | None = None) -> None:
        self.model_client = model_client

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
                suggestion_id=str(payload["suggestion_id"]) if payload.get("suggestion_id") else None,
                artifact_type=str(payload["artifact_type"]) if payload.get("artifact_type") else None,
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
        run_repository.mark_running(run.id)

        context_pack = PaperbaseResearchContextBuilder(session).build(
            collection_id=str(payload["collection_id"]),
            task_type=skill_id,
            message=str(payload.get("user_message") or ""),
            selected_paper_ids=[str(item) for item in list(payload.get("selected_paper_ids") or [])],
            workspace_id=str(payload["workspace_id"]) if payload.get("workspace_id") else None,
            source_ids=[str(item) for item in list(payload.get("source_ids") or [])],
        )
        run_repository.create_context_pack(
            run_id=run.id,
            collection_id=str(payload["collection_id"]),
            workspace_id=str(payload["workspace_id"]) if payload.get("workspace_id") else None,
            task_type=skill_id,
            context_json=context_pack.context,
            selected_item_counts=context_pack.selected_item_counts,
            readiness_warnings=context_pack.readiness_warnings,
            cache_key=context_pack.cache_key,
        )
        run_repository.append_step(
            run_id=run.id,
            step_type="context",
            label="Build study context",
            output_json={
                "selected_item_counts": context_pack.selected_item_counts,
                "readiness_warnings": context_pack.readiness_warnings,
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
                step_type="synthesis",
                label="Synthesize research artifact",
                status="failed",
                input_json={"model_policy": policy.model_policy},
                error_message=error_message,
            )
            run_repository.mark_finished(
                run.id,
                status="failed",
                model_name=self._model_name(),
                error_message=error_message,
            )
            raise
        run_repository.append_step(
            run_id=run.id,
            step_type="synthesis",
            label="Synthesize research artifact",
            status=synthesis_status,
            input_json={"model_policy": policy.model_policy},
            output_json={
                "artifact_status": artifact_status,
                "model_backed": bool(output_payload.get("model_backed")),
                "model_required": bool(output_payload.get("model_required")),
            },
            error_message=output_payload.get("setup_error"),
        )

        validation = validate_research_output(
            context=context_pack.context,
            output_payload=output_payload,
            readiness_warnings=context_pack.readiness_warnings,
            forced_status=run_status if run_status == "blocked" else None,
        )
        run_repository.create_validation_report(
            run_id=run.id,
            artifact_id=artifact_id,
            harness_status=str(validation["harness_status"]),
            missing_evidence=list(validation["missing_evidence"]),
            unsupported_claims=list(validation["unsupported_claims"]),
            readiness_blockers=list(validation["readiness_blockers"]),
            report_json=validation,
        )
        run_repository.append_step(
            run_id=run.id,
            step_type="validation",
            label="Validate evidence and readiness",
            status=str(validation["harness_status"]),
            output_json=validation,
        )

        evidence_payload = {
            **context_pack.context,
            "run_id": run.id,
            "skill_id": policy.skill_id,
            "artifact_type": str(output_payload.get("artifact_type") or policy.artifact_type),
            "model_policy": policy.model_policy,
            "validation_report": validation,
        }
        research_repository.update_artifact(
            artifact_id,
            title=str(output_payload.get("title") or artifact.title),
            status=artifact_status,
            output_payload=output_payload,
            evidence_payload=evidence_payload,
            model_name=self._model_name(),
            prompt_version=str(payload.get("prompt_version") or "research-agent-v2"),
            error_message=output_payload.get("setup_error") if artifact_status in {"blocked", "failed"} else None,
        )
        research_repository.create_message(
            thread_id=str(payload["thread_id"]),
            role="assistant",
            content=self._assistant_summary(output_payload=output_payload, artifact_status=artifact_status),
            artifact_id=artifact_id,
            metadata={
                "artifact_type": str(output_payload.get("artifact_type") or policy.artifact_type),
                "run_id": run.id,
                "skill_id": skill_id,
                "harness_status": validation["harness_status"],
            },
        )
        run_repository.mark_finished(
            run.id,
            status=run_status,
            model_name=self._model_name(),
            error_message=output_payload.get("setup_error") if run_status in {"blocked", "failed"} else None,
        )
        return {
            "run_id": run.id,
            "artifact_type": str(output_payload.get("artifact_type") or policy.artifact_type),
            "artifact_status": artifact_status,
            "evidence_paper_count": len(context_pack.context["papers"]),
        }

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
        if isinstance(artifact_id, str) and research_repository.get_artifact(artifact_id) is not None:
            research_repository.update_artifact(artifact_id, status="failed", error_message=error_message)

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
                output_payload.get("artifact_type") or artifact.artifact_type or fallback_artifact_type
            ),
            "artifact_status": artifact.status,
            "evidence_paper_count": len(
                [paper for paper in context.get("papers", []) if isinstance(paper, dict)]
            ),
        }

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
                            "model_warning": "Model-backed synthesis was unavailable; deterministic fallback was used.",
                        },
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

        model_payload = self.model_client.synthesize(
            skill_id=policy.skill_id,
            artifact_type=policy.artifact_type,
            prompt_payload={
                "message": message,
                "context": context,
                "deterministic_preview": deterministic_output,
            },
        )
        return (
            self._with_common_output(
                model_payload,
                skill_id=policy.skill_id,
                artifact_type=policy.artifact_type,
                model_backed=True,
            ),
            "completed",
            "completed",
            "completed",
        )

    def _with_common_output(
        self,
        payload: dict[str, Any],
        *,
        skill_id: str,
        artifact_type: str,
        model_backed: bool,
    ) -> dict[str, Any]:
        return {
            "artifact_type": artifact_type,
            "skill_id": skill_id,
            "model_backed": model_backed,
            "model_name": self._model_name(),
            **payload,
        }

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
            "setup_error": "OPENAI_API_KEY is required for this model-backed research skill.",
            "readiness_blockers": ["model_unavailable"],
            "deterministic_preview": deterministic_output,
            "evidence_references": self._paper_references(context),
            "next_actions": ["Configure OPENAI_API_KEY, then rerun this instruction."],
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

    def _assistant_summary(self, *, output_payload: dict[str, Any], artifact_status: str) -> str:
        title = str(output_payload.get("title") or "Research artifact")
        if artifact_status == "blocked":
            return f"{title}: model setup is required before this skill can run."
        return f"{title} generated with {output_payload.get('paper_count', 0)} paper(s) in context."
