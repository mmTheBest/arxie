"""Local JSON persistence for study-agent state."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from ra.study.models import (
    StudyAgentRun,
    StudyBrief,
    StudySource,
    utc_now_iso,
)

DEFAULT_STUDY_STORE_PATH = Path("memory/arxie_studies.json")


class StudyStoreError(RuntimeError):
    """Raised when study-agent state cannot be loaded or saved."""


class StudyNotFoundError(KeyError):
    """Raised when a study brief is missing."""

    def __init__(self, study_id: str) -> None:
        super().__init__(f"Study '{study_id}' was not found.")
        self.study_id = study_id


class StudyAlreadyExistsError(ValueError):
    """Raised when creating a duplicate study."""

    def __init__(self, study_id: str) -> None:
        super().__init__(f"Study '{study_id}' already exists.")
        self.study_id = study_id


class StudyVersionConflictError(ValueError):
    """Raised when an optimistic write uses a stale brief version."""

    def __init__(self, study_id: str, expected_version: int, current_version: int) -> None:
        super().__init__(
            f"Version conflict for study '{study_id}': "
            f"expected {expected_version}, current {current_version}."
        )
        self.study_id = study_id
        self.expected_version = expected_version
        self.current_version = current_version


class StudyRunNotFoundError(KeyError):
    """Raised when a study-agent run is missing."""

    def __init__(self, study_id: str, run_id: str) -> None:
        super().__init__(f"Run '{run_id}' was not found for study '{study_id}'.")
        self.study_id = study_id
        self.run_id = run_id


class JsonStudyStore:
    """Small JSON-backed store for local-first study-agent state."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else self.default_path()

    @staticmethod
    def default_path() -> Path:
        configured = os.getenv("RA_STUDY_STORE_PATH")
        return Path(configured) if configured else DEFAULT_STUDY_STORE_PATH

    def create_brief(self, brief: StudyBrief) -> StudyBrief:
        state = self._load_state()
        if brief.study_id in state["briefs"]:
            raise StudyAlreadyExistsError(brief.study_id)
        state["briefs"][brief.study_id] = brief.to_dict()
        state["sources"].setdefault(brief.study_id, {})
        state["runs"].setdefault(brief.study_id, {})
        self._save_state(state)
        return StudyBrief.from_dict(state["briefs"][brief.study_id])

    def get_brief(self, study_id: str) -> StudyBrief | None:
        state = self._load_state()
        payload = state["briefs"].get(str(study_id))
        if not isinstance(payload, dict):
            return None
        return StudyBrief.from_dict(payload)

    def list_briefs(self) -> tuple[StudyBrief, ...]:
        state = self._load_state()
        return tuple(
            StudyBrief.from_dict(payload)
            for _, payload in sorted(state["briefs"].items())
            if isinstance(payload, dict)
        )

    def save_brief(self, brief: StudyBrief, *, expected_version: int) -> StudyBrief:
        state = self._load_state()
        current_payload = state["briefs"].get(brief.study_id)
        if not isinstance(current_payload, dict):
            raise StudyNotFoundError(brief.study_id)
        current = StudyBrief.from_dict(current_payload)
        if current.version != expected_version:
            raise StudyVersionConflictError(
                brief.study_id,
                expected_version=expected_version,
                current_version=current.version,
            )

        updated = replace(
            brief,
            version=current.version + 1,
            created_at=current.created_at,
            updated_at=utc_now_iso(),
        )
        state["briefs"][brief.study_id] = updated.to_dict()
        self._save_state(state)
        return StudyBrief.from_dict(state["briefs"][brief.study_id])

    def add_source(self, source: StudySource) -> StudySource:
        state = self._load_state()
        if source.study_id not in state["briefs"]:
            raise StudyNotFoundError(source.study_id)
        state["sources"].setdefault(source.study_id, {})
        state["sources"][source.study_id][source.source_id] = source.to_dict()
        self._save_state(state)
        return StudySource.from_dict(state["sources"][source.study_id][source.source_id])

    def list_sources(self, study_id: str) -> tuple[StudySource, ...]:
        state = self._load_state()
        if str(study_id) not in state["briefs"]:
            raise StudyNotFoundError(str(study_id))
        sources = state["sources"].get(str(study_id), {})
        return tuple(
            StudySource.from_dict(payload)
            for _, payload in sorted(sources.items())
            if isinstance(payload, dict)
        )

    def save_run(self, run: StudyAgentRun) -> StudyAgentRun:
        state = self._load_state()
        if run.study_id not in state["briefs"]:
            raise StudyNotFoundError(run.study_id)
        state["runs"].setdefault(run.study_id, {})
        state["runs"][run.study_id][run.run_id] = run.to_dict()
        self._save_state(state)
        return StudyAgentRun.from_dict(state["runs"][run.study_id][run.run_id])

    def get_run(self, study_id: str, run_id: str) -> StudyAgentRun | None:
        state = self._load_state()
        if str(study_id) not in state["briefs"]:
            raise StudyNotFoundError(str(study_id))
        payload = state["runs"].get(str(study_id), {}).get(str(run_id))
        if not isinstance(payload, dict):
            return None
        return StudyAgentRun.from_dict(payload)

    def list_runs(self, study_id: str) -> tuple[StudyAgentRun, ...]:
        state = self._load_state()
        if str(study_id) not in state["briefs"]:
            raise StudyNotFoundError(str(study_id))
        runs = state["runs"].get(str(study_id), {})
        return tuple(
            StudyAgentRun.from_dict(payload)
            for _, payload in sorted(runs.items())
            if isinstance(payload, dict)
        )

    def _load_state(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"briefs": {}, "sources": {}, "runs": {}}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StudyStoreError(f"Study store is not valid JSON: {self.path}") from exc
        except OSError as exc:
            raise StudyStoreError(f"Study store could not be read: {self.path}") from exc

        if not isinstance(raw, dict):
            raise StudyStoreError("Study store root must be an object.")
        return {
            "briefs": dict(raw.get("briefs", {})),
            "sources": dict(raw.get("sources", {})),
            "runs": dict(raw.get("runs", {})),
        }

    def _save_state(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self.path.parent),
                delete=False,
            ) as handle:
                handle.write(payload)
                handle.write("\n")
                temp_name = handle.name
            Path(temp_name).replace(self.path)
        except OSError as exc:
            raise StudyStoreError(f"Study store could not be written: {self.path}") from exc
