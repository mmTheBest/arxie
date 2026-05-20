"""Local project registry for VSCode-like Paperbase library folders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.bootstrap import initialize_database
from paperbase.db.session import make_session_factory

PROJECT_DIR_NAME = ".arxie"
PROJECT_DATABASE_NAME = "paperbase.sqlite3"


@dataclass(frozen=True, slots=True)
class ProjectSummary:
    """A local Arxie project backed by one project-specific Paperbase DB."""

    id: str
    title: str
    root_path: str
    database_path: str
    last_opened_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "root_path": self.root_path,
            "database_path": self.database_path,
            "last_opened_at": self.last_opened_at,
        }


class ProjectNotFoundError(KeyError):
    """Raised when a request references an unknown project id."""


class ProjectRegistry:
    """Persist and resolve recent local Paperbase projects."""

    def __init__(self, *, registry_path: str | Path = "data/projects.json") -> None:
        self.registry_path = Path(registry_path)
        self._session_factories: dict[str, sessionmaker[Session]] = {}

    def open_project(self, *, root_path: str | Path, title: str | None = None) -> ProjectSummary:
        root = Path(root_path).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve()
        arxie_dir = root / PROJECT_DIR_NAME
        uploads_dir = arxie_dir / "uploads"
        cache_dir = arxie_dir / "cache"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        database_path = arxie_dir / PROJECT_DATABASE_NAME
        database_url = self._database_url(database_path)
        initialize_database(database_url)

        project = ProjectSummary(
            id=self.project_id_for_path(root),
            title=(title or root.name or "Arxie project").strip(),
            root_path=str(root),
            database_path=str(database_path),
            last_opened_at=_utc_now_iso(),
        )
        projects = {item.id: item for item in self.list_projects()}
        projects[project.id] = project
        self._write_projects(projects.values())
        self._session_factories[project.id] = make_session_factory(database_url)
        return project

    def list_projects(self) -> list[ProjectSummary]:
        payload = self._read_payload()
        projects = [
            self._project_from_dict(item)
            for item in payload.get("projects", [])
            if isinstance(item, dict)
        ]
        return sorted(projects, key=lambda item: item.last_opened_at, reverse=True)

    def get_project(self, project_id: str) -> ProjectSummary | None:
        for project in self.list_projects():
            if project.id == project_id:
                return project
        return None

    def session_factory_for(self, project_id: str) -> sessionmaker[Session]:
        project = self.get_project(project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)
        if project_id not in self._session_factories:
            initialize_database(self._database_url(Path(project.database_path)))
            self._session_factories[project_id] = make_session_factory(
                self._database_url(Path(project.database_path))
            )
        return self._session_factories[project_id]

    @staticmethod
    def project_id_for_path(root_path: str | Path) -> str:
        root = Path(root_path).expanduser().resolve()
        digest = sha256(str(root).encode("utf-8")).hexdigest()[:16]
        return f"project-{digest}"

    @staticmethod
    def _database_url(database_path: Path) -> str:
        return f"sqlite:///{database_path}"

    def _read_payload(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"projects": []}
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"projects": []}
        return payload if isinstance(payload, dict) else {"projects": []}

    def _write_projects(self, projects: Any) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"projects": [project.to_dict() for project in projects]}
        self.registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _project_from_dict(self, payload: dict[str, Any]) -> ProjectSummary:
        root_path = str(payload.get("root_path") or "")
        database_path = str(payload.get("database_path") or Path(root_path) / PROJECT_DIR_NAME / PROJECT_DATABASE_NAME)
        title = str(payload.get("title") or Path(root_path).name or "Arxie project")
        return ProjectSummary(
            id=str(payload.get("id") or self.project_id_for_path(root_path)),
            title=title,
            root_path=root_path,
            database_path=database_path,
            last_opened_at=str(payload.get("last_opened_at") or _utc_now_iso()),
        )


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
