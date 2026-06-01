#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any, NamedTuple

COMMENT_MARKER = "<!-- arxie-pr-quality -->"

REQUIRED_SECTIONS = (
    "Summary",
    "Why",
    "What Changed",
    "Verification",
    "Risks",
    "Docs Updated",
    "Model Used",
    "Release/Main Impact",
)

PLACEHOLDER_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "todo",
    "tbd",
    "- none",
    "not applicable",
}

CODE_PREFIXES = ("src/", "services/")
DOC_PREFIXES = ("docs/",)
TEST_PREFIXES = ("tests/",)
RUNTIME_PREFIXES = ("src/", "services/", "infra/", "scripts/", ".github/")
RUNTIME_FILES = (
    "pyproject.toml",
    "uv.lock",
    "Dockerfile",
    ".env.example",
    "alembic.ini",
)
FORBIDDEN_PREFIXES = (
    ".venv/",
    ".worktrees/",
    "data/",
    "logs/",
    "results/",
    "__pycache__/",
)
FORBIDDEN_SUFFIXES = (
    ".db",
    ".db-shm",
    ".db-wal",
    ".sqlite",
    ".sqlite3",
    ".sqlite3-shm",
    ".sqlite3-wal",
)

SENSITIVE_PATH_PREFIXES = (
    ".github/scripts/",
    ".github/workflows/",
    "infra/",
    "services/paperbase_api/routes/ingest.py",
    "services/paperbase_api/upload_parser.py",
    "services/paperbase_api/path_policy.py",
)
SENSITIVE_EXACT_PATHS = {
    "AGENTS.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "docs/RELEASE_MANIFEST.md",
    "docs/REPO_MANAGEMENT.md",
    "scripts/with-env.sh",
    "src/paperbase/db/session.py",
    "src/paperbase/model_providers.py",
}
RELEASE_SURFACE_PREFIXES = ("src/", "services/", "infra/")
RELEASE_SURFACE_FILES = {
    ".env.example",
    "Dockerfile",
    "README.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    "alembic.ini",
    "docs/API.md",
    "docs/MIGRATION.md",
    "docs/RELEASE_MANIFEST.md",
    "docs/USAGE_EXAMPLES.md",
}

SECRET_PATTERNS = (
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)(?:api[_-]?key|token|secret|password|credential)\s*[:=]\s*[\"']?[A-Za-z0-9_./+=:-]{16,}"
    ),
)


class PullRequestFile(NamedTuple):
    filename: str
    status: str
    patch: str | None = None


class QualityReport(NamedTuple):
    failures: list[str]
    informational: list[str]
    labels: set[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def analyze_pull_request(
    *,
    title: str,
    body: str,
    files: list[PullRequestFile],
    author: str,
    branch: str,
) -> QualityReport:
    del author, branch
    failures: list[str] = []
    informational: list[str] = []
    labels: set[str] = set()
    changed_paths = [file.filename for file in files]
    labels.update(_area_labels(changed_paths))

    if not title.strip():
        failures.append("Add a descriptive PR title.")

    missing_sections = _missing_required_sections(body)
    if missing_sections:
        failures.append(
            "Complete the PR template sections: " + ", ".join(missing_sections) + "."
        )

    if _touches_code(changed_paths):
        if not _touches_tests(changed_paths):
            labels.add("needs-tests")
            failures.append(
                "Add or update tests under `tests/` for runtime changes under "
                "`src/` or `services/`."
            )

    if _touches_runtime_or_ci(changed_paths) and not _touches_docs(changed_paths):
        labels.add("needs-docs")
        failures.append(
            "Update the matching docs for behavior, runtime, workflow, or release-surface changes."
        )

    if _touches_docs(changed_paths):
        labels.add("area:docs")

    if _touches_sensitive_paths(changed_paths):
        labels.update({"security-review-required", "maintainer-review-required"})
        informational.append("Touches security-sensitive or maintainer-owned repository paths.")

    if _touches_release_surface(changed_paths):
        labels.add("release-surface")
        informational.append("Touches files that may ship on the public `main` release branch.")

    for path in changed_paths:
        if _is_forbidden_artifact(path):
            labels.add("security-review-required")
            failures.append(f"Remove forbidden local artifact `{path}` from the PR.")

    for file in files:
        if file.patch and _patch_has_secret_like_value(file.patch):
            labels.add("security-review-required")
            failures.append(f"Remove secret-like value from `{file.filename}` before review.")

    return QualityReport(failures=failures, informational=informational, labels=labels)


def build_comment(*, author: str, report: QualityReport) -> str:
    status = "passed" if report.passed else "needs attention"
    lines = [
        COMMENT_MARKER,
        f"### Arxie PR quality check: {status}",
        "",
        f"@{author}, this bot checks PR hygiene without checking out or executing PR code.",
        "",
    ]

    if report.failures:
        lines.append("Required fixes:")
        lines.extend(f"- [ ] {failure}" for failure in report.failures)
        lines.append("")
    else:
        lines.append("No blocking PR hygiene issues were detected.")
        lines.append("")

    if report.informational:
        lines.append("Notes:")
        lines.extend(f"- {item}" for item in report.informational)
        lines.append("")

    if report.labels:
        labels = ", ".join(f"`{label}`" for label in sorted(report.labels))
        lines.append(f"Suggested labels: {labels}")
        lines.append("")

    lines.append("Maintainers still need to review code quality, product behavior, and")
    lines.append("GitHub branch settings.")
    return "\n".join(lines)


def _missing_required_sections(body: str) -> list[str]:
    sections = _parse_sections(body)
    missing: list[str] = []
    for required in REQUIRED_SECTIONS:
        value = _visible_section_text(sections.get(required.casefold(), ""))
        normalized = re.sub(r"\s+", " ", value).strip().casefold()
        if normalized in PLACEHOLDER_VALUES:
            missing.append(required)
    return missing


def _visible_section_text(value: str) -> str:
    without_comments = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL)
    return without_comments.strip()


def _parse_sections(body: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in body.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current = heading.group(1).strip().casefold()
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def _touches_code(paths: list[str]) -> bool:
    return any(path.startswith(CODE_PREFIXES) for path in paths)


def _touches_tests(paths: list[str]) -> bool:
    return any(path.startswith(TEST_PREFIXES) for path in paths)


def _touches_docs(paths: list[str]) -> bool:
    return any(path.startswith(DOC_PREFIXES) or path == "README.md" for path in paths)


def _touches_runtime_or_ci(paths: list[str]) -> bool:
    return any(
        path.startswith(RUNTIME_PREFIXES)
        or path.startswith(".github/workflows/")
        or path in RUNTIME_FILES
        for path in paths
    )


def _touches_sensitive_paths(paths: list[str]) -> bool:
    return any(
        path in SENSITIVE_EXACT_PATHS or path.startswith(SENSITIVE_PATH_PREFIXES)
        for path in paths
    )


def _touches_release_surface(paths: list[str]) -> bool:
    return any(
        path in RELEASE_SURFACE_FILES or path.startswith(RELEASE_SURFACE_PREFIXES)
        for path in paths
    )


def _is_forbidden_artifact(path: str) -> bool:
    if path == ".env":
        return True
    if path.startswith(".env.") and path != ".env.example":
        return True
    return path.startswith(FORBIDDEN_PREFIXES) or path.endswith(FORBIDDEN_SUFFIXES)


def _patch_has_secret_like_value(patch: str) -> bool:
    added_lines = "\n".join(
        line for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    return any(pattern.search(added_lines) for pattern in SECRET_PATTERNS)


def _area_labels(paths: list[str]) -> set[str]:
    labels: set[str] = set()
    for path in paths:
        if path.startswith("services/paperbase_api/static/"):
            labels.add("area:ui")
        elif path.startswith("services/paperbase_api/"):
            labels.add("area:api")
        elif path.startswith("services/paperbase_worker/"):
            labels.add("area:worker")
        elif path == "src/paperbase/model_providers.py" or path.startswith(
            "src/paperbase/extract/"
        ):
            labels.add("area:model-provider")
        elif path.startswith("src/paperbase/"):
            labels.add("area:paperbase-core")
        elif path.startswith(".github/"):
            labels.add("area:ci")
    return labels


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    pr_number = os.environ.get("PR_NUMBER")
    if not token or not repository or not pr_number:
        print("GITHUB_TOKEN, GITHUB_REPOSITORY, and PR_NUMBER are required.", file=sys.stderr)
        return 2

    client = GitHubClient(token=token, repository=repository)
    pr = client.get_pull_request(pr_number)
    files = client.list_pull_request_files(pr_number)
    report = analyze_pull_request(
        title=str(pr.get("title") or ""),
        body=str(pr.get("body") or ""),
        files=files,
        author=str(pr.get("user", {}).get("login") or "contributor"),
        branch=str(pr.get("head", {}).get("ref") or ""),
    )
    comment = build_comment(
        author=str(pr.get("user", {}).get("login") or "contributor"),
        report=report,
    )
    client.upsert_issue_comment(pr_number, comment)
    client.add_labels(pr_number, sorted(report.labels))
    return 0 if report.passed else 1


class GitHubClient:
    def __init__(self, *, token: str, repository: str) -> None:
        self.token = token
        self.repository = repository
        self.base_url = f"https://api.github.com/repos/{repository}"

    def get_pull_request(self, pr_number: str) -> dict[str, Any]:
        return self._request_json(f"{self.base_url}/pulls/{pr_number}")

    def list_pull_request_files(self, pr_number: str) -> list[PullRequestFile]:
        files: list[PullRequestFile] = []
        page = 1
        while True:
            payload = self._request_json(
                f"{self.base_url}/pulls/{pr_number}/files?per_page=100&page={page}"
            )
            if not isinstance(payload, list) or not payload:
                return files
            for item in payload:
                files.append(
                    PullRequestFile(
                        filename=str(item.get("filename", "")),
                        status=str(item.get("status", "")),
                        patch=item.get("patch"),
                    )
                )
            page += 1

    def upsert_issue_comment(self, issue_number: str, body: str) -> None:
        comments = self._request_json(
            f"{self.base_url}/issues/{issue_number}/comments?per_page=100"
        )
        if isinstance(comments, list):
            for comment in comments:
                if COMMENT_MARKER in str(comment.get("body", "")):
                    self._request_json(
                        str(comment["url"]),
                        method="PATCH",
                        payload={"body": body},
                    )
                    return
        self._request_json(
            f"{self.base_url}/issues/{issue_number}/comments",
            method="POST",
            payload={"body": body},
        )

    def add_labels(self, issue_number: str, labels: list[str]) -> None:
        if not labels:
            return
        try:
            self._request_json(
                f"{self.base_url}/issues/{issue_number}/labels",
                method="POST",
                payload={"labels": labels},
            )
        except urllib.error.HTTPError as exc:
            if exc.code in {404, 422}:
                print(
                    "Skipping label application because one or more labels are missing.",
                    file=sys.stderr,
                )
                return
            raise

    def _request_json(
        self,
        url: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "arxie-pr-quality",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
