"""Executable mypy release policy for the v0.2 release cut."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

_SUCCESS_RE = re.compile(r"Success: no issues found in (?P<checked>\d+) source files?")
_FAILURE_RE = re.compile(
    r"Found (?P<errors>\d+) errors in (?P<files>\d+) files? \(checked (?P<checked>\d+) source files?\)"
)


@dataclass(frozen=True, slots=True)
class MypySummary:
    """Parsed mypy summary line."""

    error_count: int
    files_with_errors: int
    checked_source_files: int | None
    passed: bool
    summary_line: str


@dataclass(frozen=True, slots=True)
class MypyGateCheck:
    """Single executable mypy policy check."""

    name: str
    description: str
    args: tuple[str, ...]
    max_errors: int
    max_files_with_errors: int


@dataclass(frozen=True, slots=True)
class MypyReleasePolicy:
    """Collection of mypy checks required for a release policy."""

    policy_id: str
    checks: tuple[MypyGateCheck, ...]


def build_v02_mypy_release_policy() -> MypyReleasePolicy:
    """Return the v0.2 typing policy adopted on 2026-03-16."""

    return MypyReleasePolicy(
        policy_id="arxie-v0.2-mypy-release-gate",
        checks=(
            MypyGateCheck(
                name="strict_scope",
                description=(
                    "Proposal workflow core and release-gate evaluator must stay strict-clean "
                    "when checked directly."
                ),
                args=(
                    "--follow-imports=skip",
                    "src/ra/proposal",
                    "src/ra/eval",
                ),
                max_errors=0,
                max_files_with_errors=0,
            ),
            MypyGateCheck(
                name="repo_baseline",
                description=(
                    "Full-repo strict debt is temporarily accepted only as a no-regression "
                    "baseline while non-v0.2 modules are remediated."
                ),
                args=("src", "tests"),
                max_errors=437,
                max_files_with_errors=44,
            ),
        ),
    )


def parse_mypy_summary(output: str, *, exit_code: int) -> MypySummary:
    """Parse mypy summary output into structured counts."""

    del exit_code  # The summary line is the source of truth for policy accounting.

    success_match = _SUCCESS_RE.search(output)
    if success_match is not None:
        summary_line = success_match.group(0)
        return MypySummary(
            error_count=0,
            files_with_errors=0,
            checked_source_files=int(success_match.group("checked")),
            passed=True,
            summary_line=summary_line,
        )

    failure_match = _FAILURE_RE.search(output)
    if failure_match is not None:
        summary_line = failure_match.group(0)
        return MypySummary(
            error_count=int(failure_match.group("errors")),
            files_with_errors=int(failure_match.group("files")),
            checked_source_files=int(failure_match.group("checked")),
            passed=False,
            summary_line=summary_line,
        )

    raise ValueError("Unable to parse mypy summary output.")


def evaluate_mypy_release_policy(
    policy: MypyReleasePolicy,
    *,
    run_check: Callable[[tuple[str, ...]], tuple[int, str]],
) -> dict[str, Any]:
    """Evaluate a mypy release policy using an injected command runner."""

    results: list[dict[str, Any]] = []
    overall_pass = True

    for check in policy.checks:
        exit_code, output = run_check(check.args)
        try:
            summary = parse_mypy_summary(output, exit_code=exit_code)
        except ValueError as exc:
            overall_pass = False
            results.append(
                {
                    "name": check.name,
                    "description": check.description,
                    "args": list(check.args),
                    "max_errors": check.max_errors,
                    "max_files_with_errors": check.max_files_with_errors,
                    "pass": False,
                    "parse_error": str(exc),
                }
            )
            continue

        passed = (
            summary.error_count <= check.max_errors
            and summary.files_with_errors <= check.max_files_with_errors
        )
        overall_pass = overall_pass and passed
        results.append(
            {
                "name": check.name,
                "description": check.description,
                "args": list(check.args),
                "max_errors": check.max_errors,
                "max_files_with_errors": check.max_files_with_errors,
                "error_count": summary.error_count,
                "files_with_errors": summary.files_with_errors,
                "checked_source_files": summary.checked_source_files,
                "summary_line": summary.summary_line,
                "pass": passed,
            }
        )

    return {
        "policy_id": policy.policy_id,
        "overall_pass": overall_pass,
        "checks": results,
    }


def _run_mypy_check(args: tuple[str, ...], *, python_bin: str) -> tuple[int, str]:
    command = [python_bin, "-m", "mypy", *args]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    output_parts = [completed.stdout.strip(), completed.stderr.strip()]
    output = "\n".join(part for part in output_parts if part).strip()
    return completed.returncode, output


def run_v02_mypy_release_gate(*, python_bin: str | None = None) -> dict[str, Any]:
    """Run the v0.2 mypy release policy in the current repository."""

    interpreter = python_bin or sys.executable
    policy = build_v02_mypy_release_policy()
    report = evaluate_mypy_release_policy(
        policy,
        run_check=lambda args: _run_mypy_check(args, python_bin=interpreter),
    )
    report["python_bin"] = interpreter
    report["commands"] = [
        [interpreter, "-m", "mypy", *check.args]
        for check in policy.checks
    ]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the release typing gate."""

    parser = argparse.ArgumentParser(description="Run the Arxie v0.2 mypy release gate")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to invoke `python -m mypy`.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_v02_mypy_release_gate(python_bin=args.python_bin)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
