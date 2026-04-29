from __future__ import annotations

from pathlib import Path
import subprocess

from typer.testing import CliRunner

from paperbase.launcher import cli
from paperbase.launcher.cli import app


def test_run_command_starts_stack_waits_for_ready_and_opens_browser(monkeypatch) -> None:
    runner = CliRunner()
    compose_calls: list[list[str]] = []
    ready_calls: list[tuple[str, float]] = []
    service_checks: list[tuple[tuple[str, ...], str | None]] = []
    opened_urls: list[str] = []

    monkeypatch.setattr(
        "paperbase.launcher.cli._run_compose",
        lambda args, env=None: compose_calls.append(list(args)),
    )
    monkeypatch.setattr("paperbase.launcher.cli._ensure_container_runtime", lambda: None)
    monkeypatch.setattr(
        "paperbase.launcher.cli._compose_env",
        lambda: {"OPENAI_API_KEY": "present"},
    )
    monkeypatch.setattr(
        "paperbase.launcher.cli._ensure_runtime_services_running",
        lambda services, env=None: service_checks.append((tuple(services), env.get("OPENAI_API_KEY") if env else None)),
    )
    monkeypatch.setattr(
        "paperbase.launcher.cli._wait_until_ready",
        lambda base_url, timeout_seconds: ready_calls.append((base_url, timeout_seconds)),
    )
    monkeypatch.setattr(
        "paperbase.launcher.cli.webbrowser.open",
        lambda url: opened_urls.append(url),
    )

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    assert compose_calls == [
        ["up", "-d", "postgres", "elasticsearch", "minio", "redis"],
        ["run", "--rm", "paperbase-migrate"],
        ["up", "-d", "paperbase-api", "paperbase-worker"],
    ]
    assert service_checks == [(("paperbase-api", "paperbase-worker"), "present")]
    assert ready_calls == [("http://localhost:8080", 120.0)]
    assert opened_urls == ["http://localhost:8080/app"]
    assert "Arxie is ready at http://localhost:8080/app" in result.stdout


def test_run_command_supports_no_browser(monkeypatch) -> None:
    runner = CliRunner()
    opened_urls: list[str] = []

    monkeypatch.setattr("paperbase.launcher.cli._run_compose", lambda args, env=None: None)
    monkeypatch.setattr("paperbase.launcher.cli._ensure_container_runtime", lambda: None)
    monkeypatch.setattr("paperbase.launcher.cli._compose_env", lambda: {})
    monkeypatch.setattr(
        "paperbase.launcher.cli._ensure_runtime_services_running",
        lambda services, env=None: None,
    )
    monkeypatch.setattr(
        "paperbase.launcher.cli._wait_until_ready",
        lambda base_url, timeout_seconds: None,
    )
    monkeypatch.setattr(
        "paperbase.launcher.cli.webbrowser.open",
        lambda url: opened_urls.append(url),
    )

    result = runner.invoke(app, ["run", "--no-browser"])

    assert result.exit_code == 0
    assert opened_urls == []


def test_install_shortcut_writes_double_clickable_command_file(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "install-shortcut",
            "--output",
            str(tmp_path / "Arxie.command"),
            "--python",
            "/tmp/example-venv/bin/python",
        ],
    )

    assert result.exit_code == 0
    shortcut_path = tmp_path / "Arxie.command"
    assert shortcut_path.exists()
    shortcut_text = shortcut_path.read_text(encoding="utf-8")
    assert '/tmp/example-venv/bin/arxie-local' in shortcut_text
    assert " run" in shortcut_text
    assert "chmod +x" not in shortcut_text


def test_open_command_opens_workspace_url(monkeypatch) -> None:
    runner = CliRunner()
    opened_urls: list[str] = []

    monkeypatch.setattr(
        "paperbase.launcher.cli.webbrowser.open",
        lambda url: opened_urls.append(url),
    )

    result = runner.invoke(app, ["open", "--base-url", "http://127.0.0.1:9999"])

    assert result.exit_code == 0
    assert opened_urls == ["http://127.0.0.1:9999/app"]


def test_ensure_container_runtime_starts_colima_for_colima_context(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, cwd=None, check=None, stdout=None, stderr=None, capture_output=False, text=False):  # noqa: ANN001
        calls.append(list(command))
        if command[:2] == ["docker", "info"]:
            if calls.count(["docker", "info"]) == 1:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        if command[:3] == ["docker", "context", "show"]:
            return subprocess.CompletedProcess(command, 0, stdout="colima\n")
        if command[:2] == ["colima", "start"]:
            return subprocess.CompletedProcess(command, 0)
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/opt/homebrew/bin/colima" if name == "colima" else None)

    cli._ensure_container_runtime()

    assert calls == [
        ["docker", "info"],
        ["docker", "context", "show"],
        ["colima", "start"],
        ["docker", "info"],
    ]


def test_run_compose_retries_transient_docker_failures(monkeypatch) -> None:
    calls: list[list[str]] = []
    sleep_calls: list[float] = []

    def fake_run(
        command,
        cwd=None,
        check=None,
        capture_output=False,
        text=False,
        env=None,
    ):  # noqa: ANN001
        calls.append(list(command))
        if len(calls) == 1:
            raise subprocess.CalledProcessError(
                1,
                command,
                stderr='failed to do request: Get "https://registry-1.docker.io/v2/": EOF',
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    cli._run_compose(["run", "--rm", "paperbase-migrate"])

    assert calls == [
        [
            "docker",
            "compose",
            "-f",
            str(cli._compose_file()),
            "run",
            "--rm",
            "paperbase-migrate",
        ],
        [
            "docker",
            "compose",
            "-f",
            str(cli._compose_file()),
            "run",
            "--rm",
            "paperbase-migrate",
        ],
    ]
    assert sleep_calls == [3.0]


def test_run_compose_raises_non_transient_failure(monkeypatch) -> None:
    def fake_run(
        command,
        cwd=None,
        check=None,
        capture_output=False,
        text=False,
        env=None,
    ):  # noqa: ANN001
        raise subprocess.CalledProcessError(1, command, stderr="service failed permanently")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    try:
        cli._run_compose(["up", "-d", "paperbase-api"])
    except subprocess.CalledProcessError as exc:
        assert exc.stderr == "service failed permanently"
    else:
        raise AssertionError("Expected CalledProcessError to be raised")


def test_compose_env_reads_root_env_when_running_from_worktree(monkeypatch, tmp_path: Path) -> None:
    canonical_root = tmp_path / "repo"
    worktree_root = canonical_root / ".worktrees" / "release-paperbase-v1"
    canonical_root.mkdir()
    worktree_root.mkdir(parents=True)
    (canonical_root / ".env").write_text("OPENAI_API_KEY=from-root\nPAPERBASE_API_PORT=8081\n", encoding="utf-8")

    monkeypatch.setattr(cli, "_repo_root", lambda: worktree_root)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    env = cli._compose_env()

    assert env["OPENAI_API_KEY"] == "from-root"
    assert env["PAPERBASE_API_PORT"] == "8081"


def test_ensure_runtime_services_running_raises_when_api_or_worker_missing(monkeypatch) -> None:
    def fake_run(
        command,
        cwd=None,
        check=None,
        capture_output=False,
        text=False,
        env=None,
    ):  # noqa: ANN001
        return subprocess.CompletedProcess(command, 0, stdout="paperbase-api\n")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    try:
        cli._ensure_runtime_services_running(["paperbase-api", "paperbase-worker"])
    except RuntimeError as exc:
        assert "paperbase-worker" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when a required service is missing")
