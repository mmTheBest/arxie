from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from paperbase.launcher.cli import app


def test_run_command_starts_stack_waits_for_ready_and_opens_browser(monkeypatch) -> None:
    runner = CliRunner()
    compose_calls: list[list[str]] = []
    ready_calls: list[tuple[str, float]] = []
    opened_urls: list[str] = []

    monkeypatch.setattr(
        "paperbase.launcher.cli._run_compose",
        lambda args: compose_calls.append(list(args)),
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
    assert ready_calls == [("http://localhost:8080", 120.0)]
    assert opened_urls == ["http://localhost:8080/app"]
    assert "Arxie is ready at http://localhost:8080/app" in result.stdout


def test_run_command_supports_no_browser(monkeypatch) -> None:
    runner = CliRunner()
    opened_urls: list[str] = []

    monkeypatch.setattr("paperbase.launcher.cli._run_compose", lambda args: None)
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
    assert '/tmp/example-venv/bin/python' in shortcut_text
    assert "-m paperbase.launcher.cli run" in shortcut_text
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
