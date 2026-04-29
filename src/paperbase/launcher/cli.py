"""Single-user local launcher for Arxie."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import urljoin

import httpx
import typer

app = typer.Typer(help="Convenience launcher for the local Arxie workspace.")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _compose_file() -> Path:
    return _repo_root() / "infra" / "docker-compose.paperbase.yml"


def _workspace_url(base_url: str) -> str:
    normalized = base_url.rstrip("/") + "/"
    return urljoin(normalized, "app")


def _run_compose(args: list[str]) -> None:
    command = ["docker", "compose", "-f", str(_compose_file()), *args]
    subprocess.run(command, cwd=_repo_root(), check=True)


def _ensure_container_runtime() -> None:
    try:
        subprocess.run(
            ["docker", "info"],
            cwd=_repo_root(),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    except subprocess.CalledProcessError as exc:
        context = subprocess.run(
            ["docker", "context", "show"],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if context == "colima" and shutil.which("colima"):
            typer.echo("Starting Colima for the active Docker context…")
            subprocess.run(["colima", "start"], cwd=_repo_root(), check=True)
            subprocess.run(
                ["docker", "info"],
                cwd=_repo_root(),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        raise RuntimeError(
            "Docker is not available. Start Docker Desktop or Colima, then retry."
        ) from exc


def _wait_until_ready(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    ready_url = urljoin(base_url.rstrip("/") + "/", "readyz")
    last_error: Exception | None = None

    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(ready_url)
                if response.status_code == 200:
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(1.0)

    if last_error is not None:
        raise RuntimeError(f"Arxie did not become ready at {ready_url}: {last_error}") from last_error
    raise RuntimeError(f"Arxie did not become ready at {ready_url} within {timeout_seconds:.0f}s.")


@app.command("run")
def run(
    no_browser: bool = typer.Option(False, "--no-browser", help="Start Arxie without opening a browser."),
    base_url: str = typer.Option("http://localhost:8080", help="Base URL for the local Arxie API/UI."),
    timeout_seconds: float = typer.Option(120.0, "--timeout", min=10.0, help="Seconds to wait for readiness."),
) -> None:
    """Boot the local Docker stack and open the Arxie workspace."""

    _ensure_container_runtime()
    _run_compose(["up", "-d", "postgres", "elasticsearch", "minio", "redis"])
    _run_compose(["run", "--rm", "paperbase-migrate"])
    _run_compose(["up", "-d", "paperbase-api", "paperbase-worker"])
    _wait_until_ready(base_url, timeout_seconds)

    workspace_url = _workspace_url(base_url)
    if not no_browser:
        webbrowser.open(workspace_url)
    typer.echo(f"Arxie is ready at {workspace_url}")


@app.command("open")
def open_app(
    base_url: str = typer.Option("http://localhost:8080", help="Base URL for the local Arxie API/UI."),
) -> None:
    """Open the browser workspace without restarting services."""

    workspace_url = _workspace_url(base_url)
    webbrowser.open(workspace_url)
    typer.echo(f"Opened {workspace_url}")


@app.command("down")
def down() -> None:
    """Stop the local Docker stack."""

    _run_compose(["down"])
    typer.echo("Stopped the local Arxie stack.")


@app.command("install-shortcut")
def install_shortcut(
    output: Path = typer.Option(
        Path.home() / "Desktop" / "Arxie.command",
        "--output",
        help="Where to write the double-clickable launcher script.",
    ),
    python_path: str = typer.Option(
        sys.executable,
        "--python",
        help="Python executable to embed in the shortcut script.",
    ),
) -> None:
    """Create a double-clickable .command launcher for the local workspace."""

    resolved_output = output.expanduser()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    repo_root = _repo_root()
    launcher_path = str(Path(python_path).expanduser().with_name("arxie-local"))
    script = "\n".join(
        [
            "#!/bin/zsh",
            f'cd "{repo_root}"',
            f'exec "{launcher_path}" run',
            "",
        ]
    )
    resolved_output.write_text(script, encoding="utf-8")
    current_mode = resolved_output.stat().st_mode
    resolved_output.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    typer.echo(f"Installed launcher shortcut at {resolved_output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
