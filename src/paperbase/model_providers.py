"""Model-provider helpers for Paperbase runtime clients."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ModelProvider = str
Runner = Callable[..., Any]
EXPECTED_WORKER_MODEL_PROVIDER_ENV = "PAPERBASE_EXPECTED_WORKER_MODEL_PROVIDER"
SUPPORTED_MODEL_PROVIDERS = {"openai", "codex_cli", "claude_cli", "none"}

_SUBPROCESS_COMMON_ENV_ALLOWLIST = {
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "LC_ALL",
    "LOGNAME",
    "NO_COLOR",
    "NO_PROXY",
    "PATH",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TERM",
    "TMPDIR",
    "USER",
}

_SUBPROCESS_PROVIDER_ENV_ALLOWLIST = {
    "claude_cli": {"CLAUDE_CODE_OAUTH_TOKEN"},
    "codex_cli": {"CODEX_HOME"},
}

_PROVIDER_COMMAND_ENV = {
    "claude_cli": "PAPERBASE_CLAUDE_COMMAND",
    "codex_cli": "PAPERBASE_CODEX_COMMAND",
}
_CLI_LOGIN_STATUS_TIMEOUT_SECONDS = 5.0
_CLI_LOGIN_STATUS_ENV_ALLOWLIST = _SUBPROCESS_COMMON_ENV_ALLOWLIST


class SubscriptionCLIModelError(RuntimeError):
    """Raised when a subscription-backed local model CLI fails."""


@dataclass(frozen=True, slots=True)
class ModelProviderConfig:
    provider: ModelProvider
    model_name: str | None
    openai_api_key: str | None = field(repr=False)
    subprocess_env: dict[str, str] = field(repr=False)
    cli_timeout_seconds: float
    codex_command: str
    claude_command: str
    allow_agentic_cli: bool


@dataclass(frozen=True, slots=True)
class ModelProviderStatus:
    provider: ModelProvider
    model_name: str | None
    configured: bool
    usable: bool
    missing_setup: list[str]
    setup_hints: list[str]
    warnings: list[str]
    command: str | None = None
    command_available: bool | None = None
    allow_agentic_cli: bool = False
    login_status: str = "not_applicable"
    login_status_checked: bool = False
    login_status_command: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerModelProviderStatus:
    provider: ModelProvider | None
    matches_api_provider: bool | None
    source: str
    setup_hints: list[str]
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class _CLILoginProbeResult:
    login_status: str
    checked: bool
    setup_hints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _first_command_part(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        return ""
    return parts[0] if parts else ""


def _safe_command_label(command_name: str) -> str:
    if not command_name:
        return ""
    normalized = command_name.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def load_model_provider_config(env: Mapping[str, str] | None = None) -> ModelProviderConfig:
    """Load model-provider settings for Paperbase runtime clients.

    The default remains the historical OpenAI API path. Subscription-backed CLI
    providers are opt-in so existing deployments do not start local model agents
    just because a CLI happens to be installed.
    """

    resolved_env = os.environ if env is None else env
    provider = (resolved_env.get("PAPERBASE_MODEL_PROVIDER") or "openai").strip().lower()
    if provider in {"off", "disabled"}:
        provider = "none"
    if provider not in SUPPORTED_MODEL_PROVIDERS:
        raise ValueError(
            "PAPERBASE_MODEL_PROVIDER must be one of: openai, codex_cli, claude_cli, none."
        )

    paperbase_model = (resolved_env.get("PAPERBASE_MODEL_NAME") or "").strip()
    if provider == "openai":
        model_name = paperbase_model or (resolved_env.get("RA_MODEL") or "gpt-4o-mini").strip()
    else:
        model_name = paperbase_model or None

    allow_agentic_raw = (
        resolved_env.get("PAPERBASE_ALLOW_AGENTIC_CLI") or "false"
    ).strip().lower()
    subprocess_env_allowlist = (
        _SUBPROCESS_COMMON_ENV_ALLOWLIST
        | _SUBPROCESS_PROVIDER_ENV_ALLOWLIST.get(provider, set())
    )
    return ModelProviderConfig(
        provider=provider,
        model_name=model_name or None,
        openai_api_key=(resolved_env.get("OPENAI_API_KEY") or "").strip() or None,
        subprocess_env={
            key: value
            for key, value in resolved_env.items()
            if key in subprocess_env_allowlist and value
        },
        cli_timeout_seconds=float(
            (resolved_env.get("PAPERBASE_MODEL_CLI_TIMEOUT_SECONDS") or "240").strip()
        ),
        codex_command=(resolved_env.get("PAPERBASE_CODEX_COMMAND") or "codex").strip(),
        claude_command=(resolved_env.get("PAPERBASE_CLAUDE_COMMAND") or "claude").strip(),
        allow_agentic_cli=allow_agentic_raw in {"1", "true", "yes", "on"},
    )


def _normalize_provider_name(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"off", "disabled"}:
        return "none"
    return normalized


def describe_model_provider_status(
    env: Mapping[str, str] | None = None,
    *,
    command_resolver: Callable[[str], str | None] = shutil.which,
    login_status_runner: Runner | None = None,
) -> ModelProviderStatus:
    """Return secret-safe model-provider readiness for UI/runtime diagnostics."""

    resolved_env = os.environ if env is None else env
    config = load_model_provider_config(resolved_env)
    if config.provider == "none":
        return ModelProviderStatus(
            provider=config.provider,
            model_name=config.model_name,
            configured=True,
            usable=False,
            missing_setup=[],
            setup_hints=["Model-backed Paperbase synthesis is disabled."],
            warnings=["PAPERBASE_MODEL_PROVIDER=none disables model-backed features."],
            allow_agentic_cli=config.allow_agentic_cli,
            login_status="not_applicable",
        )

    if config.provider == "openai":
        missing_setup = [] if config.openai_api_key else ["OPENAI_API_KEY"]
        configured = not missing_setup
        return ModelProviderStatus(
            provider=config.provider,
            model_name=config.model_name,
            configured=configured,
            usable=configured,
            missing_setup=missing_setup,
            setup_hints=[] if configured else ["Set OPENAI_API_KEY for OpenAI-backed synthesis."],
            warnings=[],
            allow_agentic_cli=config.allow_agentic_cli,
            login_status="not_applicable",
        )

    command = config.codex_command if config.provider == "codex_cli" else config.claude_command
    command_name = _first_command_part(command)
    command_label = _safe_command_label(command_name)
    command_available = bool(command_name and command_resolver(command_name))
    missing_setup: list[str] = []
    setup_hints: list[str] = []
    warnings: list[str] = []
    login_status = "not_checked"
    login_status_checked = False
    login_status_command = (
        "codex login status" if config.provider == "codex_cli" else "claude setup-token"
    )

    command_env = _PROVIDER_COMMAND_ENV[config.provider]
    if not command_available:
        missing_setup.append(command_env)
        if command_label:
            setup_hints.append(f"Install or configure the `{command_label}` command.")
        else:
            setup_hints.append(f"Install or configure {command_env}.")

    if config.provider == "claude_cli" and "CLAUDE_CODE_OAUTH_TOKEN" not in config.subprocess_env:
        missing_setup.append("CLAUDE_CODE_OAUTH_TOKEN")
        login_status = "logged_out"
        setup_hints.append(
            "Run Claude setup-token and restart Arxie with CLAUDE_CODE_OAUTH_TOKEN set."
        )
    elif config.provider == "claude_cli":
        login_status = "token_configured"

    if config.provider == "codex_cli" and not config.allow_agentic_cli:
        missing_setup.append("PAPERBASE_ALLOW_AGENTIC_CLI")
        setup_hints.append(
            "Set PAPERBASE_ALLOW_AGENTIC_CLI=true only for trusted local corpora."
        )
        warnings.append("codex_cli is an agentic local runtime and is disabled by default.")
    elif config.provider == "codex_cli" and "CODEX_HOME" not in config.subprocess_env:
        missing_setup.append("CODEX_HOME")
        setup_hints.append(
            "Set CODEX_HOME to the Codex config directory Arxie should use, "
            "then run codex login in that environment."
        )

    configured = not missing_setup
    if config.provider == "codex_cli" and command_available and configured:
        probe = _probe_codex_login_status(
            command=command,
            env=resolved_env,
            runner=login_status_runner or subprocess.run,
        )
        login_status = probe.login_status
        login_status_checked = probe.checked
        setup_hints.extend(probe.setup_hints)
        warnings.extend(probe.warnings)
        if probe.login_status == "logged_out":
            missing_setup.append("codex login")
            configured = False

    return ModelProviderStatus(
        provider=config.provider,
        model_name=config.model_name,
        configured=configured,
        usable=configured,
        missing_setup=missing_setup,
        setup_hints=setup_hints,
        warnings=warnings,
        command=command_label or None,
        command_available=command_available,
        allow_agentic_cli=config.allow_agentic_cli,
        login_status=login_status,
        login_status_checked=login_status_checked,
        login_status_command=login_status_command,
    )


def _probe_codex_login_status(
    *,
    command: str,
    env: Mapping[str, str],
    runner: Runner,
) -> _CLILoginProbeResult:
    try:
        args = [*shlex.split(command), "login", "status"]
    except ValueError:
        return _CLILoginProbeResult(
            login_status="unknown",
            checked=False,
            warnings=["Codex login status could not be checked because the command is invalid."],
        )

    if not args:
        return _CLILoginProbeResult(
            login_status="unknown",
            checked=False,
            warnings=["Codex login status could not be checked because no command is configured."],
        )

    try:
        with tempfile.TemporaryDirectory(prefix="arxie-codex-status-") as temp_dir:
            safe_env = _login_status_env(env=env, provider="codex_cli")
            safe_env["HOME"] = temp_dir
            safe_env["XDG_CONFIG_HOME"] = str(Path(temp_dir) / ".config")
            safe_env["XDG_CACHE_HOME"] = str(Path(temp_dir) / ".cache")
            safe_env["XDG_STATE_HOME"] = str(Path(temp_dir) / ".state")
            result = runner(
                args,
                cwd=temp_dir,
                env=safe_env,
                text=True,
                capture_output=True,
                timeout=_CLI_LOGIN_STATUS_TIMEOUT_SECONDS,
                check=False,
                shell=False,
            )
    except subprocess.TimeoutExpired:
        return _CLILoginProbeResult(
            login_status="unknown",
            checked=True,
            warnings=[
                "Codex login status check timed out; run codex login status locally."
            ],
        )
    except OSError:
        return _CLILoginProbeResult(
            login_status="unknown",
            checked=True,
            warnings=[
                "Codex login status could not be checked; run codex login status locally."
            ],
        )

    returncode = int(getattr(result, "returncode", 1) or 0)
    if returncode == 0:
        return _CLILoginProbeResult(login_status="logged_in", checked=True)
    return _CLILoginProbeResult(
        login_status="logged_out",
        checked=True,
        setup_hints=[
            "Run codex login, confirm with codex login status, then restart Arxie or retry."
        ],
    )


def _login_status_env(*, env: Mapping[str, str], provider: ModelProvider) -> dict[str, str]:
    allowlist = (
        _CLI_LOGIN_STATUS_ENV_ALLOWLIST
        | _SUBPROCESS_PROVIDER_ENV_ALLOWLIST.get(provider, set())
    )
    safe_env = {key: value for key, value in env.items() if key in allowlist and value}
    safe_env.setdefault("PATH", os.defpath)
    return safe_env


def describe_worker_model_provider_status(
    api_provider: str,
    env: Mapping[str, str] | None = None,
) -> WorkerModelProviderStatus:
    """Return secret-safe expected worker/API model-provider agreement.

    The API process cannot prove a separate worker's live environment without a
    heartbeat protocol. This status therefore reports an explicit deployment
    expectation when configured, or an inferred shared-environment agreement
    for local and Compose deployments that start API and worker with the same
    Paperbase model-provider variables.
    """

    resolved_env = os.environ if env is None else env
    expected_provider_raw = (resolved_env.get(EXPECTED_WORKER_MODEL_PROVIDER_ENV) or "").strip()
    if expected_provider_raw:
        expected_provider = _normalize_provider_name(expected_provider_raw)
        if expected_provider not in SUPPORTED_MODEL_PROVIDERS:
            return WorkerModelProviderStatus(
                provider=None,
                matches_api_provider=False,
                source=EXPECTED_WORKER_MODEL_PROVIDER_ENV,
                setup_hints=[
                    (
                        f"Set {EXPECTED_WORKER_MODEL_PROVIDER_ENV} to one of: "
                        "openai, claude_cli, codex_cli, none."
                    )
                ],
                warnings=[
                    (
                        f"{EXPECTED_WORKER_MODEL_PROVIDER_ENV} "
                        "is not a supported Paperbase model provider."
                    )
                ],
            )
        matches_api_provider = expected_provider == api_provider
        warnings = (
            []
            if matches_api_provider
            else [
                (
                    f"Expected worker provider {expected_provider} does not match "
                    f"API provider {api_provider}."
                )
            ]
        )
        setup_hints = (
            []
            if matches_api_provider
            else [
                (
                    "Start the API and paperbase-worker with matching "
                    "PAPERBASE_MODEL_PROVIDER settings, or update the expected "
                    "worker provider diagnostic after changing worker config."
                )
            ]
        )
        return WorkerModelProviderStatus(
            provider=expected_provider,
            matches_api_provider=matches_api_provider,
            source=EXPECTED_WORKER_MODEL_PROVIDER_ENV,
            setup_hints=setup_hints,
            warnings=warnings,
        )

    return WorkerModelProviderStatus(
        provider=api_provider,
        matches_api_provider=True,
        source="shared PAPERBASE_MODEL_PROVIDER environment",
        setup_hints=[
            (
                f"Set {EXPECTED_WORKER_MODEL_PROVIDER_ENV} when API and worker "
                "are launched with separate model-provider environments."
            )
        ],
        warnings=[
            (
                "Worker/API provider agreement is inferred from shared startup "
                "configuration; this is not a live worker heartbeat."
            )
        ],
    )


def build_subscription_cli_json_client(
    config: ModelProviderConfig,
    *,
    runner: Runner = subprocess.run,
) -> SubscriptionCLIJsonClient:
    if config.provider == "codex_cli":
        command = config.codex_command
    elif config.provider == "claude_cli":
        command = config.claude_command
    else:
        raise ValueError("Subscription CLI clients require codex_cli or claude_cli provider.")
    return SubscriptionCLIJsonClient(
        provider=config.provider,
        command=command,
        model=config.model_name,
        timeout_seconds=config.cli_timeout_seconds,
        runner=runner,
        allow_agentic_tools=config.allow_agentic_cli,
        base_env=config.subprocess_env,
    )


class SubscriptionCLIJsonClient:
    """JSON completion client backed by a local authenticated AI subscription CLI."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        command: str,
        model: str | None,
        timeout_seconds: float,
        runner: Runner = subprocess.run,
        temp_dir: str | Path | None = None,
        allow_agentic_tools: bool = False,
        base_env: Mapping[str, str] | None = None,
    ) -> None:
        if provider not in {"codex_cli", "claude_cli"}:
            raise ValueError("provider must be codex_cli or claude_cli.")
        self.provider = provider
        self.command = command
        self.model = model.strip() if isinstance(model, str) and model.strip() else None
        self.timeout_seconds = timeout_seconds
        self.runner = runner
        self.temp_dir = Path(temp_dir) if temp_dir is not None else None
        self.allow_agentic_tools = allow_agentic_tools
        self.base_env = base_env
        self.model_name = f"{provider}:{self.model or 'default'}"

    def complete_json(self, *, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_prompt(system_prompt=system_prompt, user_payload=user_payload)
        if self.provider == "codex_cli":
            raw_output = self._run_codex(prompt)
        else:
            raw_output = self._run_claude(system_prompt=system_prompt, prompt=prompt)
        return self._parse_json_payload(raw_output)

    def _run_codex(self, prompt: str) -> str:
        if not self.allow_agentic_tools:
            raise SubscriptionCLIModelError(
                "codex_cli uses an agentic CLI runtime and requires "
                "PAPERBASE_ALLOW_AGENTIC_CLI=true on trusted local corpora."
            )
        output_path = self._make_temp_output_path()
        args = [
            *self._command_parts(),
            "exec",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--output-last-message",
            str(output_path),
        ]
        try:
            with self._isolated_cwd() as isolated_cwd:
                args.extend(["--cd", str(isolated_cwd)])
                if self.model:
                    args.extend(["--model", self.model])
                args.append("-")
                result = self._run_process(args, input_text=prompt, cwd=isolated_cwd)
                self._raise_for_failed_process(result)
                file_output = output_path.read_text(encoding="utf-8").strip()
                return file_output or str(getattr(result, "stdout", ""))
        finally:
            output_path.unlink(missing_ok=True)

    def _run_claude(self, *, system_prompt: str, prompt: str) -> str:
        args = [
            *self._command_parts(),
            "-p",
            "--output-format",
            "json",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--setting-sources",
            "",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "",
            "--system-prompt",
            system_prompt,
        ]
        if self.model:
            args.extend(["--model", self.model])
        with self._isolated_cwd() as isolated_cwd:
            result = self._run_process(args, input_text=prompt, cwd=isolated_cwd)
        self._raise_for_failed_process(result)
        return str(getattr(result, "stdout", ""))

    def _run_process(self, args: list[str], *, input_text: str, cwd: Path) -> Any:
        try:
            return self.runner(
                args,
                cwd=str(cwd),
                env=self._safe_subprocess_env(cwd),
                input=input_text,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SubscriptionCLIModelError(
                f"{self.provider} command timed out after {self.timeout_seconds:g} seconds."
            ) from exc
        except OSError as exc:
            raise SubscriptionCLIModelError(
                f"{self.provider} command could not be started. Check command configuration."
            ) from exc

    def _raise_for_failed_process(self, result: Any) -> None:
        returncode = int(getattr(result, "returncode", 0) or 0)
        if returncode == 0:
            return
        raise SubscriptionCLIModelError(
            f"{self.provider} command failed with exit code {returncode}."
        )

    def _safe_subprocess_env(self, cwd: Path) -> dict[str, str]:
        allowlist = (
            _SUBPROCESS_COMMON_ENV_ALLOWLIST
            | _SUBPROCESS_PROVIDER_ENV_ALLOWLIST[self.provider]
        )
        source_env = self.base_env if self.base_env is not None else os.environ
        env = {
            key: value
            for key, value in source_env.items()
            if key in allowlist and value
        }
        env.setdefault("PATH", os.defpath)
        env["HOME"] = str(cwd)
        env["XDG_CONFIG_HOME"] = str(cwd / ".config")
        env["XDG_CACHE_HOME"] = str(cwd / ".cache")
        env["XDG_STATE_HOME"] = str(cwd / ".state")
        return env

    @contextmanager
    def _isolated_cwd(self) -> Iterator[Path]:
        with tempfile.TemporaryDirectory(
            prefix="arxie-model-work-",
            dir=self.temp_dir,
        ) as temp_dir:
            yield Path(temp_dir)

    def _command_parts(self) -> list[str]:
        parts = shlex.split(self.command)
        if not parts:
            raise SubscriptionCLIModelError(f"No command configured for {self.provider}.")
        return parts

    def _make_temp_output_path(self) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix="arxie-codex-",
            dir=self.temp_dir,
            delete=False,
        )
        handle.close()
        return Path(handle.name)

    def _build_prompt(self, *, system_prompt: str, user_payload: dict[str, Any]) -> str:
        return json.dumps(
            {
                "system": system_prompt,
                "response_contract": (
                    "Return exactly one valid JSON object. Do not include markdown, prose, "
                    "tool calls, file edits, or hidden reasoning."
                ),
                "payload": user_payload,
            },
            ensure_ascii=True,
        )

    def _parse_json_payload(self, raw_output: str) -> dict[str, Any]:
        content = self._strip_fenced_json(raw_output.strip())
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise SubscriptionCLIModelError(
                f"{self.provider} did not return valid JSON."
            ) from exc
        if isinstance(parsed, dict) and isinstance(parsed.get("result"), str):
            try:
                return self._parse_json_payload(parsed["result"])
            except SubscriptionCLIModelError:
                if _looks_like_cli_envelope(parsed):
                    raise
        if not isinstance(parsed, dict):
            raise SubscriptionCLIModelError(f"{self.provider} JSON payload must be an object.")
        return parsed

    def _strip_fenced_json(self, content: str) -> str:
        if not content.startswith("```"):
            return content
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()


def _looks_like_cli_envelope(payload: dict[str, Any]) -> bool:
    envelope_keys = {
        "type",
        "subtype",
        "session_id",
        "duration_ms",
        "duration_api_ms",
        "is_error",
        "num_turns",
        "total_cost_usd",
        "usage",
    }
    return bool(envelope_keys.intersection(payload.keys()))
