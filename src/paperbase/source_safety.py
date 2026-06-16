"""Secret-aware safety checks for user-provided Study source content."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceSecretFinding:
    """A non-sensitive description of a secret-like source content match."""

    kind: str


_SECRET_VALUE_MIN_LENGTH = 12
_AUTHORIZATION_TOKEN_MIN_LENGTH = 20
_BASIC_AUTH_TOKEN_MIN_LENGTH = 8
_SECRET_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    \b
    (?P<name>
        [A-Z0-9_]* (?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL) [A-Z0-9_]*
    )
    \b
    \s*[:=]\s*
    (?P<quote>['"]?)
    (?P<value>[^\s'",;#]+)
    (?P=quote)
    """
)
_AUTHORIZATION_CREDENTIAL_RE = re.compile(
    rf"""(?ix)
    ["']? \b authorization \b ["']? \s* : \s* ["']?
    (?P<scheme>bearer|basic) \s+
    (?P<value>[A-Za-z0-9._~+/=-]{{{_BASIC_AUTH_TOKEN_MIN_LENGTH},}})
    """
)
_HTTP_SECRET_HEADER_RE = re.compile(
    rf"""(?ix)
    ["']? \b
    (?P<name>
        x[-_]?api[-_]?key
        | api[-_]?key
        | x[-_]?auth[-_]?token
    )
    \b ["']? \s* : \s* ["']?
    (?:(?:bearer|basic) \s+)?
    (?P<value>[A-Za-z0-9._~+/=-]{{{_SECRET_VALUE_MIN_LENGTH},}})
    """
)
_JWT_RE = re.compile(
    r"""(?x)
    \b
    eyJ[A-Za-z0-9_-]{8,}
    \.
    eyJ[A-Za-z0-9_-]{8,}
    \.
    [A-Za-z0-9_-]{16,}
    \b
    """
)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
_ANTHROPIC_KEY_RE = re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")
_GOOGLE_API_KEY_RE = re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b")
_GITHUB_TOKEN_RE = re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b")
_GITLAB_TOKEN_RE = re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")
_HUGGINGFACE_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")
_SLACK_TOKEN_RE = re.compile(r"\b(?:xox[abprs]|xapp)-[A-Za-z0-9-]{20,}\b")
_AWS_ACCESS_KEY_RE = re.compile(r"\bA(?:KIA|SIA)[A-Z0-9]{16}\b")
_URL_CREDENTIAL_RE = re.compile(
    r"""(?ix)
    \b
    (?:
        postgres(?:ql)?(?:\+[a-z0-9_]+)?
        |mysql(?:\+[a-z0-9_]+)?
        |mariadb(?:\+[a-z0-9_]+)?
        |mongodb(?:\+srv)?
        |redis
        |rediss
        |amqp
        |amqps
    )
    ://
    [^/\s:@]*
    :
    [^@\s/]*
    @
    """
)
_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",
    re.IGNORECASE,
)


def detect_source_secret(text: str | None) -> SourceSecretFinding | None:
    """Return a non-sensitive finding when source text contains obvious secrets."""

    if not text:
        return None
    assignment_match = _SECRET_ASSIGNMENT_RE.search(text)
    if assignment_match is not None:
        value = assignment_match.group("value").strip()
        if len(value) >= _SECRET_VALUE_MIN_LENGTH:
            return SourceSecretFinding(kind="secret_assignment")
    if _JWT_RE.search(text):
        return SourceSecretFinding(kind="jwt")
    for authorization_match in _AUTHORIZATION_CREDENTIAL_RE.finditer(text):
        scheme = authorization_match.group("scheme").lower()
        value = authorization_match.group("value")
        if scheme == "basic" or len(value) >= _AUTHORIZATION_TOKEN_MIN_LENGTH:
            return SourceSecretFinding(kind=f"authorization_{scheme}")
    if _HTTP_SECRET_HEADER_RE.search(text):
        return SourceSecretFinding(kind="http_secret_header")
    if _ANTHROPIC_KEY_RE.search(text):
        return SourceSecretFinding(kind="anthropic_api_key")
    if _OPENAI_KEY_RE.search(text):
        return SourceSecretFinding(kind="openai_api_key")
    if _GOOGLE_API_KEY_RE.search(text):
        return SourceSecretFinding(kind="google_api_key")
    if _GITHUB_TOKEN_RE.search(text):
        return SourceSecretFinding(kind="github_token")
    if _GITLAB_TOKEN_RE.search(text):
        return SourceSecretFinding(kind="gitlab_token")
    if _HUGGINGFACE_TOKEN_RE.search(text):
        return SourceSecretFinding(kind="huggingface_token")
    if _SLACK_TOKEN_RE.search(text):
        return SourceSecretFinding(kind="slack_token")
    if _AWS_ACCESS_KEY_RE.search(text):
        return SourceSecretFinding(kind="aws_access_key_id")
    if _URL_CREDENTIAL_RE.search(text):
        return SourceSecretFinding(kind="url_credentials")
    if _PRIVATE_KEY_BLOCK_RE.search(text):
        return SourceSecretFinding(kind="private_key")
    return None
