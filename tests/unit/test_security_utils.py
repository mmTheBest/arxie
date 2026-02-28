from __future__ import annotations

import pytest

from ra.utils.security import sanitize_identifier, sanitize_user_text, validate_public_http_url


def test_sanitize_user_text_rejects_empty_and_control_chars() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        sanitize_user_text("   ", field_name="query", max_length=10)
    with pytest.raises(ValueError, match="control characters"):
        sanitize_user_text("abc\x00def", field_name="query", max_length=20)


def test_sanitize_identifier_rejects_whitespace() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        sanitize_identifier("DOI:10.1000/xyz abc")


def test_validate_public_http_url_accepts_public_host() -> None:
    assert (
        validate_public_http_url("https://arxiv.org/pdf/2101.00001.pdf")
        == "https://arxiv.org/pdf/2101.00001.pdf"
    )


def test_validate_public_http_url_rejects_private_or_local_hosts() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        validate_public_http_url("http://localhost/internal")
    with pytest.raises(ValueError, match="not allowed"):
        validate_public_http_url("http://127.0.0.1/private")
