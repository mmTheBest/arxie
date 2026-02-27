from __future__ import annotations

import pytest

from ra.utils.config import load_config, mask_secret


def test_mask_secret_shows_only_last_4_chars() -> None:
    assert mask_secret(None) is None
    assert mask_secret("") == ""
    assert mask_secret("abcd") == "***abcd"
    assert mask_secret("sk-123456") == "***3456"


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0000")
    monkeypatch.delenv("RA_MODEL", raising=False)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("RA_LOG_DIR", raising=False)
    monkeypatch.delenv("RA_LOG_LEVEL", raising=False)

    cfg = load_config()
    assert cfg.openai_api_key == "sk-test-0000"
    assert cfg.ra_model == "gpt-4o-mini"
    assert cfg.semantic_scholar_api_key is None
    assert cfg.ra_log_dir == "data/logs"
    assert cfg.ra_log_level == "INFO"

    # Ensure repr is safe.
    rep = repr(cfg)
    assert "sk-test-0000" not in rep
    assert "***0000" in rep


def test_load_config_missing_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match=r"OPENAI_API_KEY"):
        load_config()
