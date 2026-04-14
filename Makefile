.PHONY: test-clean-baseline

test-clean-baseline:
	.venv/bin/python -m pytest tests/ -q --ignore=tests/integration
