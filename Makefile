.PHONY: test-clean-baseline paperbase-api paperbase-worker paperbase-db-upgrade paperbase-compose-config

test-clean-baseline:
	.venv/bin/python -m pytest tests/ -q --ignore=tests/integration

paperbase-api:
	.venv/bin/paperbase-api

paperbase-worker:
	.venv/bin/paperbase-worker

paperbase-db-upgrade:
	.venv/bin/paperbase-db upgrade

paperbase-compose-config:
	docker compose -f infra/docker-compose.paperbase.yml config
