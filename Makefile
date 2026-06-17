.PHONY: arxie-api arxie-worker arxie-db-upgrade paperbase-api paperbase-worker paperbase-db-upgrade paperbase-compose-config

arxie-api:
	.venv/bin/arxie-api

arxie-worker:
	.venv/bin/arxie-worker

arxie-db-upgrade:
	.venv/bin/arxie-db upgrade

paperbase-api:
	.venv/bin/paperbase-api

paperbase-worker:
	.venv/bin/paperbase-worker

paperbase-db-upgrade:
	.venv/bin/paperbase-db upgrade

paperbase-compose-config:
	docker compose -f infra/docker-compose.paperbase.yml config
