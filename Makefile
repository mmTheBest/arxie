.PHONY: paperbase-api paperbase-worker paperbase-db-upgrade paperbase-compose-config

paperbase-api:
	.venv/bin/paperbase-api

paperbase-worker:
	.venv/bin/paperbase-worker

paperbase-db-upgrade:
	.venv/bin/paperbase-db upgrade

paperbase-compose-config:
	docker compose -f infra/docker-compose.paperbase.yml config
