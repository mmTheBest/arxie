from __future__ import annotations

from fastapi.testclient import TestClient

from services.paperbase_api.app import create_app
from services.paperbase_api.health import DependencyCheckResult, ReadinessReport


class _HealthyDependencyChecker:
    def check(self) -> ReadinessReport:
        return ReadinessReport(
            ready=True,
            dependencies=[
                DependencyCheckResult(name="database", ok=True, detail="ok", required=True),
                DependencyCheckResult(name="search", ok=True, detail="ok", required=True),
                DependencyCheckResult(name="redis", ok=True, detail="ok", required=True),
                DependencyCheckResult(name="object_store", ok=True, detail="ok", required=True),
            ],
        )


class _UnhealthyDependencyChecker:
    def check(self) -> ReadinessReport:
        return ReadinessReport(
            ready=False,
            dependencies=[
                DependencyCheckResult(name="database", ok=True, detail="ok", required=True),
                DependencyCheckResult(
                    name="search",
                    ok=False,
                    detail="connection refused",
                    required=True,
                ),
            ],
        )


def test_health_and_livez_report_service_metadata() -> None:
    client = TestClient(create_app(dependency_checker=_HealthyDependencyChecker()))

    health_response = client.get("/health")
    livez_response = client.get("/livez")

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert health_response.json()["service"] == "paperbase-api"
    assert livez_response.status_code == 200
    assert livez_response.json() == health_response.json()


def test_readyz_reports_dependency_status_when_ready() -> None:
    client = TestClient(create_app(dependency_checker=_HealthyDependencyChecker()))

    response = client.get("/readyz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["service"] == "paperbase-api"
    assert {item["name"] for item in payload["dependencies"]} == {
        "database",
        "search",
        "redis",
        "object_store",
    }
    assert all(item["ok"] for item in payload["dependencies"])


def test_readyz_returns_503_when_required_dependency_is_unavailable() -> None:
    client = TestClient(create_app(dependency_checker=_UnhealthyDependencyChecker()))

    response = client.get("/readyz")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["dependencies"][1] == {
        "name": "search",
        "ok": False,
        "detail": "connection refused",
        "required": True,
    }
