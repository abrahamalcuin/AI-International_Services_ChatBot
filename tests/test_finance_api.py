from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from finance_api.repository import SnapshotRecord
from finance_api.router import get_repository, router


@dataclass
class MemoryFinanceRepository:
    snapshot: dict[str, Any] | None = None
    revision: int = 0
    updated_at: str | None = None

    def get_snapshot(self) -> SnapshotRecord | None:
        if self.snapshot is None:
            return None

        return SnapshotRecord(
            snapshot=self.snapshot,
            revision=self.revision,
            updated_at=self.updated_at or "2026-05-26T00:00:00Z",
        )

    def save_snapshot(self, snapshot: dict[str, Any]) -> SnapshotRecord:
        self.snapshot = snapshot
        self.revision += 1
        self.updated_at = "2026-05-26T00:00:00Z"
        return self.get_snapshot()  # type: ignore[return-value]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("FINANCE_PIN", "109428")
    monkeypatch.setenv("FINANCE_TOKEN_SECRET", "test-finance-secret")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    repository = MemoryFinanceRepository()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_repository] = lambda: repository

    return TestClient(app)


def login(client: TestClient) -> str:
    response = client.post("/finance/auth/login", json={"pin": "109428"})
    assert response.status_code == 200
    token = response.json()["token"]
    assert isinstance(token, str)
    assert token
    return token


def test_finance_login_accepts_configured_pin_and_returns_token(client: TestClient) -> None:
    response = client.post("/finance/auth/login", json={"pin": "109428"})

    assert response.status_code == 200
    assert response.json()["token"]


def test_finance_login_rejects_wrong_pin(client: TestClient) -> None:
    response = client.post("/finance/auth/login", json={"pin": "000000"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid finance PIN."


def test_snapshot_requires_bearer_token(client: TestClient) -> None:
    response = client.get("/finance/snapshot")

    assert response.status_code == 401
    assert response.json()["detail"] == "Finance authorization is required."


def test_snapshot_round_trip_persists_latest_snapshot(client: TestClient) -> None:
    token = login(client)
    snapshot = {
        "accounts": [{"id": "checking", "name": "Checking", "balance": 100}],
        "budgets": [],
        "transactions": [],
    }

    save_response = client.put(
        "/finance/snapshot",
        headers={"Authorization": f"Bearer {token}"},
        json={"snapshot": snapshot},
    )
    load_response = client.get("/finance/snapshot", headers={"Authorization": f"Bearer {token}"})

    assert save_response.status_code == 200
    assert save_response.json()["snapshot"] == snapshot
    assert save_response.json()["revision"] == 1
    assert save_response.json()["updatedAt"] == "2026-05-26T00:00:00Z"
    assert load_response.status_code == 200
    assert load_response.json() == save_response.json()


def test_finance_health_reports_database_configuration(client: TestClient) -> None:
    response = client.get("/finance/health")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "databaseConfigured": False,
        "pinConfigured": True,
    }


def test_main_app_mounts_finance_router() -> None:
    from main import app

    route_paths = {route.path for route in app.routes}

    assert "/finance/health" in route_paths
