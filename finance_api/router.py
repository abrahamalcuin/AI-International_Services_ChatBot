from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status

from .auth import create_access_token, verify_access_token, verify_pin
from .config import FinanceSettings, get_finance_settings
from .models import (
    FinanceHealthResponse,
    FinanceLoginRequest,
    FinanceLoginResponse,
    SnapshotResponse,
    SnapshotSaveRequest,
)
from .repository import FinanceRepository, PostgresFinanceRepository

router = APIRouter(prefix="/finance", tags=["finance"])


def get_settings() -> FinanceSettings:
    return get_finance_settings()


def get_repository(settings: FinanceSettings = Depends(get_settings)) -> FinanceRepository:
    return PostgresFinanceRepository(
        database_url=settings.database_url,
        household_id=settings.household_id,
        household_name=settings.household_name,
    )


def require_finance_auth(
    authorization: str | None = Header(default=None),
    settings: FinanceSettings = Depends(get_settings),
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Finance authorization is required.",
        )

    payload = verify_access_token(authorization.removeprefix("Bearer ").strip(), settings)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Finance authorization is invalid or expired.",
        )

    return payload


@router.get("/health", response_model=FinanceHealthResponse)
async def finance_health(settings: FinanceSettings = Depends(get_settings)) -> FinanceHealthResponse:
    return FinanceHealthResponse(
        ok=True,
        databaseConfigured=settings.database_configured,
        pinConfigured=settings.pin_configured,
    )


@router.post("/auth/login", response_model=FinanceLoginResponse)
async def finance_login(
    payload: FinanceLoginRequest,
    settings: FinanceSettings = Depends(get_settings),
) -> FinanceLoginResponse:
    if not settings.pin_configured:
        raise HTTPException(status_code=503, detail="Finance PIN is not configured.")

    if not verify_pin(payload.pin, settings):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid finance PIN.")

    return FinanceLoginResponse(token=create_access_token(settings))


@router.get("/snapshot", response_model=SnapshotResponse)
async def get_finance_snapshot(
    _payload: dict = Depends(require_finance_auth),
    repository: FinanceRepository = Depends(get_repository),
) -> SnapshotResponse:
    record = repository.get_snapshot()
    if record is None:
        return SnapshotResponse(snapshot=None, revision=0, updatedAt=None)

    return SnapshotResponse(
        snapshot=record.snapshot,
        revision=record.revision,
        updatedAt=record.updated_at,
    )


@router.put("/snapshot", response_model=SnapshotResponse)
async def save_finance_snapshot(
    payload: SnapshotSaveRequest,
    _token_payload: dict = Depends(require_finance_auth),
    repository: FinanceRepository = Depends(get_repository),
) -> SnapshotResponse:
    record = repository.save_snapshot(payload.snapshot)
    return SnapshotResponse(
        snapshot=record.snapshot,
        revision=record.revision,
        updatedAt=record.updated_at,
    )
