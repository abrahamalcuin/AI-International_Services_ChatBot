from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FinanceLoginRequest(BaseModel):
    pin: str = Field(..., min_length=4)


class FinanceLoginResponse(BaseModel):
    token: str


class FinanceHealthResponse(BaseModel):
    ok: bool
    databaseConfigured: bool
    pinConfigured: bool


class SnapshotSaveRequest(BaseModel):
    snapshot: dict[str, Any]


class SnapshotResponse(BaseModel):
    snapshot: dict[str, Any] | None
    revision: int
    updatedAt: str | None
