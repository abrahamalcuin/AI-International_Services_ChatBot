from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FinanceSettings:
    pin: str
    token_secret: str
    token_max_age_seconds: int
    database_url: str
    household_id: str
    household_name: str

    @property
    def database_configured(self) -> bool:
        return bool(self.database_url)

    @property
    def pin_configured(self) -> bool:
        return bool(self.pin)


def get_finance_settings() -> FinanceSettings:
    return FinanceSettings(
        pin=os.getenv("FINANCE_PIN", "").strip(),
        token_secret=os.getenv("FINANCE_TOKEN_SECRET", os.getenv("SESSION_SECRET", "change-this-in-render")),
        token_max_age_seconds=int(os.getenv("FINANCE_TOKEN_MAX_AGE_SECONDS", "2592000")),
        database_url=os.getenv("FINANCE_DATABASE_URL", os.getenv("DATABASE_URL", "")).strip(),
        household_id=os.getenv("FINANCE_HOUSEHOLD_ID", "default-household").strip() or "default-household",
        household_name=os.getenv("FINANCE_HOUSEHOLD_NAME", "Planner").strip() or "Planner",
    )
