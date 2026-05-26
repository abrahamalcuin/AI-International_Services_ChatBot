from __future__ import annotations

import psycopg

from .config import get_finance_settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS finance_households (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS finance_snapshots (
    household_id TEXT PRIMARY KEY REFERENCES finance_households(id) ON DELETE CASCADE,
    snapshot JSONB NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def init_finance_schema() -> bool:
    settings = get_finance_settings()
    if not settings.database_configured:
        return False

    with psycopg.connect(settings.database_url) as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()

    return True
