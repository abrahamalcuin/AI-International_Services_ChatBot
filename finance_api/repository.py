from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot: dict[str, Any]
    revision: int
    updated_at: str


class FinanceRepository(Protocol):
    def get_snapshot(self) -> SnapshotRecord | None:
        ...

    def save_snapshot(self, snapshot: dict[str, Any]) -> SnapshotRecord:
        ...


class PostgresFinanceRepository:
    def __init__(self, database_url: str, household_id: str, household_name: str) -> None:
        if not database_url:
            raise RuntimeError("DATABASE_URL is required for finance persistence.")

        self.database_url = database_url
        self.household_id = household_id
        self.household_name = household_name

    def get_snapshot(self) -> SnapshotRecord | None:
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            self._ensure_household(conn)
            row = conn.execute(
                """
                SELECT snapshot, revision, updated_at
                FROM finance_snapshots
                WHERE household_id = %s
                """,
                (self.household_id,),
            ).fetchone()

        if row is None:
            return None

        return SnapshotRecord(
            snapshot=dict(row["snapshot"]),
            revision=int(row["revision"]),
            updated_at=row["updated_at"].isoformat(),
        )

    def save_snapshot(self, snapshot: dict[str, Any]) -> SnapshotRecord:
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            self._ensure_household(conn)
            row = conn.execute(
                """
                INSERT INTO finance_snapshots (household_id, snapshot, revision, updated_at)
                VALUES (%s, %s, 1, now())
                ON CONFLICT (household_id) DO UPDATE SET
                    snapshot = excluded.snapshot,
                    revision = finance_snapshots.revision + 1,
                    updated_at = now()
                RETURNING snapshot, revision, updated_at
                """,
                (self.household_id, Jsonb(snapshot)),
            ).fetchone()
            conn.commit()

        if row is None:
            raise RuntimeError("Finance snapshot save failed.")

        return SnapshotRecord(
            snapshot=dict(row["snapshot"]),
            revision=int(row["revision"]),
            updated_at=row["updated_at"].isoformat(),
        )

    def _ensure_household(self, conn: psycopg.Connection) -> None:
        conn.execute(
            """
            INSERT INTO finance_households (id, name)
            VALUES (%s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (self.household_id, self.household_name),
        )
