import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

DB_PATH = Path(os.getenv("AUTH_DB_PATH", Path(__file__).resolve().parent / "auth.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_code TEXT UNIQUE,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    is_active INTEGER NOT NULL DEFAULT 1,
    desired_role TEXT NOT NULL DEFAULT 'user',
    invite_token TEXT UNIQUE,
    invite_expires_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS login (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL UNIQUE,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
);
"""


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
        ensure_column(conn, "employees", "desired_role", "TEXT NOT NULL DEFAULT 'user'")
        ensure_column(conn, "login", "role", "TEXT NOT NULL DEFAULT 'user'")
        conn.commit()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def create_employee(employee_code: str, first_name: str, last_name: str, email: str, desired_role: str = "user") -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO employees (employee_code, first_name, last_name, email, desired_role)
            VALUES (?, ?, ?, ?, ?)
            """,
            (employee_code, first_name, last_name, email, desired_role),
        )
        conn.commit()


def generate_invite(email: str, hours: int = 72) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        employee = conn.execute("SELECT id FROM employees WHERE email = ?", (email,)).fetchone()
        if employee is None:
            raise ValueError("Employee not found")
        conn.execute(
            "UPDATE employees SET invite_token = ?, invite_expires_at = ? WHERE id = ?",
            (token, expires_at, employee["id"]),
        )
        conn.commit()
    return token, expires_at


def find_employee_by_token(token: str):
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT id, email, invite_token, invite_expires_at, is_active, desired_role
            FROM employees
            WHERE invite_token = ?
            """,
            (token,),
        ).fetchone()


def employee_has_login(employee_id: int) -> bool:
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM login WHERE employee_id = ?", (employee_id,)).fetchone()
        return row is not None


def is_token_expired(expires_at: Optional[str]) -> bool:
    if not expires_at:
        return False
    expiry = datetime.strptime(expires_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) > expiry
