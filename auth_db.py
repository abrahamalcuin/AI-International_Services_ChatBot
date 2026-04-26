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

CREATE TABLE IF NOT EXISTS seo_clients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT NOT NULL,
    industry TEXT,
    credentials TEXT,
    contact_person TEXT,
    email TEXT,
    phone TEXT,
    website_url TEXT,
    notes TEXT,
    start_date TEXT,
    due_date TEXT,
    service_tier TEXT NOT NULL CHECK (service_tier IN ('Tier 1', 'Tier 2', 'Tier 3')),
    created_by_employee_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by_employee_id) REFERENCES employees(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS seo_task_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_tier TEXT NOT NULL CHECK (service_tier IN ('Tier 1', 'Tier 2', 'Tier 3')),
    category TEXT NOT NULL CHECK (category IN ('keyword research', 'on-page', 'off-page', 'technical', 'extras')),
    task_name TEXT NOT NULL,
    task_description TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(service_tier, task_name)
);

CREATE TABLE IF NOT EXISTS seo_client_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id INTEGER NOT NULL,
    template_id INTEGER,
    category TEXT NOT NULL CHECK (category IN ('keyword research', 'on-page', 'off-page', 'technical', 'extras')),
    task_name TEXT NOT NULL,
    task_description TEXT,
    status TEXT NOT NULL DEFAULT 'Not Started' CHECK (status IN ('Not Started', 'In Progress', 'Done', 'Blocked')),
    sort_order INTEGER NOT NULL DEFAULT 0,
    assigned_to_employee_id INTEGER,
    due_date TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (client_id) REFERENCES seo_clients(id) ON DELETE CASCADE,
    FOREIGN KEY (template_id) REFERENCES seo_task_templates(id) ON DELETE SET NULL,
    FOREIGN KEY (assigned_to_employee_id) REFERENCES employees(id) ON DELETE SET NULL,
    UNIQUE(client_id, task_name)
);

CREATE INDEX IF NOT EXISTS idx_seo_clients_tier_due_date ON seo_clients(service_tier, due_date);
CREATE INDEX IF NOT EXISTS idx_seo_clients_company_name ON seo_clients(company_name);
CREATE INDEX IF NOT EXISTS idx_seo_templates_tier_category_sort ON seo_task_templates(service_tier, category, sort_order);
CREATE INDEX IF NOT EXISTS idx_seo_client_tasks_client_status ON seo_client_tasks(client_id, status);
CREATE INDEX IF NOT EXISTS idx_seo_client_tasks_category_status ON seo_client_tasks(category, status);
"""

SEO_TEMPLATE_DATA = {
    "Tier 1": [
        ("keyword research", "Baseline keyword set", "Identify primary commercial and branded keywords.", 10),
        ("keyword research", "Competitor keyword gap review", "Review competing sites for missed opportunities.", 20),
        ("on-page", "Homepage title and meta updates", "Refresh key metadata for core landing pages.", 30),
        ("on-page", "Primary service page optimization", "Update headings, copy targets, and internal links.", 40),
        ("technical", "SEO crawl health check", "Review indexing, broken links, and core crawl blockers.", 50),
        ("extras", "Monthly summary report", "Deliver concise client-ready report and next steps.", 60),
    ],
    "Tier 2": [
        ("keyword research", "Baseline keyword set", "Identify primary commercial and branded keywords.", 10),
        ("keyword research", "Competitor keyword gap review", "Review competing sites for missed opportunities.", 20),
        ("on-page", "Homepage title and meta updates", "Refresh key metadata for core landing pages.", 30),
        ("on-page", "Primary service page optimization", "Update headings, copy targets, and internal links.", 40),
        ("on-page", "Content brief recommendations", "Recommend supporting content topics for ranking growth.", 50),
        ("off-page", "Local citation and profile audit", "Review citation consistency and profile completeness.", 60),
        ("technical", "SEO crawl health check", "Review indexing, broken links, and core crawl blockers.", 70),
        ("technical", "Schema and structured data review", "Validate important structured data coverage.", 80),
        ("extras", "Monthly summary report", "Deliver concise client-ready report and next steps.", 90),
    ],
    "Tier 3": [
        ("keyword research", "Baseline keyword set", "Identify primary commercial and branded keywords.", 10),
        ("keyword research", "Competitor keyword gap review", "Review competing sites for missed opportunities.", 20),
        ("keyword research", "Content cluster expansion plan", "Map supporting keyword clusters and page intent.", 30),
        ("on-page", "Homepage title and meta updates", "Refresh key metadata for core landing pages.", 40),
        ("on-page", "Primary service page optimization", "Update headings, copy targets, and internal links.", 50),
        ("on-page", "Content brief recommendations", "Recommend supporting content topics for ranking growth.", 60),
        ("off-page", "Local citation and profile audit", "Review citation consistency and profile completeness.", 70),
        ("off-page", "Backlink opportunity review", "Identify practical outreach and link acquisition targets.", 80),
        ("technical", "SEO crawl health check", "Review indexing, broken links, and core crawl blockers.", 90),
        ("technical", "Schema and structured data review", "Validate important structured data coverage.", 100),
        ("technical", "Page speed and CWV review", "Prioritize performance and Core Web Vitals improvements.", 110),
        ("extras", "Monthly summary report", "Deliver concise client-ready report and next steps.", 120),
        ("extras", "Quarterly roadmap refresh", "Refresh priorities based on performance and business goals.", 130),
    ],
}


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def seed_seo_task_templates(conn: sqlite3.Connection) -> None:
    existing = conn.execute("SELECT COUNT(*) FROM seo_task_templates").fetchone()[0]
    if existing:
        return
    for tier, tasks in SEO_TEMPLATE_DATA.items():
        for category, task_name, task_description, sort_order in tasks:
            conn.execute(
                """
                INSERT OR IGNORE INTO seo_task_templates
                (service_tier, category, task_name, task_description, sort_order)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tier, category, task_name, task_description, sort_order),
            )


def _upsert_bootstrap_user(conn: sqlite3.Connection, pwd_ctx, code: str, email: str, username: str, password: str, first: str, last: str) -> None:
    existing_emp = conn.execute("SELECT id FROM employees WHERE email = ?", (email,)).fetchone()
    if existing_emp:
        employee_id = existing_emp[0]
    else:
        cursor = conn.execute(
            "INSERT INTO employees (employee_code, first_name, last_name, email, desired_role) VALUES (?, ?, ?, ?, ?)",
            (code, first, last, email, "admin"),
        )
        employee_id = cursor.lastrowid
    existing_login = conn.execute("SELECT id FROM login WHERE employee_id = ?", (employee_id,)).fetchone()
    if not existing_login:
        conn.execute(
            "INSERT INTO login (employee_id, username, password_hash, role) VALUES (?, ?, ?, ?)",
            (employee_id, username, pwd_ctx.hash(password), "admin"),
        )


def seed_bootstrap_admin(conn: sqlite3.Connection) -> None:
    """Upsert admin accounts from BOOTSTRAP_ADMIN_* env vars on every startup."""
    from passlib.context import CryptContext
    pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

    users = [
        ("BOOTSTRAP_ADMIN_EMAIL", "BOOTSTRAP_ADMIN_USERNAME", "BOOTSTRAP_ADMIN_PASSWORD", "ADMIN001", "Admin", "User"),
        ("BOOTSTRAP_ADMIN2_EMAIL", "BOOTSTRAP_ADMIN2_USERNAME", "BOOTSTRAP_ADMIN2_PASSWORD", "ADMIN002", "Admin2", "User"),
    ]
    for email_key, user_key, pass_key, code, first, last in users:
        email = os.getenv(email_key, "").strip()
        username = os.getenv(user_key, "").strip()
        password = os.getenv(pass_key, "").strip()
        if email and username and password:
            _upsert_bootstrap_user(conn, pwd_ctx, code, email, username, password, first, last)
            conn.commit()


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
        ensure_column(conn, "employees", "desired_role", "TEXT NOT NULL DEFAULT 'user'")
        ensure_column(conn, "login", "role", "TEXT NOT NULL DEFAULT 'user'")
        seed_seo_task_templates(conn)
        seed_bootstrap_admin(conn)
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
