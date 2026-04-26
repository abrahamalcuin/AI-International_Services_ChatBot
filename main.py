from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.parse import quote

import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from passlib.context import CryptContext
from pydantic import BaseModel, Field, validator
from starlette.middleware.sessions import SessionMiddleware

from auth_db import employee_has_login, find_employee_by_token, get_connection, init_db, is_token_expired

load_dotenv()

EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS_PATH", "embeddings.json"))
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
GENERATION_MODEL_NAME = os.getenv("GEMINI_GENERATION_MODEL", "models/gemini-flash-lite-latest")
CATEGORY_BOOST = float(os.getenv("CATEGORY_BOOST", "0.15"))
MAX_CONTEXT_CHUNKS = 25
VALID_CATEGORIES = {"incoming", "current", "graduating"}
PASSWORD_CONTEXT = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-this-in-render")
SEO_TASK_STATUSES = ["Not Started", "In Progress", "Done", "Blocked"]
SEO_SERVICE_TIERS = ["Tier 1", "Tier 2", "Tier 3"]

SYSTEM_PROMPT = """You are BYU-Idaho AdvisorBot, a helpful, accurate, and concise student support assistant.

You MUST follow these rules:

1. Answer ONLY using the information contained in the document chunks provided in the prompt.
2. If the answer is not found in the provided documents, say:
   “I could not find this information in my database, please reach out to international@byui.edu or (208) 496-1320.”
3. Never invent rules, deadlines, dates, or policies.
4. When relevant, include the links provided in the document chunks.
5. Answer the question as concisely as possible, call it the Short Answer, then summarize related info in bullet points and ask if they want more. Call it the Long Answer.
6. Speak clearly, simply, and professionally.
7. Do not reference chunks, embeddings, or the retrieval system.
8. Do not assume anything not explicitly stated in the documents.
9. If there are tips or warnings, include them briefly.
10. Only provide the source at the end.
"""

PAGE_STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #f4f5f7;
  --surface: #ffffff;
  --border: #e2e5ea;
  --border-strong: #ccd0d9;
  --text: #111827;
  --text-secondary: #4b5563;
  --text-muted: #9ca3af;
  --primary: #2563eb;
  --primary-hover: #1d4ed8;
  --primary-light: #eff6ff;
  --danger: #dc2626;
  --success: #16a34a;
  --sidebar-w: 232px;
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
  --shadow-sm: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
  --shadow-md: 0 4px 12px rgba(0,0,0,.07), 0 2px 4px rgba(0,0,0,.04);
}

/* Reset */
body { font-family: 'Inter', system-ui, -apple-system, sans-serif; font-size: 14px; line-height: 1.6; color: var(--text); background: var(--bg); min-height: 100vh; }
a { color: inherit; text-decoration: none; }

/* ── App shell ── */
.app { display: flex; min-height: 100vh; }

/* ── Sidebar ── */
.sidebar {
  width: var(--sidebar-w);
  min-height: 100vh;
  position: fixed;
  top: 0; left: 0; bottom: 0;
  background: #111827;
  display: flex;
  flex-direction: column;
  z-index: 40;
  border-right: 1px solid rgba(255,255,255,.06);
}
.sidebar-brand {
  padding: 22px 18px 18px;
  border-bottom: 1px solid rgba(255,255,255,.07);
}
.sidebar-brand .app-name { font-size: 15px; font-weight: 700; color: #fff; letter-spacing: -0.02em; }
.sidebar-brand .app-sub { font-size: 11px; color: rgba(255,255,255,.4); margin-top: 2px; text-transform: uppercase; letter-spacing: .08em; }
.sidebar-nav { flex: 1; padding: 12px 10px; display: flex; flex-direction: column; gap: 2px; }
.sidebar-nav a {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 12px;
  border-radius: var(--radius-sm);
  font-size: 13px; font-weight: 500;
  color: rgba(255,255,255,.55);
  transition: background 0.15s, color 0.15s;
}
.sidebar-nav a:hover { background: rgba(255,255,255,.07); color: rgba(255,255,255,.9); }
.sidebar-nav a.active { background: rgba(255,255,255,.12); color: #fff; }
.sidebar-nav a svg { flex-shrink: 0; opacity: .7; }
.sidebar-nav a.active svg { opacity: 1; }
.nav-section { font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: .1em; color: rgba(255,255,255,.25); padding: 14px 12px 6px; }
.sidebar-footer {
  padding: 14px 14px 18px;
  border-top: 1px solid rgba(255,255,255,.07);
}
.sidebar-user { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.avatar {
  width: 32px; height: 32px; border-radius: 50%;
  background: var(--primary);
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; color: #fff; flex-shrink: 0;
  text-transform: uppercase;
}
.sidebar-user-info .name { font-size: 13px; font-weight: 600; color: #fff; }
.sidebar-user-info .role { font-size: 11px; color: rgba(255,255,255,.4); }
.sidebar-logout {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 10px; border-radius: var(--radius-sm);
  font-size: 12.5px; font-weight: 500; color: rgba(255,255,255,.4);
  transition: background 0.15s, color 0.15s;
  width: 100%;
}
.sidebar-logout:hover { background: rgba(255,255,255,.07); color: rgba(255,255,255,.8); }

/* ── Main content ── */
.main { margin-left: var(--sidebar-w); flex: 1; min-width: 0; }
.page-shell { padding: 28px 28px; display: grid; gap: 20px; max-width: 1200px; }

/* ── Page header ── */
.page-header { margin-bottom: 4px; }
.page-header h1 { font-size: 20px; font-weight: 700; letter-spacing: -0.02em; }
.page-header p { font-size: 13px; color: var(--text-secondary); margin-top: 3px; }

/* ── Cards ── */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 22px; box-shadow: var(--shadow-sm); }
.card h2 { font-size: 14px; font-weight: 600; letter-spacing: -0.01em; margin-bottom: 16px; color: var(--text); }

/* ── Grid ── */
.grid { display: grid; gap: 16px; }
.grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }

/* ── Metrics ── */
.metric { padding: 20px; border-radius: var(--radius-md); background: var(--surface); border: 1px solid var(--border); box-shadow: var(--shadow-sm); }
.metric .label { font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; color: var(--text-muted); margin-bottom: 10px; }
.metric .value { font-size: 30px; font-weight: 700; letter-spacing: -0.03em; color: var(--text); }

/* ── Forms ── */
form { display: grid; gap: 14px; }
label { display: flex; flex-direction: column; gap: 5px; font-size: 12.5px; font-weight: 500; color: var(--text-secondary); }
input, textarea, select {
  width: 100%; padding: 8px 11px; font-size: 13.5px; font-family: inherit; color: var(--text);
  background: var(--surface); border: 1px solid var(--border-strong); border-radius: var(--radius-sm);
  outline: none; transition: border-color 0.15s, box-shadow 0.15s; appearance: auto;
}
input:focus, textarea:focus, select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37,99,235,.1); }
textarea { min-height: 80px; resize: vertical; }

/* ── Buttons ── */
button, a.button {
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  padding: 8px 15px; font-size: 13px; font-weight: 600; font-family: inherit;
  cursor: pointer; border: 1px solid transparent; border-radius: var(--radius-sm);
  background: var(--primary); color: #fff; transition: background 0.15s; text-decoration: none;
}
button:hover, a.button:hover { background: var(--primary-hover); }
button.secondary, a.secondary { background: var(--surface); color: var(--text); border-color: var(--border-strong); }
button.secondary:hover, a.secondary:hover { background: var(--bg); }
button.danger { background: var(--danger); }
button.danger:hover { background: #b91c1c; }

/* ── Tables ── */
.table { width: 100%; border-collapse: collapse; font-size: 13px; }
.table thead { border-bottom: 1px solid var(--border); }
.table th { padding: 9px 12px; text-align: left; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; color: var(--text-muted); }
.table td { padding: 11px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }
.table tbody tr:hover { background: #fafbfc; }
.table tbody tr:last-child td { border-bottom: none; }
.table strong { font-weight: 600; color: var(--text); }

/* ── Badges ── */
.badge { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 999px; font-size: 11.5px; font-weight: 600; background: var(--primary-light); color: var(--primary); border: 1px solid rgba(37,99,235,.12); }
.badge.green { background: #f0fdf4; color: #16a34a; border-color: rgba(22,163,74,.15); }
.badge.yellow { background: #fefce8; color: #ca8a04; border-color: rgba(202,138,4,.15); }
.badge.red { background: #fef2f2; color: var(--danger); border-color: rgba(220,38,38,.15); }
.badge.gray { background: #f8fafc; color: #64748b; border-color: rgba(100,116,139,.15); }

/* ── Avatar chip ── */
.assignee-chip { display: inline-flex; align-items: center; gap: 5px; padding: 2px 8px 2px 4px; border-radius: 999px; background: #f1f5f9; border: 1px solid var(--border); font-size: 11.5px; font-weight: 500; color: var(--text-secondary); }
.assignee-chip .av { width: 18px; height: 18px; border-radius: 50%; background: var(--primary); color: #fff; font-size: 9px; font-weight: 700; display: flex; align-items: center; justify-content: center; text-transform: uppercase; }

/* ── Utilities ── */
.muted { color: var(--text-muted); font-size: 12.5px; }
.error { color: var(--danger); font-size: 13px; }
.ok { color: var(--success); font-size: 13px; }

/* ── Login ── */
.login-wrap { min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 24px; background: var(--bg); }
.login-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 36px 32px; width: 100%; max-width: 380px; box-shadow: var(--shadow-md); }
.login-card h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 6px; }
.login-card .subtitle { color: var(--text-secondary); font-size: 13px; margin-bottom: 28px; }

/* ── Mobile ── */
@media (max-width: 900px) {
  .sidebar { transform: translateX(-100%); transition: transform 0.2s; }
  .sidebar.open { transform: translateX(0); }
  .main { margin-left: 0; }
  .page-shell { padding: 16px; }
  .grid.two, .grid.three { grid-template-columns: 1fr; }
  .mobile-menu-btn { display: flex !important; }
}
.mobile-menu-btn { display: none; position: fixed; top: 14px; left: 14px; z-index: 50; background: #111827; color: #fff; border: none; border-radius: var(--radius-sm); padding: 8px; cursor: pointer; }
</style>"""


class SourceChunk(BaseModel):
    id: str
    category: str
    source: str
    score: float
    text: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5)
    category: Optional[str] = Field(default=None)
    top_k: int = Field(default=MAX_CONTEXT_CHUNKS, ge=1, le=MAX_CONTEXT_CHUNKS)

    @validator("category")
    def normalize_category(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip().lower()
        if normalized not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}")
        return normalized


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class SEOClientCreate(BaseModel):
    company_name: str = Field(..., min_length=2)
    industry: Optional[str] = None
    credentials: Optional[str] = None
    contact_person: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website_url: Optional[str] = None
    notes: Optional[str] = None
    start_date: Optional[str] = None
    due_date: Optional[str] = None
    service_tier: str

    @validator("service_tier")
    def validate_service_tier(cls, value: str) -> str:
        value = value.strip()
        if value not in SEO_SERVICE_TIERS:
            raise ValueError(f"service_tier must be one of {SEO_SERVICE_TIERS}")
        return value


class SEOTaskStatusUpdate(BaseModel):
    status: str

    @validator("status")
    def validate_status(cls, value: str) -> str:
        value = value.strip()
        if value not in SEO_TASK_STATUSES:
            raise ValueError(f"status must be one of {SEO_TASK_STATUSES}")
        return value


@dataclass
class EmbeddingRecord:
    id: str
    category: str
    source: str
    text: str
    vector: np.ndarray


def configure_genai() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("The GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)


def normalize_vector(vector: Sequence[float]) -> np.ndarray:
    array = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0:
        raise ValueError("Embedding vector has zero magnitude.")
    return array / norm


def load_embedding_index(path: Path) -> List[EmbeddingRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file '{path}' was not found. Run build_embeddings.py first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for record in data.get("records", []):
        records.append(
            EmbeddingRecord(
                id=record["id"],
                category=record["category"].lower(),
                source=record["source"],
                text=record["text"],
                vector=normalize_vector(record["embedding"]),
            )
        )
    if not records:
        raise RuntimeError(f"No embeddings found inside '{path}'.")
    return records


class GeminiClient:
    def __init__(self) -> None:
        configure_genai()
        self.generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)

    def embed(self, text: str, task_type: str) -> np.ndarray:
        response = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=text, task_type=task_type)
        embedding = response.get("embedding")
        if not embedding:
            raise RuntimeError("Gemini did not return an embedding vector.")
        return normalize_vector(embedding)

    def generate_answer(self, prompt: str) -> str:
        response = self.generation_model.generate_content(
            prompt,
            generation_config={
                "temperature": float(os.getenv("GENERATION_TEMPERATURE", "0.2")),
                "max_output_tokens": int(os.getenv("GENERATION_MAX_OUTPUT_TOKENS", "1024")),
                "top_p": 0.9,
            },
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")
        return response.text.strip()


def rank_chunks(query_vector: np.ndarray, category: Optional[str], top_k: int) -> List[Tuple[float, EmbeddingRecord]]:
    scored: List[Tuple[float, EmbeddingRecord]] = []
    for record in EMBEDDING_INDEX:
        score = float(np.dot(query_vector, record.vector))
        if category and record.category == category:
            score += CATEGORY_BOOST
        scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[: min(top_k, MAX_CONTEXT_CHUNKS)]


def build_prompt(question: str, scored_chunks: List[Tuple[float, EmbeddingRecord]]) -> str:
    sections = []
    for idx, (score, record) in enumerate(scored_chunks, start=1):
        sections.append(f"Chunk {idx} | Category: {record.category} | Source: {record.source} | Score: {score:.4f}\n{record.text}")
    context_block = "\n\n---\n\n".join(sections)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context_block}\n\nUser question: {question.strip()}\n\nFinal answer:" 


def render_page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>{PAGE_STYLE}<title>{title} — SEO Manager</title></head><body>{body}</body></html>")


def sidebar_html(active: str, user) -> str:
    initials = (user["first_name"][0] + user["last_name"][0]).upper() if user else "?"
    name = f"{user['first_name']} {user['last_name']}" if user else ""
    role = user["role"].capitalize() if user else ""
    nav_icon = {
        "dashboard": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>',
        "clients":   '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        "tasks":     '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>',
        "logout":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>',
    }
    def nav_link(key, label, href):
        cls = "active" if active == key else ""
        return f"<a href='{href}' class='{cls}'>{nav_icon[key]}{label}</a>"
    return f"""
    <button class='mobile-menu-btn' onclick="document.querySelector('.sidebar').classList.toggle('open')">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
    </button>
    <nav class='sidebar'>
      <div class='sidebar-brand'>
        <div class='app-name'>SEO Manager</div>
        <div class='app-sub'>Workspace</div>
      </div>
      <div class='sidebar-nav'>
        <div class='nav-section'>Main</div>
        {nav_link('dashboard', 'Dashboard', '/SEO')}
        {nav_link('clients', 'Clients', '/SEO/clients')}
        {nav_link('tasks', 'Tasks', '/SEO/tasks')}
      </div>
      <div class='sidebar-footer'>
        <div class='sidebar-user'>
          <div class='avatar'>{initials}</div>
          <div class='sidebar-user-info'>
            <div class='name'>{name}</div>
            <div class='role'>{role}</div>
          </div>
        </div>
        <a href='/logout' class='sidebar-logout'>{nav_icon['logout']} Sign out</a>
      </div>
    </nav>"""


def render_app_page(title: str, body: str, active: str, user) -> HTMLResponse:
    sidebar = sidebar_html(active, user)
    return HTMLResponse(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>{PAGE_STYLE}<title>{title} — SEO Manager</title></head><body><div class='app'>{sidebar}<div class='main'>{body}</div></div></body></html>")


def current_user(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT employees.id, employees.first_name, employees.last_name, employees.email, login.username, login.role
            FROM employees
            JOIN login ON login.employee_id = employees.id
            WHERE employees.id = ?
            """,
            (user_id,),
        ).fetchone()


def require_user(request: Request):
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def seo_metrics() -> dict:
    with get_connection() as conn:
        return {
            "client_count": conn.execute("SELECT COUNT(*) FROM seo_clients").fetchone()[0],
            "task_count": conn.execute("SELECT COUNT(*) FROM seo_client_tasks").fetchone()[0],
            "blocked_count": conn.execute("SELECT COUNT(*) FROM seo_client_tasks WHERE status = 'Blocked'").fetchone()[0],
            "done_count": conn.execute("SELECT COUNT(*) FROM seo_client_tasks WHERE status = 'Done'").fetchone()[0],
        }


def create_client_tasks(conn, client_id: int, service_tier: str, due_date: Optional[str]) -> int:
    templates = conn.execute(
        """
        SELECT id, category, task_name, task_description, sort_order, default_assignee
        FROM seo_task_templates
        WHERE service_tier = ? AND is_active = 1
        ORDER BY sort_order, id
        """,
        (service_tier,),
    ).fetchall()
    for template in templates:
        conn.execute(
            """
            INSERT INTO seo_client_tasks
            (client_id, template_id, category, task_name, task_description, sort_order, due_date, assigned_to_username)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (client_id, template["id"], template["category"], template["task_name"],
             template["task_description"], template["sort_order"], due_date,
             template["default_assignee"]),
        )
    return len(templates)


def seo_dashboard_html(user) -> str:
    metrics = seo_metrics()
    tier_options = "".join(f"<option value='{tier}'>{tier}</option>" for tier in SEO_SERVICE_TIERS)
    status_options_js = json.dumps(SEO_TASK_STATUSES)
    return f"""
    <div class='page-shell'>
      <div class='page-header'>
        <h1>Dashboard</h1>
        <p>Good to see you, {user['first_name']}.</p>
      </div>
      <div class='grid three'>
        <div class='metric'><div class='label'>Clients</div><div class='value' id='metricClients'>{metrics['client_count']}</div></div>
        <div class='metric'><div class='label'>Total tasks</div><div class='value' id='metricOpen'>{metrics['task_count']}</div></div>
        <div class='metric'><div class='label'>Blocked</div><div class='value' id='metricBlocked'>{metrics['blocked_count']}</div></div>
      </div>
      <div class='grid two'>
        <div class='card'>
          <h2>Add client</h2>
          <form id='clientForm'>
            <div class='grid two'>
              <label>Company name<input name='company_name' required></label>
              <label>Industry<input name='industry'></label>
              <label>Contact person<input name='contact_person'></label>
              <label>Email<input name='email' type='email'></label>
              <label>Phone<input name='phone'></label>
              <label>Website URL<input name='website_url' type='url'></label>
              <label>Start date<input name='start_date' type='date'></label>
              <label>Due date<input name='due_date' type='date'></label>
              <label>Service tier<select name='service_tier'>{tier_options}</select></label>
            </div>
            <label>WordPress / website credentials<textarea name='credentials' placeholder='Stored here because it was a required field. Consider encrypted storage later.'></textarea></label>
            <label>Notes<textarea name='notes'></textarea></label>
            <button type='submit'>Create client and tasks</button>
          </form>
        </div>
        <div class='card'>
          <h2>Template summary</h2>
          <p class='muted'>Each tier uses normalized, deduplicated task templates.</p>
          <div id='templateSummary' class='grid'></div>
        </div>
      </div>
      <div class='grid two'>
        <div class='card'>
          <h2>Clients</h2>
          <table class='table'>
            <thead><tr><th>Client</th><th>Tier</th><th>Start</th><th>Due</th><th>Contact</th></tr></thead>
            <tbody id='clientList'></tbody>
          </table>
        </div>
        <div class='card'>
          <h2>Recent tasks</h2>
          <table class='table'>
            <thead><tr><th>Task</th><th>Category</th><th>Assignee</th><th>Status</th><th>Update</th></tr></thead>
            <tbody id='taskList'></tbody>
          </table>
        </div>
      </div>
    </div>
    <script>
      const STATUSES = {status_options_js};
      function statusClass(s) {{
        return {{  'Not Started':'gray','In Progress':'','Done':'green','Blocked':'red' }}[s]||'';
      }}
      function assigneeChip(u) {{
        if (!u) return '<span class="muted">—</span>';
        return `<span class="assignee-chip"><span class="av">${{u[0]}}</span>${{u}}</span>`;
      }}
      async function loadDashboard() {{
        const [clientsRes, tasksRes, templatesRes] = await Promise.all([
          fetch('/api/seo/clients'),
          fetch('/api/seo/tasks'),
          fetch('/api/seo/templates')
        ]);
        const clients = await clientsRes.json();
        const tasks = await tasksRes.json();
        const templates = await templatesRes.json();
        document.getElementById('metricClients').textContent = clients.length;
        document.getElementById('metricOpen').textContent = tasks.length;
        document.getElementById('metricBlocked').textContent = tasks.filter(t => t.status === 'Blocked').length;
        document.getElementById('clientList').innerHTML = clients.length ? clients.map(client => `
          <tr>
            <td><strong>${{client.company_name}}</strong><div class='muted'>${{client.industry || '—'}}</div></td>
            <td><span class='badge'>${{client.service_tier}}</span></td>
            <td>${{client.start_date || '—'}}</td>
            <td>${{client.due_date || '—'}}</td>
            <td>${{client.contact_person || '—'}}</td>
          </tr>`).join('') : "<tr><td colspan='5' class='muted' style='padding:20px 12px;'>No clients yet.</td></tr>";
        document.getElementById('taskList').innerHTML = tasks.length ? tasks.slice(0, 10).map(task => `
          <tr>
            <td><strong>${{task.task_name}}</strong><div class='muted'>${{task.company_name}}</div></td>
            <td><span class='badge gray'>${{task.category}}</span></td>
            <td>${{assigneeChip(task.assigned_to_username)}}</td>
            <td><span class='badge ${{statusClass(task.status)}}'>${{task.status}}</span></td>
            <td><select onchange="updateTaskStatus(${{task.id}}, this.value)">${{STATUSES.map(status => `<option value="${{status}}" ${{status === task.status ? 'selected' : ''}}>${{status}}</option>`).join('')}}</select></td>
          </tr>`).join('') : "<tr><td colspan='5' class='muted' style='padding:20px 12px;'>No tasks yet.</td></tr>";
        document.getElementById('templateSummary').innerHTML = templates.map(item => `<div class='metric'><div class='label'>${{item.service_tier}}</div><div class='value' style='font-size:22px'>${{item.task_count}}</div><div class='muted'>tasks</div></div>`).join('');
      }}
      async function createClient(event) {{
        event.preventDefault();
        const payload = Object.fromEntries(new FormData(event.target).entries());
        const res = await fetch('/api/seo/clients', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify(payload) }});
        const data = await res.json();
        if (!res.ok) {{ alert(data.detail || 'Unable to create client'); return; }}
        event.target.reset();
        alert(`Created ${{data.company_name}} and generated ${{data.generated_task_count}} tasks.`);
        await loadDashboard();
      }}
      async function updateTaskStatus(taskId, status) {{
        const res = await fetch(`/api/seo/tasks/${{taskId}}`, {{ method: 'PATCH', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ status }}) }});
        if (!res.ok) {{
          const data = await res.json();
          alert(data.detail || 'Unable to update task');
          return;
        }}
        await loadDashboard();
      }}
      document.addEventListener('DOMContentLoaded', () => {{
        document.getElementById('clientForm').addEventListener('submit', createClient);
        loadDashboard();
      }});
    </script>
    """


def seo_clients_html() -> str:
    tier_options = "".join(f"<option value='{t}'>{t}</option>" for t in SEO_SERVICE_TIERS)
    status_options_js = json.dumps(SEO_TASK_STATUSES)
    return f"""
    <div class='page-shell'>
      <div class='page-header'>
        <h1>Clients</h1>
        <p>All onboarded SEO clients.</p>
      </div>
      <div class='card'>
        <div style='display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;'>
          <input id='clientSearch' placeholder='Search clients...' style='flex:1;min-width:200px;'>
          <select id='tierFilter'><option value=''>All tiers</option>{tier_options}</select>
        </div>
        <table class='table' id='clientTable'>
          <thead>
            <tr>
              <th>Company</th><th>Tier</th><th>Industry</th><th>Contact</th>
              <th>Website</th><th>Start</th><th>Due</th><th>Tasks</th><th>Actions</th>
            </tr>
          </thead>
          <tbody id='clientRows'></tbody>
        </table>
      </div>
      <div id='clientDetailModal' style='display:none;position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:100;overflow:auto;'>
        <div class='card' style='max-width:640px;margin:60px auto;position:relative;'>
          <button onclick="document.getElementById('clientDetailModal').style.display='none'" class='secondary' style='position:absolute;top:12px;right:12px;'>Close</button>
          <div id='clientDetailContent'></div>
        </div>
      </div>
    </div>
    <script>
      const STATUSES = {status_options_js};
      let allClients = [];
      async function loadClients() {{
        const res = await fetch('/api/seo/clients');
        allClients = await res.json();
        renderClients();
      }}
      function renderClients() {{
        const q = document.getElementById('clientSearch').value.toLowerCase();
        const tier = document.getElementById('tierFilter').value;
        const filtered = allClients.filter(c =>
          (!q || c.company_name.toLowerCase().includes(q) || (c.contact_person||'').toLowerCase().includes(q) || (c.industry||'').toLowerCase().includes(q)) &&
          (!tier || c.service_tier === tier)
        );
        document.getElementById('clientRows').innerHTML = filtered.length ? filtered.map(c => `
          <tr>
            <td><strong>${{c.company_name}}</strong></td>
            <td><span class='badge'>${{c.service_tier}}</span></td>
            <td>${{c.industry || '—'}}</td>
            <td>${{c.contact_person || '—'}}<div class='muted'>${{c.email || ''}}</div></td>
            <td>${{c.website_url ? `<a href="${{c.website_url}}" target="_blank" rel="noreferrer">${{c.website_url.replace(/^https?:\\/\\//, '')}}</a>` : '—'}}</td>
            <td>${{c.start_date || '—'}}</td>
            <td>${{c.due_date || '—'}}</td>
            <td><a href="/SEO/tasks?client_id=${{c.id}}" class="button secondary" style="font-size:12px;padding:6px 10px;">View tasks</a></td>
            <td>
              <button onclick="showClientDetail(${{c.id}})" style='font-size:12px;padding:6px 10px;background:#eef2ff;color:#111;border:0;border-radius:8px;cursor:pointer;'>Details</button>
              <button onclick="deleteClient(${{c.id}}, '${{c.company_name.replace(/'/g, \"\\\\'\")}}')" style='font-size:12px;padding:6px 10px;background:#fee2e2;color:#b91c1c;border:0;border-radius:8px;cursor:pointer;margin-left:4px;'>Delete</button>
            </td>
          </tr>`).join('') : "<tr><td colspan='9' class='muted'>No clients match.</td></tr>";
      }}
      async function showClientDetail(id) {{
        const res = await fetch('/api/seo/clients/' + id);
        const c = await res.json();
        document.getElementById('clientDetailContent').innerHTML = `
          <h2>${{c.company_name}}</h2>
          <p class='muted'>Service tier: <strong>${{c.service_tier}}</strong></p>
          <table class='table'>
            ${{[['Industry', c.industry], ['Contact', c.contact_person], ['Email', c.email], ['Phone', c.phone],
               ['Website', c.website_url], ['Start', c.start_date], ['Due', c.due_date],
               ['Notes', c.notes], ['Credentials', c.credentials ? '••••••' : '—']].map(([k, v]) =>
              `<tr><th style='width:120px'>${{k}}</th><td>${{v || '—'}}</td></tr>`).join('')}}
          </table>`;
        document.getElementById('clientDetailModal').style.display = 'block';
      }}
      async function deleteClient(id, name) {{
        if (!confirm('Delete ' + name + ' and all their tasks? This cannot be undone.')) return;
        const res = await fetch('/api/seo/clients/' + id, {{ method: 'DELETE' }});
        if (res.ok) {{ await loadClients(); }} else {{ alert('Delete failed.'); }}
      }}
      document.getElementById('clientSearch').addEventListener('input', renderClients);
      document.getElementById('tierFilter').addEventListener('change', renderClients);
      loadClients();
    </script>
    """


def seo_tasks_html() -> str:
    category_options = "".join(f"<option value='{c}'>{c.title()}</option>" for c in ["keyword research", "on-page", "off-page", "technical", "extras"])
    status_options = "".join(f"<option value='{s}'>{s}</option>" for s in SEO_TASK_STATUSES)
    status_options_js = json.dumps(SEO_TASK_STATUSES)
    tier_options = "".join(f"<option value='{t}'>{t}</option>" for t in SEO_SERVICE_TIERS)
    return f"""
    <div class='page-shell'>
      <div class='page-header'>
        <h1>Tasks</h1>
        <p>All client tasks across categories and tiers.</p>
      </div>
      <div class='card'>
        <div style='display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;'>
          <input id='taskSearch' placeholder='Search tasks or clients...' style='flex:1;min-width:200px;'>
          <select id='catFilter'><option value=''>All categories</option>{category_options}</select>
          <select id='statusFilter'><option value=''>All statuses</option>{status_options}</select>
          <select id='tierFilter2'><option value=''>All tiers</option>{tier_options}</select>
        </div>
        <table class='table' id='taskTable'>
          <thead>
            <tr><th>Task</th><th>Client</th><th>Category</th><th>Assignee</th><th>Status</th><th>Due</th><th>Update</th></tr>
          </thead>
          <tbody id='taskRows'></tbody>
        </table>
      </div>
    </div>
    <script>
      const STATUSES = {status_options_js};
      function statusClass(s) {{ return {{'Not Started':'gray','In Progress':'','Done':'green','Blocked':'red'}}[s]||''; }}
      function assigneeChip(u) {{ return u ? `<span class="assignee-chip"><span class="av">${{u[0]}}</span>${{u}}</span>` : '<span class="muted">—</span>'; }}
      let allTasks = [];
      async function loadTasks() {{
        const params = new URLSearchParams(window.location.search);
        const res = await fetch('/api/seo/tasks');
        allTasks = await res.json();
        const clientId = params.get('client_id');
        if (clientId) {{
          allTasks = allTasks.filter(t => String(t.client_id) === clientId);
        }}
        renderTasks();
      }}
      function renderTasks() {{
        const q = document.getElementById('taskSearch').value.toLowerCase();
        const cat = document.getElementById('catFilter').value;
        const stat = document.getElementById('statusFilter').value;
        const tier = document.getElementById('tierFilter2').value;
        const filtered = allTasks.filter(t =>
          (!q || t.task_name.toLowerCase().includes(q) || (t.company_name||'').toLowerCase().includes(q)) &&
          (!cat || t.category === cat) &&
          (!stat || t.status === stat)
        );
        document.getElementById('taskRows').innerHTML = filtered.length ? filtered.map(t => `
          <tr>
            <td><strong>${{t.task_name}}</strong><div class='muted'>${{t.task_description || ''}}</div></td>
            <td>${{t.company_name}}</td>
            <td><span class='badge gray'>${{t.category}}</span></td>
            <td>${{assigneeChip(t.assigned_to_username)}}</td>
            <td><span class='badge ${{statusClass(t.status)}}'>${{t.status}}</span></td>
            <td>${{t.due_date || '—'}}</td>
            <td><select onchange="updateStatus(${{t.id}}, this.value)">${{STATUSES.map(s => `<option value="${{s}}" ${{s===t.status?'selected':''}}>${{s}}</option>`).join('')}}</select></td>
          </tr>`).join('') : "<tr><td colspan='7' class='muted' style='padding:20px 12px'>No tasks match.</td></tr>";
      }}
      function statusColor(s) {{
        return {{
          'Not Started': '#f1f5f9',
          'In Progress': '#e0f2fe',
          'Done': '#dcfce7',
          'Blocked': '#fee2e2'
        }}[s] || '#f1f5f9';
      }}
      async function updateStatus(id, status) {{
        const res = await fetch('/api/seo/tasks/' + id, {{
          method: 'PATCH',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ status }})
        }});
        if (!res.ok) {{ alert('Update failed'); return; }}
        await loadTasks();
      }}
      ['taskSearch', 'catFilter', 'statusFilter', 'tierFilter2'].forEach(id => document.getElementById(id).addEventListener('change', renderTasks));
      document.getElementById('taskSearch').addEventListener('input', renderTasks);
      loadTasks();
    </script>
    """


app = FastAPI(
    title="BYU-Idaho Student Advisor RAG API",
    version="1.2.0",
    description="Retrieval-Augmented Generation backend that powers the BYU-Idaho chatbot.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax", https_only=True)

EMBEDDING_INDEX: List[EmbeddingRecord] = []
GEMINI_CLIENT: Optional[GeminiClient] = None


@app.on_event("startup")
async def startup_event() -> None:
    global EMBEDDING_INDEX, GEMINI_CLIENT
    init_db()
    try:
        GEMINI_CLIENT = GeminiClient()
        EMBEDDING_INDEX = load_embedding_index(EMBEDDINGS_PATH)
    except Exception as exc:
        print(f"Startup warning: chatbot features unavailable: {exc}")
        GEMINI_CLIENT = None
        EMBEDDING_INDEX = []


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/SEO") -> HTMLResponse:
    user = current_user(request)
    if user:
        return RedirectResponse(next, status_code=302)
    return render_page("Login", f"<div class='login-wrap'><div class='login-card'><h1>SEO Manager</h1><p class='subtitle'>Sign in to your workspace</p><form method='post' action='/login'><input type='hidden' name='next' value='{next}'><label>Username<input name='username' required autocomplete='username' placeholder='Your username'></label><label>Password<input type='password' name='password' required autocomplete='current-password' placeholder='••••••••'></label><button type='submit' style='width:100%;margin-top:4px;padding:11px;font-size:14px;'>Sign in</button></form></div></div>")


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...), next: str = Form("/SEO")):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT login.password_hash, login.role, employees.id, employees.is_active
            FROM login
            JOIN employees ON employees.id = login.employee_id
            WHERE login.username = ?
            """,
            (username.strip(),),
        ).fetchone()
        if row and row["is_active"] and PASSWORD_CONTEXT.verify(password, row["password_hash"]):
            request.session["user_id"] = row["id"]
            conn.execute("UPDATE login SET last_login_at = CURRENT_TIMESTAMP WHERE employee_id = ?", (row["id"],))
            conn.commit()
            return RedirectResponse(next or "/SEO", status_code=302)
    return render_page("Login", f"<div class='login-wrap'><div class='login-card'><h1>SEO Manager</h1><p class='subtitle'>Sign in to your workspace</p><p class='error' style='margin-bottom:16px;'>Incorrect username or password.</p><form method='post' action='/login'><input type='hidden' name='next' value='{next}'><label>Username<input name='username' required autocomplete='username' placeholder='Your username'></label><label>Password<input type='password' name='password' required autocomplete='current-password' placeholder='••••••••'></label><button type='submit' style='width:100%;margin-top:4px;padding:11px;font-size:14px;'>Sign in</button></form></div></div>")


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/onboarding", response_class=HTMLResponse)
async def onboarding_page(token: str):
    employee = find_employee_by_token(token.strip())
    if employee is None:
        raise HTTPException(status_code=404, detail="Invalid onboarding link.")
    if not employee["is_active"]:
        raise HTTPException(status_code=403, detail="Employee account is inactive.")
    if is_token_expired(employee["invite_expires_at"]):
        raise HTTPException(status_code=403, detail="This onboarding link has expired.")
    if employee_has_login(employee["id"]):
        return render_page("Onboarding", "<div class='card'><h1>Account already exists</h1><p class='ok'>This employee already has a login.</p><a class='button' href='/login'>Go to login</a></div>")
    return render_page("Onboarding", f"<div class='card'><h1>Create your account</h1><p>{employee['email']}</p><form method='post' action='/onboarding'><input type='hidden' name='token' value='{token}'><label>Username</label><input name='username' minlength='4' maxlength='50' required><label>Password</label><input type='password' name='password' minlength='8' required><label>Confirm password</label><input type='password' name='confirm_password' minlength='8' required><button type='submit'>Create account</button></form></div>")


@app.post("/onboarding", response_class=HTMLResponse)
async def onboarding_submit(token: str = Form(...), username: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    employee = find_employee_by_token(token.strip())
    if employee is None:
        raise HTTPException(status_code=404, detail="Invalid onboarding link.")
    if not employee["is_active"]:
        raise HTTPException(status_code=403, detail="Employee account is inactive.")
    if is_token_expired(employee["invite_expires_at"]):
        raise HTTPException(status_code=403, detail="This onboarding link has expired.")
    if employee_has_login(employee["id"]):
        return RedirectResponse("/login", status_code=302)
    if len(username.strip()) < 4:
        return render_page("Onboarding", "<div class='card'><p class='error'>Username must be at least 4 characters.</p><a href='javascript:history.back()'>Go back</a></div>")
    if len(password) < 8:
        return render_page("Onboarding", "<div class='card'><p class='error'>Password must be at least 8 characters.</p><a href='javascript:history.back()'>Go back</a></div>")
    if password != confirm_password:
        return render_page("Onboarding", "<div class='card'><p class='error'>Passwords do not match.</p><a href='javascript:history.back()'>Go back</a></div>")
    password_hash = PASSWORD_CONTEXT.hash(password)
    with get_connection() as conn:
        try:
            conn.execute(
                "INSERT INTO login (employee_id, username, password_hash, role) VALUES (?, ?, ?, ?)",
                (employee["id"], username.strip(), password_hash, employee["desired_role"]),
            )
            conn.execute("UPDATE employees SET invite_token = NULL, invite_expires_at = NULL WHERE id = ?", (employee["id"],))
            conn.commit()
        except Exception as exc:
            if "UNIQUE constraint failed: login.username" in str(exc):
                return render_page("Onboarding", "<div class='card'><p class='error'>That username is already taken.</p><a href='javascript:history.back()'>Go back</a></div>")
            raise
    return render_page("Onboarding", "<div class='card'><h1>Account created</h1><p class='ok'>Your login is ready.</p><a class='button' href='/login'>Go to login</a></div>")


@app.get("/")
async def root():
    return {"ok": True, "service": "byui-student-advisor-api", "version": "1.2.0"}


@app.get("/health")
async def health():
    return {"ok": True, "chatbotReady": GEMINI_CLIENT is not None and bool(EMBEDDING_INDEX), "authReady": True}


@app.get("/SEO", response_class=HTMLResponse)
async def seo_dashboard(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse(f"/login?next={quote('/SEO')}", status_code=302)
    return render_app_page("Dashboard", seo_dashboard_html(user), "dashboard", user)


@app.get("/SEO/clients", response_class=HTMLResponse)
async def seo_clients_page(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse(f"/login?next={quote('/SEO/clients')}", status_code=302)
    return render_app_page("Clients", seo_clients_html(), "clients", user)


@app.get("/SEO/tasks", response_class=HTMLResponse)
async def seo_tasks_page(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse(f"/login?next={quote('/SEO/tasks')}", status_code=302)
    return render_app_page("Tasks", seo_tasks_html(), "tasks", user)


@app.get("/api/seo/templates")
async def list_seo_templates(request: Request):
    require_user(request)
    with get_connection() as conn:
        rows = conn.execute("SELECT service_tier, COUNT(*) AS task_count FROM seo_task_templates WHERE is_active = 1 GROUP BY service_tier ORDER BY service_tier").fetchall()
        return [dict(row) for row in rows]


@app.get("/api/seo/clients")
async def list_seo_clients(request: Request):
    require_user(request)
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM seo_clients ORDER BY created_at DESC, id DESC").fetchall()
        return [dict(row) for row in rows]


@app.post("/api/seo/clients")
async def create_seo_client(request: Request, payload: SEOClientCreate):
    user = require_user(request)
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO seo_clients (
                company_name, industry, credentials, contact_person, email, phone,
                website_url, notes, start_date, due_date, service_tier, created_by_employee_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.company_name.strip(), payload.industry, payload.credentials, payload.contact_person,
                payload.email, payload.phone, payload.website_url, payload.notes,
                payload.start_date, payload.due_date, payload.service_tier, user["id"],
            ),
        )
        client_id = cursor.lastrowid
        generated_task_count = create_client_tasks(conn, client_id, payload.service_tier, payload.due_date)
        conn.commit()
        client = conn.execute("SELECT * FROM seo_clients WHERE id = ?", (client_id,)).fetchone()
    data = dict(client)
    data["generated_task_count"] = generated_task_count
    return data


@app.get("/api/seo/clients/{client_id}")
async def get_seo_client(client_id: int, request: Request):
    require_user(request)
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM seo_clients WHERE id = ?", (client_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Client not found")
        return dict(row)


@app.delete("/api/seo/clients/{client_id}")
async def delete_seo_client(client_id: int, request: Request):
    require_user(request)
    with get_connection() as conn:
        existing = conn.execute("SELECT id FROM seo_clients WHERE id = ?", (client_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Client not found")
        conn.execute("DELETE FROM seo_clients WHERE id = ?", (client_id,))
        conn.commit()
    return {"ok": True}


@app.get("/api/seo/tasks")
async def list_seo_tasks(request: Request):
    require_user(request)
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT seo_client_tasks.*, seo_clients.company_name
            FROM seo_client_tasks
            JOIN seo_clients ON seo_clients.id = seo_client_tasks.client_id
            ORDER BY CASE seo_client_tasks.status
                WHEN 'Blocked' THEN 1
                WHEN 'In Progress' THEN 2
                WHEN 'Not Started' THEN 3
                ELSE 4 END,
                seo_client_tasks.sort_order,
                seo_client_tasks.id
            """
        ).fetchall()
        return [dict(row) for row in rows]


@app.patch("/api/seo/tasks/{task_id}")
async def update_seo_task(task_id: int, request: Request, payload: SEOTaskStatusUpdate):
    require_user(request)
    with get_connection() as conn:
        existing = conn.execute("SELECT id FROM seo_client_tasks WHERE id = ?", (task_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Task not found")
        conn.execute(
            "UPDATE seo_client_tasks SET status = ?, completed_at = CASE WHEN ? = 'Done' THEN CURRENT_TIMESTAMP ELSE NULL END, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (payload.status, payload.status, task_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM seo_client_tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row)


class AdminUserCreate(BaseModel):
    secret: str
    email: str
    username: str
    password: str
    first_name: str = "Admin"
    last_name: str = "User"
    role: str = "admin"


@app.post("/api/admin/create-user")
async def admin_create_user(payload: AdminUserCreate):
    expected = os.getenv("ADMIN_SECRET", "")
    if not expected or payload.secret != expected:
        raise HTTPException(status_code=403, detail="Forbidden")
    from passlib.context import CryptContext
    pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
    with get_connection() as conn:
        existing_emp = conn.execute("SELECT id FROM employees WHERE email = ?", (payload.email,)).fetchone()
        if existing_emp:
            employee_id = existing_emp[0]
        else:
            cursor = conn.execute(
                "INSERT INTO employees (employee_code, first_name, last_name, email, desired_role) VALUES (?, ?, ?, ?, ?)",
                (f"EMP{secrets.token_hex(4).upper()}", payload.first_name, payload.last_name, payload.email, payload.role),
            )
            employee_id = cursor.lastrowid
        existing_login = conn.execute("SELECT id FROM login WHERE employee_id = ?", (employee_id,)).fetchone()
        if existing_login:
            conn.execute(
                "UPDATE login SET username = ?, password_hash = ?, role = ? WHERE employee_id = ?",
                (payload.username, pwd_ctx.hash(payload.password), payload.role, employee_id),
            )
            action = "updated"
        else:
            conn.execute(
                "INSERT INTO login (employee_id, username, password_hash, role) VALUES (?, ?, ?, ?)",
                (employee_id, payload.username, pwd_ctx.hash(payload.password), payload.role),
            )
            action = "created"
        conn.commit()
    return {"ok": True, "action": action, "username": payload.username}


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    if GEMINI_CLIENT is None or not EMBEDDING_INDEX:
        raise HTTPException(status_code=503, detail="Chatbot features are unavailable right now.")
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question text cannot be empty.")
    query_vector = GEMINI_CLIENT.embed(question, task_type="retrieval_query")
    scored_chunks = rank_chunks(query_vector, payload.category, payload.top_k)
    if not scored_chunks:
        raise HTTPException(status_code=500, detail="No knowledge chunks are available. Rebuild the embeddings index.")
    prompt = build_prompt(question, scored_chunks)
    answer = await run_in_threadpool(GEMINI_CLIENT.generate_answer, prompt)
    sources = [SourceChunk(id=record.id, category=record.category, source=record.source, score=round(score, 4), text=record.text) for score, record in scored_chunks]
    return AskResponse(answer=answer, sources=sources)
