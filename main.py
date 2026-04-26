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
.nav-tree { display: grid; gap: 2px; }
.nav-tree summary {
  list-style: none;
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 9px 12px;
  border-radius: var(--radius-sm);
  font-size: 13px; font-weight: 500;
  color: rgba(255,255,255,.55);
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
.nav-tree summary::-webkit-details-marker { display: none; }
.nav-tree summary:hover { background: rgba(255,255,255,.07); color: rgba(255,255,255,.9); }
.nav-tree[open] summary, .nav-tree.is-active summary { background: rgba(255,255,255,.12); color: #fff; }
.nav-tree-label { display: inline-flex; align-items: center; gap: 10px; }
.nav-tree-caret { font-size: 11px; opacity: .8; transition: transform 0.15s; }
.nav-tree[open] .nav-tree-caret { transform: rotate(90deg); }
.nav-tree-links { display: grid; gap: 2px; padding: 4px 0 2px 36px; }
.nav-tree-links a { font-size: 12.5px; padding: 8px 10px; color: rgba(255,255,255,.5); }
.nav-tree-links a.active { background: rgba(255,255,255,.09); color: #fff; }
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

.hero-card { padding: 28px; display: grid; gap: 18px; }
.hero-card h2 { margin-bottom: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.03em; }
.hero-card p { max-width: 72ch; color: var(--text-secondary); }
.stat-grid { display: grid; gap: 16px; grid-template-columns: repeat(4, minmax(0, 1fr)); }
.metric-subtle { font-size: 12px; color: var(--text-muted); margin-top: 6px; }
.insight-list { display: grid; gap: 12px; }
.insight-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
.progress-track { flex: 1; height: 10px; background: #edf1f6; border-radius: 999px; overflow: hidden; }
.progress-bar { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #2563eb, #60a5fa); }
.split-grid { display: grid; gap: 16px; grid-template-columns: 1.2fr 0.8fr; }
.stack { display: grid; gap: 16px; }
.card-head { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-bottom: 18px; }
.card-head h2 { margin-bottom: 0; }
.chip-row { display: flex; flex-wrap: wrap; gap: 10px; }
.chip { display: inline-flex; align-items: center; gap: 8px; padding: 10px 12px; border-radius: 999px; background: #f8fafc; border: 1px solid var(--border); font-size: 12.5px; color: var(--text-secondary); }
.editor-layout { display: grid; gap: 16px; grid-template-columns: 260px 1fr; align-items: start; }
.template-task-list { display: grid; gap: 12px; }
.template-task-card { padding: 18px; border: 1px solid var(--border); border-radius: var(--radius-md); background: #fbfcfe; display: grid; gap: 12px; }
.template-toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; }
.template-toolbar .button { white-space: nowrap; }
.hint-box { padding: 14px 16px; border-radius: var(--radius-md); background: #f8fafc; border: 1px solid var(--border); color: var(--text-secondary); font-size: 12.5px; }

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

/* ── Kanban Board ── */
.board-filters { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; align-items: center; }
.board-filters select, .board-filters input { width: auto; padding: 7px 11px; font-size: 13px; }
.board-wrap { overflow-x: auto; padding-bottom: 24px; }
.board { display: flex; gap: 14px; min-width: max-content; align-items: flex-start; }
.board-col { width: 268px; flex-shrink: 0; display: flex; flex-direction: column; background: #f8f9fb; border-radius: var(--radius-lg); border: 1px solid var(--border); overflow: hidden; }
.col-header { display: flex; align-items: center; gap: 8px; padding: 13px 14px 11px; border-bottom: 1px solid var(--border); background: var(--surface); }
.col-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.col-title { font-size: 12.5px; font-weight: 700; letter-spacing: -.01em; flex: 1; }
.col-count { font-size: 11px; font-weight: 600; color: var(--text-muted); background: var(--bg); padding: 1px 7px; border-radius: 999px; border: 1px solid var(--border); }
.col-cards { padding: 10px; display: flex; flex-direction: column; gap: 8px; min-height: 60px; }
.task-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px 13px; transition: box-shadow 0.15s, border-color 0.15s; position: relative; overflow: hidden; }
.task-card:hover { box-shadow: var(--shadow-md); border-color: var(--border-strong); }
.task-card-strip { height: 3px; border-radius: 999px; margin-bottom: 10px; }
.task-card-title { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 3px; line-height: 1.4; }
.task-card-client { font-size: 11.5px; color: var(--text-muted); margin-bottom: 10px; }
.task-card-footer { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
.task-card-footer select { padding: 3px 6px; font-size: 11px; border-radius: 4px; width: auto; }
.due-label { font-size: 11px; color: var(--text-muted); }
.empty-col { font-size: 12px; color: var(--text-muted); text-align: center; padding: 18px 10px; }

/* ── Calendar ── */
.cal-page { display:flex; flex-direction:column; height:100vh; overflow:hidden; }
.cal-toolbar { display:flex; align-items:center; gap:10px; padding:13px 24px; border-bottom:1px solid var(--border); background:var(--surface); flex-shrink:0; flex-wrap:wrap; }
.cal-toolbar h1 { font-size:18px; font-weight:700; letter-spacing:-.02em; margin-right:4px; }
.cal-nav-btn { background:var(--surface); color:var(--text); border:1px solid var(--border-strong); padding:5px 12px; font-size:15px; line-height:1; border-radius:var(--radius-sm); cursor:pointer; }
.cal-nav-btn:hover { background:var(--bg); }
.cal-week-label { font-size:14px; font-weight:600; min-width:200px; text-align:center; }
.cal-head { display:grid; grid-template-columns:56px repeat(7,1fr); border-bottom:1px solid var(--border); background:var(--surface); flex-shrink:0; }
.cal-head-spacer { border-right:1px solid var(--border); }
.cal-head-cell { padding:9px 6px 8px; text-align:center; border-right:1px solid var(--border); }
.cal-head-cell .dow { font-size:10.5px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:.06em; }
.cal-head-cell .dom { font-size:22px; font-weight:300; color:var(--text); line-height:1.2; width:34px; height:34px; display:flex; align-items:center; justify-content:center; border-radius:50%; margin:3px auto 0; }
.cal-head-cell.today .dom { background:var(--primary); color:#fff; font-weight:700; }
.cal-body { display:flex; flex:1; overflow-y:auto; min-height:0; }
.cal-time-col { width:56px; flex-shrink:0; }
.cal-time-label { height:64px; display:flex; align-items:flex-start; justify-content:flex-end; padding:4px 8px 0 0; font-size:10.5px; color:var(--text-muted); font-weight:500; white-space:nowrap; }
.cal-days { display:grid; grid-template-columns:repeat(7,1fr); flex:1; border-left:1px solid var(--border); }
.cal-day-col { border-right:1px solid var(--border); position:relative; cursor:pointer; }
.cal-slot { height:32px; border-bottom:1px solid rgba(0,0,0,.04); box-sizing:border-box; }
.cal-slot.hour-line { border-bottom:1px solid var(--border); }
.cal-event { position:absolute; left:2px; right:2px; border-radius:4px; padding:3px 6px; font-size:11.5px; font-weight:600; color:#fff; overflow:hidden; cursor:pointer; z-index:1; line-height:1.3; box-sizing:border-box; }
.cal-event:hover { filter:brightness(.88); }
.cal-event-title { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.cal-event-time { font-size:10px; opacity:.82; font-weight:400; }
/* Scheduling assistant */
.sched-wrap { border:1px solid var(--border); border-radius:var(--radius-md); overflow:hidden; }
.sched-time-axis { display:flex; padding-left:88px; background:var(--bg); border-bottom:1px solid var(--border); }
.sched-axis-tick { font-size:10px; color:var(--text-muted); flex:1; padding:4px 0; text-align:left; }
.sched-row { display:flex; align-items:center; border-bottom:1px solid var(--border); }
.sched-row:last-child { border-bottom:none; }
.sched-row-label { width:88px; flex-shrink:0; font-size:11.5px; font-weight:600; padding:6px 10px; color:var(--text); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sched-row-track { flex:1; position:relative; height:32px; min-width:0; }
.sched-block { position:absolute; top:3px; bottom:3px; border-radius:3px; }
/* Availability modal */
.avail-row { display:flex; align-items:center; gap:12px; padding:9px 0; border-bottom:1px solid var(--border); }
.avail-row:last-child { border-bottom:none; }
.avail-day { width:38px; font-size:12.5px; font-weight:600; flex-shrink:0; }
/* Tab strip */
.tab-strip { display:flex; gap:0; border-bottom:1px solid var(--border); margin-bottom:16px; }
.tab-btn { padding:8px 16px; font-size:13px; font-weight:500; color:var(--text-muted); border:none; background:none; cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-1px; }
.tab-btn.active-tab { color:var(--primary); border-bottom-color:var(--primary); font-weight:600; }

/* ── Mobile ── */
@media (max-width: 900px) {
  .sidebar { transform: translateX(-100%); transition: transform 0.2s; }
  .sidebar.open { transform: translateX(0); }
  .main { margin-left: 0; }
  .page-shell { padding: 16px; }
  .stat-grid, .split-grid, .editor-layout { grid-template-columns: 1fr; }
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


class SEOTaskCreate(BaseModel):
    client_id: int
    task_name: str = Field(..., min_length=1)
    task_description: Optional[str] = None
    category: str
    status: str = "Not Started"
    assigned_to_username: Optional[str] = None
    due_date: Optional[str] = None

    @validator("category")
    def validate_category(cls, value: str) -> str:
        valid = ["keyword research", "on-page", "off-page", "technical", "extras"]
        if value not in valid:
            raise ValueError(f"category must be one of {valid}")
        return value

    @validator("status")
    def validate_status(cls, value: str) -> str:
        if value not in SEO_TASK_STATUSES:
            raise ValueError(f"status must be one of {SEO_TASK_STATUSES}")
        return value


class SEOTemplateTaskUpdate(BaseModel):
    id: Optional[int] = None
    category: str
    task_name: str = Field(..., min_length=1)
    task_description: Optional[str] = None
    sort_order: int = 0
    default_assignee: Optional[str] = None
    is_active: bool = True

    @validator("category")
    def validate_category(cls, value: str) -> str:
        valid = ["keyword research", "on-page", "off-page", "technical", "extras"]
        if value not in valid:
            raise ValueError(f"category must be one of {valid}")
        return value


class SEOTemplateTierUpdate(BaseModel):
    tasks: List[SEOTemplateTaskUpdate]


class CalendarEventCreate(BaseModel):
    title: str = Field(..., min_length=1)
    description: Optional[str] = None
    start_datetime: str
    end_datetime: str
    is_all_day: bool = False
    color: str = "#2563eb"
    attendee_employee_ids: List[int] = []


class AvailabilitySlot(BaseModel):
    day_of_week: int
    start_time: str = "09:00"
    end_time: str = "17:00"
    is_working: bool = True


class AvailabilityUpdate(BaseModel):
    slots: List[AvailabilitySlot]


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


def clean_username_name(username: str) -> str:
    username = (username or "").strip()
    if not username:
        return ""
    cleaned = re.sub(r"[_\-.]+", " ", username)
    cleaned = re.sub(r"\d+$", "", cleaned).strip()
    if not cleaned:
        cleaned = username
    return " ".join(part.capitalize() for part in cleaned.split())


def user_display_name(user) -> str:
    if not user:
        return ""
    first_name = (user["first_name"] or "").strip()
    last_name = (user["last_name"] or "").strip()
    username = (user["username"] or "").strip()
    if first_name.lower() == "admin" and last_name.lower() == "user" and username:
        return clean_username_name(username)
    full_name = f"{first_name} {last_name}".strip()
    return full_name or clean_username_name(username)


def user_first_name(user) -> str:
    if not user:
        return ""
    first_name = (user["first_name"] or "").strip()
    username = (user["username"] or "").strip()
    if first_name and first_name.lower() != "admin":
        return first_name
    cleaned = clean_username_name(username)
    return cleaned.split()[0] if cleaned else first_name


def sidebar_html(active: str, user) -> str:
    display_name = user_display_name(user)
    initials = "".join(part[0] for part in display_name.split()[:2]).upper() if display_name else "?"
    name = display_name
    role = user["role"].capitalize() if user else ""
    nav_icon = {
        "dashboard": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>',
        "clients":   '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        "tasks":     '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>',
        "calendar":  '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>',
        "logout":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>',
    }
    def nav_link(key, label, href):
        cls = "active" if active == key else ""
        return f"<a href='{href}' class='{cls}'>{nav_icon[key]}{label}</a>"
    client_active = active in {"clients", "client-add", "templates"}
    client_tree_class = "nav-tree is-active" if client_active else "nav-tree"
    client_tree_open = "open" if client_active else ""
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
        <details class='{client_tree_class}' {client_tree_open}>
          <summary>
            <span class='nav-tree-label'>{nav_icon['clients']}Clients</span>
            <span class='nav-tree-caret'>&#8250;</span>
          </summary>
          <div class='nav-tree-links'>
            <a href='/SEO/clients/add' class='{'active' if active == 'client-add' else ''}'>Add client</a>
            <a href='/SEO/clients' class='{'active' if active == 'clients' else ''}'>View clients</a>
            <a href='/SEO/templates' class='{'active' if active == 'templates' else ''}'>Template editor</a>
          </div>
        </details>
        {nav_link('tasks', 'Tasks', '/SEO/tasks')}
        {nav_link('calendar', 'Calendar', '/SEO/calendar')}
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
        task_total = conn.execute("SELECT COUNT(*) FROM seo_client_tasks").fetchone()[0]
        done_count = conn.execute("SELECT COUNT(*) FROM seo_client_tasks WHERE status = 'Done'").fetchone()[0]
        blocked_count = conn.execute("SELECT COUNT(*) FROM seo_client_tasks WHERE status = 'Blocked'").fetchone()[0]
        overdue_count = conn.execute("SELECT COUNT(*) FROM seo_client_tasks WHERE due_date IS NOT NULL AND due_date < DATE('now') AND status != 'Done'").fetchone()[0]
        due_soon_count = conn.execute("SELECT COUNT(*) FROM seo_client_tasks WHERE due_date IS NOT NULL AND due_date BETWEEN DATE('now') AND DATE('now', '+7 day') AND status != 'Done'").fetchone()[0]
        completed_this_week = conn.execute("SELECT COUNT(*) FROM seo_client_tasks WHERE completed_at IS NOT NULL AND DATE(completed_at) >= DATE('now', '-6 day')").fetchone()[0]
        in_progress_count = conn.execute("SELECT COUNT(*) FROM seo_client_tasks WHERE status = 'In Progress'").fetchone()[0]
        client_count = conn.execute("SELECT COUNT(*) FROM seo_clients").fetchone()[0]
        tier_rows = conn.execute("SELECT service_tier, COUNT(*) AS total FROM seo_clients GROUP BY service_tier ORDER BY service_tier").fetchall()
        assignee_rows = conn.execute(
            """
            SELECT assigned_to_username, COUNT(*) AS total,
                   SUM(CASE WHEN status = 'Done' THEN 1 ELSE 0 END) AS done_total
            FROM seo_client_tasks
            WHERE assigned_to_username IS NOT NULL AND TRIM(assigned_to_username) != ''
            GROUP BY assigned_to_username
            ORDER BY total DESC, assigned_to_username ASC
            """
        ).fetchall()
        return {
            "client_count": client_count,
            "task_count": task_total,
            "blocked_count": blocked_count,
            "done_count": done_count,
            "overdue_count": overdue_count,
            "due_soon_count": due_soon_count,
            "completed_this_week": completed_this_week,
            "in_progress_count": in_progress_count,
            "completion_pct": round((done_count / task_total) * 100) if task_total else 0,
            "tier_rows": [dict(row) for row in tier_rows],
            "assignee_rows": [dict(row) for row in assignee_rows],
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
    first_name = user_first_name(user)
    team_rows = metrics["assignee_rows"][:4]
    team_html = "".join(
        f"""
        <div class='insight-row'>
          <span class='chip'><span class='av'>{row['assigned_to_username'][0].upper()}</span>{row['assigned_to_username']}</span>
          <div class='progress-track'><div class='progress-bar' style='width:{round((row['done_total'] / row['total']) * 100) if row['total'] else 0}%;'></div></div>
          <strong>{round((row['done_total'] / row['total']) * 100) if row['total'] else 0}%</strong>
        </div>
        """
        for row in team_rows
    ) or "<p class='muted'>No team assignments yet.</p>"
    tier_html = "".join(
        f"<div class='chip'><strong>{row['service_tier']}</strong><span>{row['total']} clients</span></div>"
        for row in metrics["tier_rows"]
    ) or "<p class='muted'>No client tiers yet.</p>"
    return f"""
    <div class='page-shell'>
      <div class='card hero-card'>
        <div class='page-header'>
          <h1>Dashboard</h1>
          <p>Good morning, {first_name}.</p>
        </div>
        <div class='stat-grid'>
          <div class='metric'>
            <div class='label'>Active clients</div>
            <div class='value'>{metrics['client_count']}</div>
            <div class='metric-subtle'>Across all service tiers</div>
          </div>
          <div class='metric'>
            <div class='label'>Overall completion</div>
            <div class='value'>{metrics['completion_pct']}%</div>
            <div class='metric-subtle'>{metrics['done_count']} of {metrics['task_count']} tasks done</div>
          </div>
          <div class='metric'>
            <div class='label'>Open workload</div>
            <div class='value'>{max(metrics['task_count'] - metrics['done_count'], 0)}</div>
            <div class='metric-subtle'>{metrics['in_progress_count']} currently in progress</div>
          </div>
          <div class='metric'>
            <div class='label'>Overdue tasks</div>
            <div class='value'>{metrics['overdue_count']}</div>
            <div class='metric-subtle'>{metrics['due_soon_count']} due in the next 7 days</div>
          </div>
        </div>
      </div>
      <div class='split-grid'>
        <div class='stack'>
          <div class='card'>
            <div class='card-head'>
              <div>
                <h2>Delivery pulse</h2>
                <p class='muted'>Pure aggregate indicators, no client-by-client breakdown on the homepage.</p>
              </div>
              <span class='badge {'red' if metrics['blocked_count'] else 'green'}'>{metrics['blocked_count']} blocked</span>
            </div>
            <div class='insight-list'>
              <div class='insight-row'><span>Tasks completed this week</span><strong>{metrics['completed_this_week']}</strong></div>
              <div class='insight-row'><span>Tasks in progress</span><strong>{metrics['in_progress_count']}</strong></div>
              <div class='insight-row'><span>Tasks due soon</span><strong>{metrics['due_soon_count']}</strong></div>
              <div class='insight-row'><span>Blocked work</span><strong>{metrics['blocked_count']}</strong></div>
            </div>
          </div>
          <div class='card'>
            <div class='card-head'>
              <div>
                <h2>Team progress</h2>
                <p class='muted'>Completion by assignee, based on total assigned tasks.</p>
              </div>
            </div>
            <div class='insight-list'>{team_html}</div>
          </div>
        </div>
        <div class='stack'>
          <div class='card'>
            <div class='card-head'>
              <div>
                <h2>Tier mix</h2>
                <p class='muted'>How your current client load is distributed.</p>
              </div>
            </div>
            <div class='chip-row'>{tier_html}</div>
          </div>
          <div class='card'>
            <div class='card-head'>
              <div>
                <h2>Workspace focus</h2>
                <p class='muted'>A quick aggregate read before drilling into clients, templates, or tasks.</p>
              </div>
            </div>
            <div class='hint-box'>
              Keep the dashboard high-level. Client management, add client, and template editing now live under the Clients navigation tree for a cleaner workflow.
            </div>
          </div>
        </div>
      </div>
    </div>
    """


def seo_add_client_html() -> str:
    tier_options = "".join(f"<option value='{tier}'>{tier}</option>" for tier in SEO_SERVICE_TIERS)
    return f"""
    <div class='page-shell'>
      <div class='page-header'>
        <h1>Add client</h1>
        <p>Create a new SEO client and automatically generate the tier-based task set.</p>
      </div>
      <div class='grid two'>
        <div class='card'>
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
            <label>WordPress / website credentials<textarea name='credentials' placeholder='Plain text storage for now, per current requirement.'></textarea></label>
            <label>Notes<textarea name='notes'></textarea></label>
            <div style='display:flex;justify-content:flex-end;gap:10px;'>
              <a href='/SEO/clients' class='button secondary'>Cancel</a>
              <button type='submit'>Create client and tasks</button>
            </div>
          </form>
        </div>
        <div class='stack'>
          <div class='card'>
            <h2>What happens next</h2>
            <div class='hint-box'>
              Once a client is created, the system copies the active task template for the selected tier and assigns default owners where available.
            </div>
          </div>
          <div class='card'>
            <h2>Suggested workflow</h2>
            <div class='insight-list'>
              <div class='insight-row'><span>1. Add the client basics</span><strong>Now</strong></div>
              <div class='insight-row'><span>2. Review generated tasks</span><strong>Tasks board</strong></div>
              <div class='insight-row'><span>3. Adjust template rules later</span><strong>Template editor</strong></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <script>
      async function createClient(event) {{
        event.preventDefault();
        const payload = Object.fromEntries(new FormData(event.target).entries());
        const res = await fetch('/api/seo/clients', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify(payload) }});
        const data = await res.json();
        if (!res.ok) {{ alert(data.detail || 'Unable to create client'); return; }}
        window.location.href = '/SEO/clients';
      }}
      document.addEventListener('DOMContentLoaded', () => {{
        document.getElementById('clientForm').addEventListener('submit', createClient);
      }});
    </script>
    """


def seo_template_editor_html() -> str:
    tier_options = "".join(f"<option value='{tier}'>{tier}</option>" for tier in SEO_SERVICE_TIERS)
    return f"""
    <div class='page-shell'>
      <div class='page-header'>
        <h1>Template editor</h1>
        <p>Adjust tier templates before pushing the updated task bundle into future client setups.</p>
      </div>
      <div class='editor-layout'>
        <div class='stack'>
          <div class='card'>
            <h2>Template scope</h2>
            <label>Service tier<select id='templateTier'>{tier_options}</select></label>
            <div class='hint-box' style='margin-top:14px;'>
              Changes here update the saved template for future clients. Existing client task lists stay untouched unless you deliberately edit them in the tasks area.
            </div>
          </div>
          <div class='card'>
            <h2>Editor tips</h2>
            <div class='insight-list'>
              <div class='insight-row'><span>Keep names concise</span><strong>Better scanability</strong></div>
              <div class='insight-row'><span>Use sort order gaps</span><strong>Easier reordering</strong></div>
              <div class='insight-row'><span>Disable instead of delete</span><strong>Safer rollout</strong></div>
            </div>
          </div>
        </div>
        <div class='stack'>
          <div class='card'>
            <div class='template-toolbar'>
              <div>
                <h2>Task bundle</h2>
                <p class='muted'>Edit names, categories, assignees, and activation state.</p>
              </div>
              <div style='display:flex;gap:10px;'>
                <button type='button' class='secondary' id='addTemplateTaskBtn'>Add task</button>
                <button type='button' id='saveTemplateBtn'>Save template</button>
              </div>
            </div>
            <div id='templateTaskList' class='template-task-list'></div>
          </div>
        </div>
      </div>
    </div>
    <script>
      const CATEGORIES = ['keyword research', 'on-page', 'off-page', 'technical', 'extras'];
      let templateTasks = [];
      function taskCard(task, index) {{
        return `
          <div class='template-task-card' data-index='${{index}}'>
            <div class='grid two'>
              <label>Task name<input data-field='task_name' value="${{(task.task_name || '').replace(/"/g, '&quot;')}}"></label>
              <label>Category
                <select data-field='category'>
                  ${{CATEGORIES.map(cat => `<option value="${{cat}}" ${{cat === task.category ? 'selected' : ''}}>${{cat}}</option>`).join('')}}
                </select>
              </label>
              <label>Default assignee<input data-field='default_assignee' value="${{(task.default_assignee || '').replace(/"/g, '&quot;')}}"></label>
              <label>Sort order<input data-field='sort_order' type='number' value='${{task.sort_order ?? 0}}'></label>
            </div>
            <label>Task description<textarea data-field='task_description'>${{task.task_description || ''}}</textarea></label>
            <div style='display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;'>
              <label style='flex-direction:row;align-items:center;gap:8px;color:var(--text);'>
                <input data-field='is_active' type='checkbox' style='width:auto;' ${{task.is_active ? 'checked' : ''}}>
                Active in future client rollouts
              </label>
              <button type='button' class='secondary' onclick='removeTemplateTask(${{index}})'>Remove</button>
            </div>
          </div>`;
      }}
      function renderTemplateTasks() {{
        const container = document.getElementById('templateTaskList');
        container.innerHTML = templateTasks.length
          ? templateTasks.map((task, index) => taskCard(task, index)).join('')
          : "<div class='hint-box'>No tasks yet for this tier. Add one to start building the template.</div>";
        container.querySelectorAll('.template-task-card').forEach(card => {{
          const index = Number(card.dataset.index);
          card.querySelectorAll('[data-field]').forEach(el => {{
            el.addEventListener('input', () => updateTemplateField(index, el.dataset.field, el.type === 'checkbox' ? el.checked : el.value));
            el.addEventListener('change', () => updateTemplateField(index, el.dataset.field, el.type === 'checkbox' ? el.checked : el.value));
          }});
        }});
      }}
      function updateTemplateField(index, field, value) {{
        templateTasks[index][field] = field === 'sort_order' ? Number(value || 0) : value;
      }}
      function removeTemplateTask(index) {{
        templateTasks.splice(index, 1);
        renderTemplateTasks();
      }}
      function addTemplateTask() {{
        templateTasks.push({{ task_name: '', category: 'keyword research', task_description: '', sort_order: (templateTasks.length + 1) * 10, default_assignee: '', is_active: true }});
        renderTemplateTasks();
      }}
      async function loadTemplateTier() {{
        const tier = document.getElementById('templateTier').value;
        const res = await fetch(`/api/seo/templates/${{encodeURIComponent(tier)}}`);
        templateTasks = await res.json();
        renderTemplateTasks();
      }}
      async function saveTemplateTier() {{
        const tier = document.getElementById('templateTier').value;
        const res = await fetch(`/api/seo/templates/${{encodeURIComponent(tier)}}`, {{
          method: 'PUT',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ tasks: templateTasks }})
        }});
        const data = await res.json();
        if (!res.ok) {{ alert(data.detail || 'Unable to save template'); return; }}
        alert(`Saved ${{data.saved_count}} tasks for ${{tier}}.`);
        await loadTemplateTier();
      }}
      document.addEventListener('DOMContentLoaded', () => {{
        document.getElementById('templateTier').addEventListener('change', loadTemplateTier);
        document.getElementById('addTemplateTaskBtn').addEventListener('click', addTemplateTask);
        document.getElementById('saveTemplateBtn').addEventListener('click', saveTemplateTier);
        loadTemplateTier();
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


def seo_tasks_html(user) -> str:
    status_options_js = json.dumps(SEO_TASK_STATUSES)
    categories = [
        ("keyword research", "Keyword Research", "#8b5cf6"),
        ("on-page",          "On-Page",          "#2563eb"),
        ("off-page",         "Off-Page",          "#16a34a"),
        ("technical",        "Technical",         "#ea580c"),
        ("extras",           "Extras",            "#6b7280"),
    ]
    cols_js = json.dumps([{"key": k, "label": l, "color": c} for k, l, c in categories])
    current_username = ((user["username"] or "") if user else "").strip().lower()
    return f"""
    <div class='page-shell' style='max-width:none;'>
      <div class='page-header'>
        <h1>Tasks</h1>
        <p>Kanban board — all client tasks by category.</p>
      </div>
      <div class='board-filters'>
        <input id='boardSearch' placeholder='Search tasks or clients...' style='width:220px;'>
        <select id='clientFilter'><option value=''>All clients</option></select>
        <select id='statusFilter'><option value=''>All statuses</option>
          {''.join(f"<option value='{s}'>{s}</option>" for s in SEO_TASK_STATUSES)}
        </select>
        <select id='assigneeFilter'><option value=''>All assignees</option></select>
        <button class='btn btn-primary' onclick='openNewTaskModal()' style='margin-left:auto;'>+ New Task</button>
      </div>
      <div class='board-wrap'>
        <div class='board' id='board'></div>
      </div>
    </div>

    <!-- New Task Modal -->
    <div id='taskModal' style='display:none;position:fixed;inset:0;background:rgba(0,0,0,0.45);z-index:1000;align-items:center;justify-content:center;'>
      <div style='background:var(--surface);border-radius:var(--radius-lg);padding:28px 32px;width:460px;max-width:95vw;box-shadow:var(--shadow-lg);'>
        <h3 style='margin:0 0 20px;font-size:16px;'>New Task</h3>
        <form id='newTaskForm' onsubmit='submitNewTask(event)'>
          <div style='display:flex;flex-direction:column;gap:13px;'>
            <div>
              <label class='form-label'>Task Name *</label>
              <input name='task_name' required placeholder='e.g. Homepage meta update'>
            </div>
            <div>
              <label class='form-label'>Description</label>
              <input name='task_description' placeholder='Optional details'>
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
              <div>
                <label class='form-label'>Client *</label>
                <select name='client_id' id='modalClientSel' required></select>
              </div>
              <div>
                <label class='form-label'>Category *</label>
                <select name='category' required>
                  {''.join(f"<option value='{c}'>{c.title()}</option>" for c in ['keyword research','on-page','off-page','technical','extras'])}
                </select>
              </div>
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
              <div>
                <label class='form-label'>Assignee</label>
                <select name='assigned_to_username' id='modalAssigneeSel'>
                  <option value=''>Unassigned</option>
                </select>
              </div>
              <div>
                <label class='form-label'>Status</label>
                <select name='status'>
                  {''.join(f"<option value='{s}'>{s}</option>" for s in SEO_TASK_STATUSES)}
                </select>
              </div>
            </div>
            <div>
              <label class='form-label'>Due Date</label>
              <input name='due_date' type='date'>
            </div>
          </div>
          <div style='display:flex;gap:10px;justify-content:flex-end;margin-top:22px;'>
            <button type='button' class='btn' onclick='closeNewTaskModal()'>Cancel</button>
            <button type='submit' class='btn btn-primary' id='taskSubmitBtn'>Create Task</button>
          </div>
        </form>
      </div>
    </div>
    <script>
      const STATUSES = {status_options_js};
      const COLS = {cols_js};
      const CURRENT_USERNAME = {json.dumps(current_username)};

      const STRIP_COLOR = {{'Not Started':'#e2e5ea','In Progress':'#2563eb','Done':'#16a34a','Blocked':'#dc2626'}};

      function assigneeChip(u) {{
        if (!u) return '<span class="muted" style="font-size:11px">Unassigned</span>';
        const colors = {{'lirha':'#8b5cf6','ivan':'#16a34a','alcuin':'#2563eb'}};
        const bg = colors[u] || '#6b7280';
        return `<span class="assignee-chip"><span class="av" style="background:${{bg}}">${{u[0].toUpperCase()}}</span>${{u}}</span>`;
      }}

      let allTasks = [], allClients = [], allEmployees = [];

      async function load() {{
        const [tr, cr, er] = await Promise.all([fetch('/api/seo/tasks'), fetch('/api/seo/clients'), fetch('/api/calendar/employees')]);
        allTasks = await tr.json();
        allClients = await cr.json();
        allEmployees = await er.json();
        const clientSel = document.getElementById('clientFilter');
        allClients.forEach(c => {{ const o = document.createElement('option'); o.value = c.id; o.textContent = c.company_name; clientSel.appendChild(o); }});
        const assignees = [...new Set(allTasks.map(t => t.assigned_to_username).filter(Boolean))];
        const aSel = document.getElementById('assigneeFilter');
        assignees.forEach(a => {{ const o = document.createElement('option'); o.value = a; o.textContent = a; aSel.appendChild(o); }});
        render();
      }}

      function filtered() {{
        const q = document.getElementById('boardSearch').value.toLowerCase();
        const cId = document.getElementById('clientFilter').value;
        const stat = document.getElementById('statusFilter').value;
        const asgn = document.getElementById('assigneeFilter').value;
        return allTasks.filter(t =>
          (!q || t.task_name.toLowerCase().includes(q) || (t.company_name||'').toLowerCase().includes(q)) &&
          (!cId || String(t.client_id) === cId) &&
          (!stat || t.status === stat) &&
          (!asgn || t.assigned_to_username === asgn)
        );
      }}

      function render() {{
        const tasks = filtered();
        document.getElementById('board').innerHTML = COLS.map(col => {{
          const cards = tasks.filter(t => t.category === col.key);
          return `
            <div class='board-col'>
              <div class='col-header'>
                <span class='col-dot' style='background:${{col.color}}'></span>
                <span class='col-title'>${{col.label}}</span>
                <span class='col-count'>${{cards.length}}</span>
              </div>
              <div class='col-cards'>
                ${{cards.length ? cards.map(t => `
                  <div class='task-card'>
                    <div class='task-card-strip' style='background:${{STRIP_COLOR[t.status]||"#e2e5ea"}}'></div>
                    <div class='task-card-title'>${{t.task_name}}</div>
                    <div class='task-card-client'>${{t.company_name}}</div>
                    <div class='task-card-footer'>
                      ${{assigneeChip(t.assigned_to_username)}}
                      <div style='display:flex;align-items:center;gap:6px;'>
                        ${{t.due_date ? `<span class='due-label'>📅 ${{t.due_date}}</span>` : ''}}
                        <select onchange="updateStatus(${{t.id}}, this.value)" style='padding:3px 5px;font-size:11px;border-radius:4px;border:1px solid #e2e5ea;'>
                          ${{STATUSES.map(s => `<option value="${{s}}" ${{s===t.status?'selected':''}}>${{s}}</option>`).join('')}}
                        </select>
                      </div>
                    </div>
                  </div>`).join('') : "<div class='empty-col'>No tasks</div>"}}
              </div>
            </div>`;
        }}).join('');
      }}

      async function updateStatus(id, status) {{
        const res = await fetch('/api/seo/tasks/' + id, {{
          method: 'PATCH', headers: {{'Content-Type':'application/json'}},
          body: JSON.stringify({{status}})
        }});
        if (!res.ok) {{ alert('Update failed'); return; }}
        const updated = await res.json();
        const task = allTasks.find(t => t.id === id);
        if (task) task.status = updated.status;
        render();
      }}

      ['boardSearch','clientFilter','statusFilter','assigneeFilter'].forEach(id => {{
        document.getElementById(id).addEventListener('input', render);
        document.getElementById(id).addEventListener('change', render);
      }});

      function openNewTaskModal() {{
        const modal = document.getElementById('taskModal');
        modal.style.display = 'flex';
        const sel = document.getElementById('modalClientSel');
        sel.innerHTML = '<option value="">Select client...</option>';
        allClients.forEach(c => {{ const o = document.createElement('option'); o.value = c.id; o.textContent = c.company_name; sel.appendChild(o); }});
        const assigneeSel = document.getElementById('modalAssigneeSel');
        assigneeSel.innerHTML = '<option value="">Unassigned</option>';
        if (CURRENT_USERNAME) {{
          const me = document.createElement('option');
          me.value = CURRENT_USERNAME;
          me.textContent = 'Assign to me';
          assigneeSel.appendChild(me);
        }}
        allEmployees
          .filter(emp => (emp.username || '').toLowerCase() !== CURRENT_USERNAME)
          .forEach(emp => {{
            const o = document.createElement('option');
            o.value = emp.username;
            o.textContent = emp.username;
            assigneeSel.appendChild(o);
          }});
        document.getElementById('newTaskForm').reset();
      }}

      function closeNewTaskModal() {{
        document.getElementById('taskModal').style.display = 'none';
      }}

      document.getElementById('taskModal').addEventListener('click', function(e) {{
        if (e.target === this) closeNewTaskModal();
      }});

      async function submitNewTask(e) {{
        e.preventDefault();
        const btn = document.getElementById('taskSubmitBtn');
        btn.disabled = true; btn.textContent = 'Saving...';
        const fd = new FormData(e.target);
        const body = Object.fromEntries(fd.entries());
        body.client_id = parseInt(body.client_id);
        if (!body.assigned_to_username) delete body.assigned_to_username;
        if (!body.due_date) delete body.due_date;
        if (!body.task_description) delete body.task_description;
        const res = await fetch('/api/seo/tasks', {{
          method: 'POST', headers: {{'Content-Type':'application/json'}},
          body: JSON.stringify(body)
        }});
        btn.disabled = false; btn.textContent = 'Create Task';
        if (!res.ok) {{ alert('Failed to create task'); return; }}
        const newTask = await res.json();
        allTasks.push(newTask);
        closeNewTaskModal();
        render();
      }}

      load();
    </script>
    """


def seo_calendar_html(user) -> str:
    emp_id = user["id"] if user else 0
    username = user["username"] if user else ""
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    avail_rows = "".join(
        f"""<div class='avail-row'>
              <span class='avail-day'>{day_names[i]}</span>
              <label style='display:flex;align-items:center;gap:5px;font-size:12.5px;font-weight:400;cursor:pointer;min-width:90px;flex-direction:row;'>
                <input type='checkbox' name='working_{i}' {'checked' if i < 5 else ''} onchange="document.getElementById('availTimes_{i}').style.opacity=this.checked?'1':'0.35'">
                Working
              </label>
              <div id='availTimes_{i}' style='display:flex;align-items:center;gap:6px;{"" if i < 5 else "opacity:.35;"}'>
                <input type='time' name='start_{i}' value='09:00' style='width:108px;padding:5px 8px;font-size:12.5px;'>
                <span style='color:var(--text-muted);'>–</span>
                <input type='time' name='end_{i}' value='17:00' style='width:108px;padding:5px 8px;font-size:12.5px;'>
              </div>
            </div>"""
        for i in range(7)
    )
    return f"""
    <div class='cal-page'>
      <div class='cal-toolbar'>
        <h1>Calendar</h1>
        <button class='cal-nav-btn' onclick='prevWeek()'>&#8249;</button>
        <span class='cal-week-label' id='weekLabel'></span>
        <button class='cal-nav-btn' onclick='nextWeek()'>&#8250;</button>
        <button class='btn secondary' style='font-size:12.5px;padding:6px 13px;' onclick='goToday()'>Today</button>
        <div style='margin-left:auto;display:flex;gap:8px;'>
          <button class='btn secondary' style='font-size:12.5px;padding:6px 13px;' onclick='openAvailModal()'>My Availability</button>
          <button class='btn btn-primary' style='font-size:12.5px;padding:6px 13px;' onclick='openEventModal()'>+ New Event</button>
        </div>
      </div>
      <div class='cal-head' id='calHead'><div class='cal-head-spacer'></div></div>
      <div class='cal-body'>
        <div class='cal-time-col' id='timeCol'></div>
        <div class='cal-days' id='calDays'></div>
      </div>
    </div>

    <!-- New Event Modal -->
    <div id='eventModal' style='display:none;position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:200;align-items:center;justify-content:center;'>
      <div style='background:var(--surface);border-radius:var(--radius-lg);width:520px;max-width:96vw;max-height:90vh;overflow-y:auto;box-shadow:0 20px 60px rgba(0,0,0,.18);'>
        <div style='padding:22px 26px 0;'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;'>
            <span style='font-size:16px;font-weight:700;'>New Event</span>
            <button onclick='closeEventModal()' style='background:none;border:none;font-size:20px;color:var(--text-muted);cursor:pointer;padding:0;line-height:1;'>&#215;</button>
          </div>
          <div class='tab-strip'>
            <button class='tab-btn active-tab' id='tabBtnDetails' onclick='showTab("details")'>Details</button>
            <button class='tab-btn' id='tabBtnSched' onclick='showTab("sched"); updateSchedAssistant()'>Find a Time</button>
          </div>
        </div>
        <form id='newEventForm' onsubmit='submitEvent(event)' style='display:block;padding:0 26px 22px;'>
          <!-- Details tab -->
          <div id='tabDetails'>
            <div style='display:flex;flex-direction:column;gap:13px;'>
              <div>
                <label class='form-label'>Title *</label>
                <input name='title' required placeholder='Event title' autofocus>
              </div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
                <div>
                  <label class='form-label'>Start Date *</label>
                  <input name='start_date' type='date' required onchange='updateSchedAssistant()'>
                </div>
                <div>
                  <label class='form-label'>End Date</label>
                  <input name='end_date' type='date'>
                </div>
              </div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
                <div>
                  <label class='form-label'>Start Time</label>
                  <input name='start_time' type='time' value='09:00'>
                </div>
                <div>
                  <label class='form-label'>End Time</label>
                  <input name='end_time' type='time' value='10:00'>
                </div>
              </div>
              <div>
                <label class='form-label'>Color</label>
                <div style='display:flex;gap:8px;padding:4px 0;' id='colorPicker'>
                  {''.join(f'<span onclick="selectColor(this,\'{c}\')" style="width:22px;height:22px;border-radius:50%;background:{c};cursor:pointer;border:2px solid transparent;" data-color="{c}"></span>' for c in ['#2563eb','#8b5cf6','#16a34a','#ea580c','#dc2626','#0891b2','#d97706'])}
                </div>
                <input type='hidden' name='color' value='#2563eb' id='colorInput'>
              </div>
              <div>
                <label class='form-label'>Description</label>
                <textarea name='description' placeholder='Optional notes...' style='min-height:60px;'></textarea>
              </div>
              <div>
                <label class='form-label'>Attendees</label>
                <div id='attendeeGrid' style='display:flex;flex-direction:column;gap:6px;padding:4px 0;'></div>
              </div>
            </div>
          </div>
          <!-- Find a Time tab -->
          <div id='tabSched' style='display:none;'>
            <div id='schedContent' style='min-height:120px;'></div>
          </div>
          <div style='display:flex;gap:10px;justify-content:flex-end;margin-top:22px;'>
            <button type='button' class='btn secondary' onclick='closeEventModal()'>Cancel</button>
            <button type='submit' class='btn btn-primary' id='eventSubmitBtn'>Save</button>
          </div>
        </form>
      </div>
    </div>

    <!-- My Availability Modal -->
    <div id='availModal' style='display:none;position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:200;align-items:center;justify-content:center;'>
      <div style='background:var(--surface);border-radius:var(--radius-lg);width:480px;max-width:96vw;box-shadow:0 20px 60px rgba(0,0,0,.18);'>
        <div style='padding:22px 26px 0;'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;'>
            <span style='font-size:16px;font-weight:700;'>My Work Hours</span>
            <button onclick='closeAvailModal()' style='background:none;border:none;font-size:20px;color:var(--text-muted);cursor:pointer;padding:0;line-height:1;'>&#215;</button>
          </div>
        </div>
        <form id='availForm' onsubmit='submitAvailability(event)' style='display:block;padding:0 26px 22px;'>
          {avail_rows}
          <div style='display:flex;gap:10px;justify-content:flex-end;margin-top:20px;'>
            <button type='button' class='btn secondary' onclick='closeAvailModal()'>Cancel</button>
            <button type='submit' class='btn btn-primary' id='availSubmitBtn'>Save</button>
          </div>
        </form>
      </div>
    </div>

    <script>
      const CAL_START = 7, CAL_END = 21, HOUR_PX = 64;
      const DAYS = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
      const CURRENT_EMP_ID = {emp_id};
      const CURRENT_USERNAME = '{username}';

      let weekStart = getThisMonday();
      let allEvents = [], allAvailability = [], allEmployees = [];

      function getThisMonday() {{
        const d = new Date(); d.setHours(0,0,0,0);
        const day = d.getDay();
        d.setDate(d.getDate() - (day === 0 ? 6 : day - 1));
        return d;
      }}
      function addDays(d, n) {{ const x = new Date(d); x.setDate(x.getDate()+n); return x; }}
      function fmtDate(d) {{ return d.toISOString().split('T')[0]; }}
      function fmtLabel(d) {{ return d.toLocaleDateString('en-US', {{month:'short',day:'numeric'}}); }}
      function timeToMins(t) {{ const [h,m]=t.split(':').map(Number); return h*60+m; }}
      function timeToY(t) {{ return (timeToMins(t) - CAL_START*60) / 60 * HOUR_PX; }}
      function dtToY(dt) {{ return timeToY(dt.split('T')[1]||'00:00'); }}
      function dtDurPx(s,e) {{
        const sm=timeToMins(s.split('T')[1]||'00:00'), em=timeToMins(e.split('T')[1]||'00:00');
        return Math.max((em-sm)/60*HOUR_PX, 22);
      }}
      function isToday(d) {{ const t=new Date(); t.setHours(0,0,0,0); return d.getTime()===t.getTime(); }}
      function fmt12(t) {{
        if (!t) return '';
        const [h,m]=t.split(':').map(Number), ap=h<12?'AM':'PM', h12=h===0?12:h>12?h-12:h;
        return `${{h12}}:${{String(m).padStart(2,'0')}} ${{ap}}`;
      }}

      // Load all static data once
      async function loadAll() {{
        const [empR, availR] = await Promise.all([fetch('/api/calendar/employees'), fetch('/api/calendar/availability')]);
        allEmployees = await empR.json();
        allAvailability = await availR.json();
        // Populate attendee checkboxes
        document.getElementById('attendeeGrid').innerHTML = allEmployees.map(e =>
          `<label style='display:flex;align-items:center;gap:6px;font-size:12.5px;font-weight:400;cursor:pointer;flex-direction:row;'>
            <input type='checkbox' name='attendee' value='${{e.employee_id}}' onchange='updateSchedAssistant()'>
            <strong>${{e.username}}</strong> — ${{e.first_name}} ${{e.last_name}}
          </label>`
        ).join('');
        // Pre-fill availability modal from server data
        prefillAvailForm();
      }}

      function prefillAvailForm() {{
        for (let i = 0; i < 7; i++) {{
          const slot = allAvailability.find(a => a.employee_id === CURRENT_EMP_ID && a.day_of_week === i);
          const working = document.querySelector(`[name=working_${{i}}]`);
          const start = document.querySelector(`[name=start_${{i}}]`);
          const end = document.querySelector(`[name=end_${{i}}]`);
          if (slot) {{
            if (working) working.checked = slot.is_working === 1;
            if (start) start.value = slot.start_time;
            if (end) end.value = slot.end_time;
            const times = document.getElementById(`availTimes_${{i}}`);
            if (times) times.style.opacity = (slot.is_working === 1) ? '1' : '0.35';
          }}
        }}
      }}

      // Load events for current week
      async function loadWeek() {{
        const start = fmtDate(weekStart);
        const end = fmtDate(addDays(weekStart, 7));
        const r = await fetch(`/api/calendar/events?start=${{start}}&end=${{end}}`);
        allEvents = await r.json();
        renderCalendar();
      }}

      function renderCalendar() {{
        const totalPx = (CAL_END - CAL_START) * HOUR_PX;
        // Header cells
        const headEl = document.getElementById('calHead');
        headEl.innerHTML = '<div class="cal-head-spacer"></div>' +
          Array.from({{length:7}}, (_,i) => {{
            const d = addDays(weekStart, i);
            const cls = isToday(d) ? 'cal-head-cell today' : 'cal-head-cell';
            return `<div class="${{cls}}"><div class="dow">${{DAYS[i]}}</div><div class="dom">${{d.getDate()}}</div></div>`;
          }}).join('');
        // Week label
        document.getElementById('weekLabel').textContent =
          `${{fmtLabel(weekStart)}} – ${{fmtLabel(addDays(weekStart,6))}}, ${{weekStart.getFullYear()}}`;
        // Time column
        document.getElementById('timeCol').innerHTML =
          Array.from({{length: CAL_END - CAL_START}}, (_,i) => {{
            const h = CAL_START + i;
            const lbl = h===0?'12 AM':h<12?`${{h}} AM`:h===12?'12 PM':`${{h-12}} PM`;
            return `<div class="cal-time-label">${{lbl}}</div>`;
          }}).join('');
        // Day columns
        document.getElementById('calDays').innerHTML = Array.from({{length:7}}, (_,i) => {{
          const d = addDays(weekStart, i);
          const dateStr = fmtDate(d);
          const dayEvs = allEvents.filter(e => e.start_datetime.startsWith(dateStr));
          const slots = Array.from({{length:(CAL_END-CAL_START)*2}}, (_,j) =>
            `<div class="cal-slot${{j%2===0?' hour-line':''}}"></div>`
          ).join('');
          const evBlocks = dayEvs.map(ev => {{
            const top = dtToY(ev.start_datetime);
            const height = dtDurPx(ev.start_datetime, ev.end_datetime);
            const color = ev.color || '#2563eb';
            const st = ev.start_datetime.split('T')[1]||'';
            const et = ev.end_datetime.split('T')[1]||'';
            return `<div class="cal-event" style="top:${{top}}px;height:${{height}}px;background:${{color}};"
                      onclick="evClick(event,${{ev.id}},\`${{ev.title}}\`)">
                      <div class="cal-event-title">${{ev.title}}</div>
                      ${{height>36?`<div class="cal-event-time">${{fmt12(st)}} – ${{fmt12(et)}}</div>`:''}}
                    </div>`;
          }}).join('');
          return `<div class="cal-day-col" style="height:${{totalPx}}px;" onclick="colClick(event,'${{dateStr}}')">${{slots}}${{evBlocks}}</div>`;
        }}).join('');
      }}

      function colClick(e, dateStr) {{
        if (e.target.closest('.cal-event')) return;
        const col = e.currentTarget;
        const rect = col.getBoundingClientRect();
        const y = e.clientY - rect.top;
        const snapMins = CAL_START*60 + Math.round(y/HOUR_PX*60/30)*30;
        const h = Math.floor(snapMins/60), m = snapMins%60;
        const eh = Math.floor((snapMins+60)/60), em = (snapMins+60)%60;
        openEventModal(dateStr, `${{String(h).padStart(2,'0')}}:${{String(m).padStart(2,'0')}}`, `${{String(eh).padStart(2,'0')}}:${{String(em).padStart(2,'0')}}`);
      }}

      function evClick(e, id, title) {{
        e.stopPropagation();
        if (confirm(`Delete "${{title}}"?`)) {{
          fetch('/api/calendar/events/'+id, {{method:'DELETE'}}).then(() => {{
            allEvents = allEvents.filter(x => x.id !== id);
            renderCalendar();
          }});
        }}
      }}

      function prevWeek() {{ weekStart = addDays(weekStart,-7); loadWeek(); }}
      function nextWeek() {{ weekStart = addDays(weekStart,7); loadWeek(); }}
      function goToday() {{ weekStart = getThisMonday(); loadWeek(); }}

      // Color picker
      function selectColor(el, color) {{
        document.querySelectorAll('#colorPicker span').forEach(s => s.style.border='2px solid transparent');
        el.style.border = '2px solid #111';
        document.getElementById('colorInput').value = color;
      }}
      // Select first color on load
      setTimeout(() => {{ const first = document.querySelector('#colorPicker span'); if(first) selectColor(first,'#2563eb'); }}, 100);

      // New Event Modal
      function openEventModal(dateStr, startTime, endTime) {{
        document.getElementById('eventModal').style.display = 'flex';
        if (dateStr) {{
          document.querySelector('[name=start_date]').value = dateStr;
          document.querySelector('[name=end_date]').value = dateStr;
        }}
        if (startTime) document.querySelector('[name=start_time]').value = startTime;
        if (endTime) document.querySelector('[name=end_time]').value = endTime;
        showTab('details');
      }}
      function closeEventModal() {{
        document.getElementById('eventModal').style.display = 'none';
        document.getElementById('newEventForm').reset();
        setTimeout(() => {{ const first = document.querySelector('#colorPicker span'); if(first) selectColor(first,'#2563eb'); }}, 50);
      }}
      document.getElementById('eventModal').addEventListener('click', function(e) {{
        if (e.target === this) closeEventModal();
      }});

      function showTab(tab) {{
        document.getElementById('tabDetails').style.display = tab==='details'?'':'none';
        document.getElementById('tabSched').style.display = tab==='sched'?'':'none';
        document.getElementById('tabBtnDetails').classList.toggle('active-tab', tab==='details');
        document.getElementById('tabBtnSched').classList.toggle('active-tab', tab==='sched');
      }}

      async function submitEvent(e) {{
        e.preventDefault();
        const btn = document.getElementById('eventSubmitBtn');
        btn.disabled = true; btn.textContent = 'Saving...';
        const fd = new FormData(e.target);
        const startDate = fd.get('start_date');
        const endDate = fd.get('end_date') || startDate;
        const startTime = fd.get('start_time') || '09:00';
        const endTime = fd.get('end_time') || '10:00';
        const attendees = [...document.querySelectorAll('[name=attendee]:checked')].map(c => parseInt(c.value));
        const body = {{
          title: fd.get('title'),
          description: fd.get('description') || null,
          start_datetime: `${{startDate}}T${{startTime}}`,
          end_datetime: `${{endDate}}T${{endTime}}`,
          color: fd.get('color') || '#2563eb',
          attendee_employee_ids: attendees,
        }};
        const r = await fetch('/api/calendar/events', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(body)}});
        btn.disabled=false; btn.textContent='Save';
        if (!r.ok) {{ alert('Failed to save'); return; }}
        const ev = await r.json();
        allEvents.push(ev);
        closeEventModal();
        renderCalendar();
      }}

      // Scheduling assistant
      function updateSchedAssistant() {{
        if (document.getElementById('tabSched').style.display === 'none') return;
        const dateStr = document.querySelector('[name=start_date]')?.value;
        const selected = [...document.querySelectorAll('[name=attendee]:checked')].map(c => parseInt(c.value));
        const container = document.getElementById('schedContent');
        if (!selected.length) {{
          container.innerHTML = '<div style="color:var(--text-muted);font-size:13px;padding:20px 0;">Select attendees in the Details tab.</div>';
          return;
        }}
        const SCHED_S=8, SCHED_E=18, PX=42;
        const totalPx = (SCHED_E - SCHED_S) * PX;
        let dayOfWeek = 1;
        if (dateStr) {{
          const d = new Date(dateStr+'T00:00:00');
          dayOfWeek = d.getDay()===0 ? 6 : d.getDay()-1;
        }}
        const ticks = Array.from({{length: SCHED_E-SCHED_S+1}}, (_,i) => {{
          const h = SCHED_S+i;
          return `<div class="sched-axis-tick">${{h<12?h+'AM':h===12?'12PM':(h-12)+'PM'}}</div>`;
        }}).join('');
        const rows = selected.map(empId => {{
          const emp = allEmployees.find(e => e.employee_id===empId);
          if (!emp) return '';
          const avail = allAvailability.find(a => a.employee_id===empId && a.day_of_week===dayOfWeek);
          let availBlock = '';
          if (avail && avail.is_working) {{
            const left = Math.max((timeToMins(avail.start_time)-SCHED_S*60)/60*PX, 0);
            const width = Math.max((timeToMins(avail.end_time)-timeToMins(avail.start_time))/60*PX, 0);
            availBlock = `<div class="sched-block" style="left:${{left}}px;width:${{width}}px;background:#dcfce7;border:1px solid #86efac;"></div>`;
          }}
          const busy = allEvents.filter(ev =>
            dateStr && ev.start_datetime.startsWith(dateStr) &&
            ev.attendees && ev.attendees.some(a => a.employee_id===empId)
          ).map(ev => {{
            const left = Math.max((timeToMins(ev.start_datetime.split('T')[1]||'00:00')-SCHED_S*60)/60*PX, 0);
            const width = Math.max((timeToMins(ev.end_datetime.split('T')[1]||'00:00')-timeToMins(ev.start_datetime.split('T')[1]||'00:00'))/60*PX, 0);
            return `<div class="sched-block" style="left:${{left}}px;width:${{width}}px;background:#dbeafe;border:1px solid #93c5fd;" title="${{ev.title}}"></div>`;
          }}).join('');
          return `<div class="sched-row">
            <div class="sched-row-label">${{emp.username}}</div>
            <div class="sched-row-track" style="width:${{totalPx}}px;">${{availBlock}}${{busy}}</div>
          </div>`;
        }}).join('');
        container.innerHTML = `
          <div class="sched-wrap">
            <div class="sched-time-axis">${{ticks}}</div>
            ${{rows}}
          </div>
          <div style="margin-top:10px;display:flex;gap:14px;font-size:11.5px;color:var(--text-muted);">
            <span style="display:flex;align-items:center;gap:5px;"><span style="width:12px;height:12px;background:#dcfce7;border:1px solid #86efac;border-radius:2px;display:inline-block;"></span>Available</span>
            <span style="display:flex;align-items:center;gap:5px;"><span style="width:12px;height:12px;background:#dbeafe;border:1px solid #93c5fd;border-radius:2px;display:inline-block;"></span>Busy</span>
          </div>`;
      }}

      // Availability modal
      function openAvailModal() {{
        document.getElementById('availModal').style.display = 'flex';
        prefillAvailForm();
      }}
      function closeAvailModal() {{
        document.getElementById('availModal').style.display = 'none';
      }}
      document.getElementById('availModal').addEventListener('click', function(e) {{
        if (e.target === this) closeAvailModal();
      }});

      async function submitAvailability(e) {{
        e.preventDefault();
        const btn = document.getElementById('availSubmitBtn');
        btn.disabled=true; btn.textContent='Saving...';
        const form = e.target;
        const slots = Array.from({{length:7}}, (_,i) => ({{
          day_of_week: i,
          start_time: form.querySelector(`[name=start_${{i}}]`)?.value || '09:00',
          end_time: form.querySelector(`[name=end_${{i}}]`)?.value || '17:00',
          is_working: form.querySelector(`[name=working_${{i}}]`)?.checked || false,
        }}));
        const r = await fetch('/api/calendar/availability', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{slots}})}});
        btn.disabled=false; btn.textContent='Save';
        if (!r.ok) {{ alert('Failed to save'); return; }}
        const ar = await fetch('/api/calendar/availability');
        allAvailability = await ar.json();
        closeAvailModal();
      }}

      loadAll().then(() => loadWeek());
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


@app.get("/SEO/clients/add", response_class=HTMLResponse)
async def seo_add_client_page(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse(f"/login?next={quote('/SEO/clients/add')}", status_code=302)
    return render_app_page("Add client", seo_add_client_html(), "client-add", user)


@app.get("/SEO/templates", response_class=HTMLResponse)
async def seo_templates_page(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse(f"/login?next={quote('/SEO/templates')}", status_code=302)
    return render_app_page("Template editor", seo_template_editor_html(), "templates", user)


@app.get("/SEO/tasks", response_class=HTMLResponse)
async def seo_tasks_page(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse(f"/login?next={quote('/SEO/tasks')}", status_code=302)
    return render_app_page("Tasks", seo_tasks_html(user), "tasks", user)


@app.get("/api/seo/templates")
async def list_seo_templates(request: Request):
    require_user(request)
    with get_connection() as conn:
        rows = conn.execute("SELECT service_tier, COUNT(*) AS task_count FROM seo_task_templates WHERE is_active = 1 GROUP BY service_tier ORDER BY service_tier").fetchall()
        return [dict(row) for row in rows]


@app.get("/api/seo/templates/{service_tier}")
async def get_seo_template_tier(service_tier: str, request: Request):
    require_user(request)
    if service_tier not in SEO_SERVICE_TIERS:
        raise HTTPException(status_code=404, detail="Service tier not found")
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, category, task_name, task_description, sort_order, default_assignee, is_active
            FROM seo_task_templates
            WHERE service_tier = ?
            ORDER BY sort_order, id
            """,
            (service_tier,),
        ).fetchall()
        return [dict(row) for row in rows]


@app.put("/api/seo/templates/{service_tier}")
async def update_seo_template_tier(service_tier: str, request: Request, payload: SEOTemplateTierUpdate):
    require_user(request)
    if service_tier not in SEO_SERVICE_TIERS:
        raise HTTPException(status_code=404, detail="Service tier not found")
    with get_connection() as conn:
        conn.execute("DELETE FROM seo_task_templates WHERE service_tier = ?", (service_tier,))
        for index, task in enumerate(payload.tasks):
            conn.execute(
                """
                INSERT INTO seo_task_templates
                (service_tier, category, task_name, task_description, sort_order, default_assignee, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    service_tier,
                    task.category,
                    task.task_name.strip(),
                    task.task_description,
                    task.sort_order if task.sort_order is not None else (index + 1) * 10,
                    task.default_assignee,
                    1 if task.is_active else 0,
                ),
            )
        conn.commit()
    return {"ok": True, "saved_count": len(payload.tasks)}


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


@app.post("/api/seo/tasks")
async def create_seo_task(request: Request, payload: SEOTaskCreate):
    require_user(request)
    with get_connection() as conn:
        client = conn.execute("SELECT id FROM seo_clients WHERE id = ?", (payload.client_id,)).fetchone()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        max_sort = conn.execute(
            "SELECT COALESCE(MAX(sort_order), 0) FROM seo_client_tasks WHERE client_id = ? AND category = ?",
            (payload.client_id, payload.category),
        ).fetchone()[0]
        cursor = conn.execute(
            """
            INSERT INTO seo_client_tasks
              (client_id, category, task_name, task_description, status, assigned_to_username, due_date, sort_order)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (payload.client_id, payload.category, payload.task_name, payload.task_description,
             payload.status, payload.assigned_to_username, payload.due_date, max_sort + 10),
        )
        conn.commit()
        row = conn.execute(
            "SELECT seo_client_tasks.*, seo_clients.company_name FROM seo_client_tasks JOIN seo_clients ON seo_clients.id = seo_client_tasks.client_id WHERE seo_client_tasks.id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        return dict(row)


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


@app.get("/SEO/calendar", response_class=HTMLResponse)
async def seo_calendar_page(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse(f"/login?next={quote('/SEO/calendar')}", status_code=302)
    return render_app_page("Calendar", seo_calendar_html(user), "calendar", user)


@app.get("/api/calendar/employees")
async def list_calendar_employees(request: Request):
    require_user(request)
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT l.employee_id, l.username, e.first_name, e.last_name
            FROM login l
            JOIN employees e ON e.id = l.employee_id
            WHERE e.is_active = 1
            ORDER BY l.username
            """
        ).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/calendar/events")
async def list_calendar_events(request: Request, start: Optional[str] = None, end: Optional[str] = None):
    require_user(request)
    with get_connection() as conn:
        if start and end:
            events = conn.execute(
                """
                SELECT ce.*, l.username AS created_by_username
                FROM calendar_events ce
                LEFT JOIN login l ON l.employee_id = ce.created_by_employee_id
                WHERE ce.start_datetime >= ? AND ce.start_datetime < ?
                ORDER BY ce.start_datetime
                """,
                (start, end),
            ).fetchall()
        else:
            events = conn.execute(
                """
                SELECT ce.*, l.username AS created_by_username
                FROM calendar_events ce
                LEFT JOIN login l ON l.employee_id = ce.created_by_employee_id
                ORDER BY ce.start_datetime
                """
            ).fetchall()
        result = []
        for ev in events:
            ev_dict = dict(ev)
            attendees = conn.execute(
                """
                SELECT l.username, l.employee_id
                FROM calendar_event_attendees cea
                JOIN login l ON l.employee_id = cea.employee_id
                WHERE cea.event_id = ?
                """,
                (ev_dict["id"],),
            ).fetchall()
            ev_dict["attendees"] = [dict(a) for a in attendees]
            result.append(ev_dict)
        return result


@app.post("/api/calendar/events")
async def create_calendar_event(request: Request, payload: CalendarEventCreate):
    user = require_user(request)
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO calendar_events (title, description, start_datetime, end_datetime, is_all_day, color, created_by_employee_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (payload.title, payload.description, payload.start_datetime, payload.end_datetime,
             int(payload.is_all_day), payload.color, user["id"]),
        )
        event_id = cursor.lastrowid
        for emp_id in payload.attendee_employee_ids:
            conn.execute(
                "INSERT OR IGNORE INTO calendar_event_attendees (event_id, employee_id) VALUES (?, ?)",
                (event_id, emp_id),
            )
        conn.commit()
        ev = dict(conn.execute("SELECT * FROM calendar_events WHERE id = ?", (event_id,)).fetchone())
        attendees = conn.execute(
            "SELECT l.username, l.employee_id FROM calendar_event_attendees cea JOIN login l ON l.employee_id = cea.employee_id WHERE cea.event_id = ?",
            (event_id,),
        ).fetchall()
        ev["attendees"] = [dict(a) for a in attendees]
        return ev


@app.delete("/api/calendar/events/{event_id}")
async def delete_calendar_event(event_id: int, request: Request):
    require_user(request)
    with get_connection() as conn:
        existing = conn.execute("SELECT id FROM calendar_events WHERE id = ?", (event_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Event not found")
        conn.execute("DELETE FROM calendar_events WHERE id = ?", (event_id,))
        conn.commit()
        return {"ok": True}


@app.get("/api/calendar/availability")
async def get_calendar_availability(request: Request):
    require_user(request)
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT ca.*, l.username
            FROM calendar_availability ca
            JOIN login l ON l.employee_id = ca.employee_id
            """
        ).fetchall()
        return [dict(r) for r in rows]


@app.post("/api/calendar/availability")
async def set_calendar_availability(request: Request, payload: AvailabilityUpdate):
    user = require_user(request)
    with get_connection() as conn:
        for slot in payload.slots:
            conn.execute(
                """
                INSERT INTO calendar_availability (employee_id, day_of_week, start_time, end_time, is_working)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(employee_id, day_of_week) DO UPDATE SET
                    start_time = excluded.start_time,
                    end_time = excluded.end_time,
                    is_working = excluded.is_working
                """,
                (user["id"], slot.day_of_week, slot.start_time, slot.end_time, int(slot.is_working)),
            )
        conn.commit()
        return {"ok": True}


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
