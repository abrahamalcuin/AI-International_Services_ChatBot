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

SYSTEM_PROMPT = """You are BYU-Idaho AdvisorBot, a helpful, accurate, and concise student support assistant.

You MUST follow these rules:

1. Answer ONLY using the information contained in the document chunks provided in the prompt.
2. If the answer is not found in the provided documents, say:
   “I could not find this information in my database, please reach out to international@byui.edu or (208) 496-1320.”
3. Never invent rules, deadlines, dates, or policies.
4. When relevant, include the links provided in the document chunks.
5. Answer the question as concisely as possible call it the "Short Answer" and keep it short without losing important information, 
after this summarize other related information in the md into bullet points and ask if they would like to learn more about it. Call this the "Long Answer"
5. Speak clearly, simply, and professionally.
6. Tailor the tone to a student asking for help.
7. Do not reference the concept of “chunks” or “embeddings.”
8. Do not reference the retrieval system.
9. Do not assume anything not explicitly stated in the documents.
10. If a question spans multiple topics, combine the relevant information logically.
11. If there are tips or warnings in the documents, include them briefly.
12. Keep answers detailed, include anything relevant in the md file.
13. Provide the link of the source with “This response is AI generated, please verify information through this link:{insert source link in yaml or sources}  “ 
14. Provide relevant topics that haven’t been tackled and ask the user through bullet points and ask the user if they would like to know more about any of them. 
15. If there are any tables, polish them and make them look clean.
16. Only provide the source at the end, do not do parenthetical citations. 


Always begin reasoning from the content of the provided documents. Use them as your only
knowledge source for final answers.

If the user requests something outside the document scope (e.g., medical, financial advice,
or policy speculation), politely decline and direct them to official BYU-Idaho offices.

"""

PAGE_STYLE = """
<style>
body { font-family: Arial, sans-serif; max-width: 520px; margin: 40px auto; padding: 16px; line-height: 1.5; }
form { display: grid; gap: 12px; }
input { padding: 10px; font-size: 16px; }
button, a.button { padding: 12px; font-size: 16px; cursor: pointer; text-decoration: none; display: inline-block; }
.error { color: #b00020; }
.ok { color: #0a7a2f; }
.card { border: 1px solid #ddd; border-radius: 10px; padding: 18px; box-shadow: 0 2px 10px rgba(0,0,0,.06); }
</style>
"""


class SourceChunk(BaseModel):
    id: str
    category: str
    source: str
    score: float
    text: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Student's natural language question.")
    category: Optional[str] = Field(
        default=None,
        description="Optional focus area: incoming, current, or graduating.",
    )
    top_k: int = Field(
        default=MAX_CONTEXT_CHUNKS,
        ge=1,
        le=MAX_CONTEXT_CHUNKS,
        description="How many context chunks to send to Gemini (capped at 25).",
    )

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
        raise FileNotFoundError(
            f"Embeddings file '{path}' was not found. Run build_embeddings.py first."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for record in data.get("records", []):
        normalized_vector = normalize_vector(record["embedding"])
        records.append(
            EmbeddingRecord(
                id=record["id"],
                category=record["category"].lower(),
                source=record["source"],
                text=record["text"],
                vector=normalized_vector,
            )
        )

    if not records:
        raise RuntimeError(
            f"No embeddings found inside '{path}'. Add Markdown files and rebuild the index."
        )

    return records


class GeminiClient:
    def __init__(self) -> None:
        configure_genai()
        self.generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)

    def embed(self, text: str, task_type: str) -> np.ndarray:
        response = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=text,
            task_type=task_type,
        )
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


def rank_chunks(
    query_vector: np.ndarray, category: Optional[str], top_k: int
) -> List[Tuple[float, EmbeddingRecord]]:
    scored: List[Tuple[float, EmbeddingRecord]] = []
    for record in EMBEDDING_INDEX:
        score = float(np.dot(query_vector, record.vector))
        if category and record.category == category:
            score += CATEGORY_BOOST
        scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[: min(top_k, MAX_CONTEXT_CHUNKS)]


def build_prompt(question: str, scored_chunks: List[Tuple[float, EmbeddingRecord]]) -> str:
    context_sections = []
    for idx, (score, record) in enumerate(scored_chunks, start=1):
        context_sections.append(
            f"Chunk {idx} | Category: {record.category} | Source: {record.source} | Score: {score:.4f}\n{record.text}"
        )

    context_block = "\n\n---\n\n".join(context_sections)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_block}\n\n"
        f"User question: {question.strip()}\n\n"
        "Final answer (reference the relevant chunks by mentioning their sources):"
    )


def render_page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>{PAGE_STYLE}<title>{title}</title></head><body>{body}</body></html>")


def current_user(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT employees.id, employees.first_name, employees.last_name, employees.email, login.username
            FROM employees
            JOIN login ON login.employee_id = employees.id
            WHERE employees.id = ?
            """,
            (user_id,),
        ).fetchone()


app = FastAPI(
    title="BYU-Idaho Student Advisor RAG API",
    version="1.0.0",
    description="Retrieval-Augmented Generation backend that powers the BYU-Idaho chatbot.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax", https_only=True)

EMBEDDING_INDEX: List[EmbeddingRecord] = []
GEMINI_CLIENT: Optional[GeminiClient] = None


@app.on_event("startup")
async def startup_event() -> None:
    global EMBEDDING_INDEX, GEMINI_CLIENT
    init_db()
    GEMINI_CLIENT = GeminiClient()
    EMBEDDING_INDEX = load_embedding_index(EMBEDDINGS_PATH)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/SEO") -> HTMLResponse:
    user = current_user(request)
    if user:
        return RedirectResponse(next, status_code=302)
    return render_page(
        "Login",
        f"<div class='card'><h1>Login</h1><form method='post' action='/login'>"
        f"<input type='hidden' name='next' value='{next}'>"
        "<label>Username</label><input name='username' required>"
        "<label>Password</label><input type='password' name='password' required>"
        "<button type='submit'>Sign in</button></form></div>",
    )


@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/SEO"),
):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT login.password_hash, employees.id, employees.is_active
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

    return render_page(
        "Login",
        f"<div class='card'><h1>Login</h1><p class='error'>Invalid username or password.</p>"
        f"<form method='post' action='/login'><input type='hidden' name='next' value='{next}'>"
        "<label>Username</label><input name='username' required>"
        "<label>Password</label><input type='password' name='password' required>"
        "<button type='submit'>Sign in</button></form></div>",
    )


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

    return render_page(
        "Onboarding",
        f"<div class='card'><h1>Create your account</h1><p>{employee['email']}</p>"
        f"<form method='post' action='/onboarding'><input type='hidden' name='token' value='{token}'>"
        "<label>Username</label><input name='username' minlength='4' maxlength='50' required>"
        "<label>Password</label><input type='password' name='password' minlength='8' required>"
        "<label>Confirm password</label><input type='password' name='confirm_password' minlength='8' required>"
        "<button type='submit'>Create account</button></form></div>",
    )


@app.post("/onboarding", response_class=HTMLResponse)
async def onboarding_submit(
    token: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
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
                "INSERT INTO login (employee_id, username, password_hash) VALUES (?, ?, ?)",
                (employee["id"], username.strip(), password_hash),
            )
            conn.execute(
                "UPDATE employees SET invite_token = NULL, invite_expires_at = NULL WHERE id = ?",
                (employee["id"],),
            )
            conn.commit()
        except Exception as exc:
            if "UNIQUE constraint failed: login.username" in str(exc):
                return render_page("Onboarding", "<div class='card'><p class='error'>That username is already taken.</p><a href='javascript:history.back()'>Go back</a></div>")
            raise

    return render_page("Onboarding", "<div class='card'><h1>Account created</h1><p class='ok'>Your login is ready.</p><a class='button' href='/login'>Go to login</a></div>")


@app.get("/SEO", response_class=HTMLResponse)
async def seo_dashboard(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse(f"/login?next={quote('/SEO')}", status_code=302)
    return render_page(
        "SEO",
        f"<div class='card'><h1>SEO Dashboard</h1><p>Welcome, {user['first_name']} {user['last_name']}.</p><p>You are signed in as <strong>{user['username']}</strong>.</p><p>This route is now protected and prompts for login when visited anonymously.</p><a class='button' href='/logout'>Logout</a></div>",
    )


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    if GEMINI_CLIENT is None or not EMBEDDING_INDEX:
        raise HTTPException(status_code=503, detail="Service is initializing. Retry in a moment.")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question text cannot be empty.")

    query_vector = GEMINI_CLIENT.embed(question, task_type="retrieval_query")
    scored_chunks = rank_chunks(query_vector, payload.category, payload.top_k)

    if not scored_chunks:
        raise HTTPException(
            status_code=500,
            detail="No knowledge chunks are available. Rebuild the embeddings index.",
        )

    prompt = build_prompt(question, scored_chunks)
    answer = await run_in_threadpool(GEMINI_CLIENT.generate_answer, prompt)

    sources = [
        SourceChunk(
            id=record.id,
            category=record.category,
            source=record.source,
            score=round(score, 4),
            text=record.text,
        )
        for score, record in scored_chunks
    ]

    return AskResponse(answer=answer, sources=sources)
