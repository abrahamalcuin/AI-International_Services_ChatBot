from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

load_dotenv()

EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS_PATH", "embeddings.json"))
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
GENERATION_MODEL_NAME = os.getenv("GEMINI_GENERATION_MODEL", "models/gemini-flash-lite-latest")
CATEGORY_BOOST = float(os.getenv("CATEGORY_BOOST", "0.15"))
MAX_CONTEXT_CHUNKS = 25
VALID_CATEGORIES = {"incoming", "current", "graduating"}

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

EMBEDDING_INDEX: List[EmbeddingRecord] = []
GEMINI_CLIENT: Optional[GeminiClient] = None


@app.on_event("startup")
async def startup_event() -> None:
    global EMBEDDING_INDEX, GEMINI_CLIENT
    GEMINI_CLIENT = GeminiClient()
    EMBEDDING_INDEX = load_embedding_index(EMBEDDINGS_PATH)


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
