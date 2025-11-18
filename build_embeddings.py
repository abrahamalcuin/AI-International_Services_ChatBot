#!/usr/bin/env python3
"""
Utility script that ingests all Markdown knowledge files, chunks them into
Gemini-friendly passages, and writes their embeddings to embeddings.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import google.generativeai as genai
from dotenv import load_dotenv

EMBEDDING_MODEL = "models/text-embedding-004"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Gemini embeddings for all Markdown files in the knowledge base."
    )
    parser.add_argument(
        "--knowledge-dir",
        default="knowledge",
        type=Path,
        help="Root folder that contains category subfolders with Markdown files.",
    )
    parser.add_argument(
        "--output",
        default="embeddings.json",
        type=Path,
        help="Path where the embeddings JSON should be written.",
    )
    parser.add_argument(
        "--chunk-size",
        default=1200,
        type=int,
        help="Maximum number of characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        default=200,
        type=int,
        help="Number of overlapping characters between consecutive chunks.",
    )
    return parser.parse_args()


def normalize_newlines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Returns the entire Markdown document as a single chunk.

    The chunk_size/overlap parameters are ignored to satisfy the requirement
    that each file maps to exactly one embedding.
    """
    clean_text = normalize_newlines(text)
    return [clean_text] if clean_text else []


def configure_genai() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is required.")
    genai.configure(api_key=api_key)


def embed_chunk(text: str) -> List[float]:
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document",
    )
    embedding = response.get("embedding")
    if not embedding:
        raise RuntimeError("Gemini did not return an embedding vector.")
    return embedding


def iter_markdown_files(base_dir: Path) -> Iterable[Path]:
    for path in sorted(base_dir.rglob("*.md")):
        if path.is_file():
            yield path


def determine_category(base_dir: Path, file_path: Path) -> str:
    relative_path = file_path.relative_to(base_dir)
    return relative_path.parts[0] if relative_path.parts else "uncategorized"


def build_embeddings(
    knowledge_dir: Path, output_path: Path, chunk_size: int, overlap: int
) -> Dict:
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory '{knowledge_dir}' does not exist.")

    records = []
    file_paths = list(iter_markdown_files(knowledge_dir))
    if not file_paths:
        raise FileNotFoundError(
            f"No Markdown files were found inside '{knowledge_dir}'. "
            "Add .md files under the incoming/current/graduating folders."
        )

    print(f"Found {len(file_paths)} Markdown files. Generating embeddings...")

    for file_path in file_paths:
        relative_path = file_path.relative_to(knowledge_dir)
        category = determine_category(knowledge_dir, file_path)
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size, overlap)

        if not chunks:
            continue

        for idx, chunk in enumerate(chunks):
            embedding = embed_chunk(chunk)
            record = {
                "id": f"{relative_path.as_posix()}::chunk-{idx}",
                "category": category,
                "source": relative_path.as_posix(),
                "chunk_index": idx,
                "text": chunk,
                "embedding": embedding,
            }
            records.append(record)

    payload = {
        "model": EMBEDDING_MODEL,
        "chunk_size_chars": chunk_size,
        "chunk_overlap_chars": overlap,
        "record_count": len(records),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_base": knowledge_dir.as_posix(),
        "records": records,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    configure_genai()
    payload = build_embeddings(
        knowledge_dir=args.knowledge_dir,
        output_path=args.output,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    print(
        f"Embedding generation completed. {payload['record_count']} chunks saved to {args.output}."
    )


if __name__ == "__main__":
    main()
