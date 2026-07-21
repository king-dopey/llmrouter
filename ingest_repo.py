from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models as qdrant_models  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None
    qdrant_models = None

logger = logging.getLogger("router.ingest_repo")

DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "repo_chunks")
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
DEFAULT_EMBEDDING_MODEL = os.getenv("QDRANT_EMBEDDING_MODEL", "qwen3-embedding:4b")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")

SUPPORTED_TEXT_EXTENSIONS = {
    ".md",
    ".rst",
    ".txt",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
}


@dataclass(frozen=True)
class RepoChunk:
    repo: str
    branch: str | None
    path: str
    language: str
    symbol: str | None
    chunk_type: str
    line_start: int
    line_end: int
    chunk_text: str
    imports: list[str]
    hash: str
    updated_at: str | None = None


def _hash_chunk(chunk: RepoChunk) -> str:
    raw = f"{chunk.repo}|{chunk.branch or ''}|{chunk.path}|{chunk.line_start}|{chunk.line_end}|{chunk.chunk_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _repo_name(repo_path: Path) -> str:
    return repo_path.name


def _read_text_lines(file_path: Path) -> list[str]:
    return file_path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _python_imports(tree: ast.AST) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
    return sorted(set(imports))


def _python_chunks(repo: str, file_path: Path, branch: str | None = None) -> list[RepoChunk]:
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _plain_text_chunks(repo, file_path, language="python", branch=branch)

    lines = source.splitlines()
    imports = _python_imports(tree)
    chunks: list[RepoChunk] = []

    def _slice(node: ast.AST) -> tuple[int, int, str]:
        start = max(1, getattr(node, "lineno", 1))
        end = max(start, getattr(node, "end_lineno", start))
        text = "\n".join(lines[start - 1 : end])
        return start, end, text

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start, end, text = _slice(node)
            chunks.append(
                RepoChunk(
                    repo=repo,
                    branch=branch,
                    path=str(file_path.as_posix()),
                    language="python",
                    symbol=node.name,
                    chunk_type="class" if isinstance(node, ast.ClassDef) else "function",
                    line_start=start,
                    line_end=end,
                    chunk_text=text,
                    imports=imports,
                    hash="",
                )
            )

    if not chunks:
        chunks = _plain_text_chunks(repo, file_path, language="python", branch=branch)
    return [RepoChunk(**{**asdict(chunk), "hash": _hash_chunk(chunk)}) for chunk in chunks]


def _plain_text_chunks(repo: str, file_path: Path, *, language: str, branch: str | None = None) -> list[RepoChunk]:
    lines = _read_text_lines(file_path)
    chunks: list[RepoChunk] = []
    window = 80
    step = 60
    start = 1
    while start <= len(lines):
        end = min(len(lines), start + window - 1)
        text = "\n".join(lines[start - 1 : end])
        chunks.append(
            RepoChunk(
                repo=repo,
                branch=branch,
                path=str(file_path.as_posix()),
                language=language,
                symbol=None,
                chunk_type="text",
                line_start=start,
                line_end=end,
                chunk_text=text,
                imports=[],
                hash="",
            )
        )
        if end == len(lines):
            break
        start += step
    return [RepoChunk(**{**asdict(chunk), "hash": _hash_chunk(chunk)}) for chunk in chunks]


def _text_file_chunks(repo: str, file_path: Path, *, branch: str | None = None) -> list[RepoChunk]:
    suffix = file_path.suffix.lower()
    if suffix == ".py":
        return _python_chunks(repo, file_path, branch=branch)
    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        return _plain_text_chunks(repo, file_path, language=suffix.lstrip(".") or "text", branch=branch)
    return []


def collect_repo_chunks(repo_path: str | Path, *, branch: str | None = None) -> list[RepoChunk]:
    repo_dir = Path(repo_path).resolve()
    repo = _repo_name(repo_dir)
    chunks: list[RepoChunk] = []
    for file_path in repo_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part.startswith(".") and part not in {".env"} for part in file_path.parts):
            continue
        chunks.extend(_text_file_chunks(repo, file_path, branch=branch))
    return chunks


async def _ollama_embedding(text: str, *, model: str, base_url: str | None = None) -> list[float]:
    base = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{base}/api/embeddings", json={"model": model, "prompt": text})
        response.raise_for_status()
        data = response.json()
        return [float(value) for value in (data.get("embedding") or [])]


def _qdrant_client(qdrant_url: str | None = None):
    if QdrantClient is None:
        return None
    return QdrantClient(url=qdrant_url or DEFAULT_QDRANT_URL)


def _ensure_collection(client: Any, collection: str, vector_size: int) -> None:
    if qdrant_models is None:
        return
    try:
        client.get_collection(collection)
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE),
        )


async def ingest_repo(
    repo_path: str | Path,
    *,
    collection: str = DEFAULT_COLLECTION,
    qdrant_url: str | None = None,
    embedding_model: str | None = None,
    ollama_base_url: str | None = None,
    branch: str | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    chunks = collect_repo_chunks(repo_path, branch=branch)
    if not chunks:
        return []

    model = embedding_model or DEFAULT_EMBEDDING_MODEL
    embedded_points: list[dict[str, Any]] = []
    for chunk in chunks:
        embedding = await _ollama_embedding(chunk.chunk_text, model=model, base_url=ollama_base_url)
        embedded_points.append(
            {
                "id": chunk.hash,
                "vector": embedding,
                "payload": {
                    "repo": chunk.repo,
                    "branch": chunk.branch,
                    "path": chunk.path,
                    "language": chunk.language,
                    "symbol": chunk.symbol,
                    "chunk_type": chunk.chunk_type,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "imports": chunk.imports,
                    "hash": chunk.hash,
                    "updated_at": chunk.updated_at,
                    "chunk_text": chunk.chunk_text,
                },
            }
        )

    if dry_run:
        return embedded_points

    client = _qdrant_client(qdrant_url)
    if client is None:
        logger.info("router.ingest_repo: qdrant-client unavailable, returning dry-run payloads")
        return embedded_points

    _ensure_collection(client, collection, len(embedded_points[0]["vector"]))
    client.upsert(collection_name=collection, points=embedded_points)
    return embedded_points


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a repo into Qdrant repo_chunks")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    points = asyncio.run(
        ingest_repo(
            args.repo,
            collection=args.collection,
            qdrant_url=args.qdrant_url,
            embedding_model=args.embedding_model,
            branch=args.branch,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps({"count": len(points), "points": points[:3]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
