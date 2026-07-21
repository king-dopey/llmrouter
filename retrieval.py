from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import httpx

try:
    from qdrant_client import QdrantClient  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None

logger = logging.getLogger("router.retrieval")

DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "repo_chunks")
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
DEFAULT_EMBEDDING_MODEL = os.getenv("QDRANT_EMBEDDING_MODEL", "qwen3-embedding:4b")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")


@dataclass(frozen=True)
class RetrievedChunk:
    repo: str
    branch: str | None
    path: str
    symbol: str | None
    score: float
    chunk_text: str
    line_start: int | None = None
    line_end: int | None = None
    language: str | None = None


def _tokenize(text: str) -> list[str]:
    return [item for item in re.findall(r"[a-z0-9]+", text.lower()) if len(item) > 1]


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if not vector_a or not vector_b:
        return 0.0
    size = min(len(vector_a), len(vector_b))
    dot = sum(vector_a[i] * vector_b[i] for i in range(size))
    norm_a = math.sqrt(sum(vector_a[i] * vector_a[i] for i in range(size)))
    norm_b = math.sqrt(sum(vector_b[i] * vector_b[i] for i in range(size)))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _rrf_rank_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _chunk_identity(chunk: RetrievedChunk) -> str:
    payload = f"{chunk.repo}|{chunk.branch or ''}|{chunk.path}|{chunk.symbol or ''}|{chunk.line_start or ''}|{chunk.line_end or ''}|{chunk.chunk_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _point_payload_to_chunk(point: Any) -> RetrievedChunk:
    payload = point.payload or {}
    vector = point.vector
    chunk_text = payload.get("chunk_text") or payload.get("content") or ""
    score = float(getattr(point, "score", 0.0) or 0.0)
    return RetrievedChunk(
        repo=str(payload.get("repo", "")),
        branch=payload.get("branch"),
        path=str(payload.get("path", "")),
        symbol=payload.get("symbol"),
        score=score,
        chunk_text=str(chunk_text),
        line_start=payload.get("line_start"),
        line_end=payload.get("line_end"),
        language=payload.get("language"),
    )


async def _embed_query(query: str, *, embedding_model: str | None = None, ollama_base_url: str | None = None) -> list[float]:
    model = embedding_model or DEFAULT_EMBEDDING_MODEL
    base_url = (ollama_base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    payload = {"model": model, "prompt": query}
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{base_url}/api/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding") or []
        return [float(value) for value in embedding]


def _local_sparse_score(query_tokens: list[str], text: str) -> float:
    if not query_tokens or not text:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    text_set = set(text_tokens)
    overlap = sum(1 for token in query_tokens if token in text_set)
    return overlap / max(1, len(query_tokens))


def _qdrant_client(qdrant_url: str | None = None):
    if QdrantClient is None:
        return None
    return QdrantClient(url=qdrant_url or DEFAULT_QDRANT_URL)


def _scroll_candidates_sync(
    *,
    qdrant_url: str | None,
    collection: str,
    filters: dict[str, Any] | None,
    limit: int,
) -> list[Any]:
    client = _qdrant_client(qdrant_url)
    if client is None:
        return []

    must_match: dict[str, Any] = filters or {}
    points, _ = client.scroll(collection_name=collection, limit=limit, with_payload=True, with_vectors=True)
    if not must_match:
        return list(points)

    filtered = []
    for point in points:
        payload = point.payload or {}
        if all(payload.get(key) in value if isinstance(value, list) else payload.get(key) == value for key, value in must_match.items()):
            filtered.append(point)
    return filtered


def _coerce_vector(raw_vector: Any) -> list[float]:
    if raw_vector is None:
        return []
    if isinstance(raw_vector, list):
        return [float(value) for value in raw_vector]
    if isinstance(raw_vector, dict):
        vector = raw_vector.get("default") or next(iter(raw_vector.values()), [])
        if isinstance(vector, list):
            return [float(value) for value in vector]
    return []


def _rank_retrieved_chunks(
    *,
    query_tokens: list[str],
    query_vector: list[float],
    points: Iterable[Any],
    repo: str,
    branch: str | None,
    filters: dict[str, Any] | None,
) -> list[RetrievedChunk]:
    candidates: list[tuple[float, RetrievedChunk]] = []
    for point in points:
        payload = point.payload or {}
        if repo and payload.get("repo") != repo:
            continue
        if branch and payload.get("branch") != branch:
            continue
        if filters:
            skip = False
            for key, expected in filters.items():
                if isinstance(expected, list):
                    if payload.get(key) not in expected:
                        skip = True
                        break
                elif payload.get(key) != expected:
                    skip = True
                    break
            if skip:
                continue

        chunk_text = str(payload.get("chunk_text") or payload.get("content") or "")
        sparse_score = _local_sparse_score(query_tokens, chunk_text)
        dense_score = _cosine_similarity(query_vector, _coerce_vector(point.vector))
        score = (0.65 * dense_score) + (0.35 * sparse_score)
        chunk = RetrievedChunk(
            repo=str(payload.get("repo", repo)),
            branch=payload.get("branch", branch),
            path=str(payload.get("path", "")),
            symbol=payload.get("symbol"),
            score=score,
            chunk_text=chunk_text,
            line_start=payload.get("line_start"),
            line_end=payload.get("line_end"),
            language=payload.get("language"),
        )
        candidates.append((score, chunk))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in candidates]


def _dedupe_chunks(chunks: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
    ordered: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        ordered.setdefault(_chunk_identity(chunk), chunk)
    return list(ordered.values())


def format_retrieval_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""
    lines = ["Retrieved context:"]
    for index, chunk in enumerate(chunks, start=1):
        header = f"{index}. {chunk.path}"
        if chunk.symbol:
            header += f"::{chunk.symbol}"
        if chunk.line_start is not None or chunk.line_end is not None:
            header += f" [{chunk.line_start or '?'}-{chunk.line_end or '?'}]"
        lines.append(header)
        lines.append("```text")
        lines.append(chunk.chunk_text)
        lines.append("```")
    return "\n".join(lines)


async def retrieve_context(
    *,
    repo: str,
    query: str,
    branch: str | None = None,
    top_k: int = 20,
    final_k: int = 8,
    filters: dict[str, Any] | None = None,
    qdrant_url: str | None = None,
    collection: str | None = None,
    embedding_model: str | None = None,
    ollama_base_url: str | None = None,
) -> list[dict[str, Any]]:
    if not repo or not query.strip():
        return []

    client = _qdrant_client(qdrant_url)
    if client is None:
        return []

    try:
        query_vector = await _embed_query(query, embedding_model=embedding_model, ollama_base_url=ollama_base_url)
    except Exception as exc:
        logger.info("router.retrieval: embedding unavailable, falling back to lexical scoring only: %s", exc)
        query_vector = []

    collection_name = collection or DEFAULT_COLLECTION
    points = await asyncio.to_thread(
        _scroll_candidates_sync,
        qdrant_url=qdrant_url,
        collection=collection_name,
        filters=filters,
        limit=max(top_k, final_k, 50),
    )
    if not points:
        return []

    query_tokens = _tokenize(query)
    ranked = _rank_retrieved_chunks(
        query_tokens=query_tokens,
        query_vector=query_vector,
        points=points,
        repo=repo,
        branch=branch,
        filters=filters,
    )
    unique = _dedupe_chunks(ranked)
    return [asdict(chunk) for chunk in unique[:final_k]]


__all__ = ["RetrievedChunk", "retrieve_context", "format_retrieval_context"]
