from __future__ import annotations
import json
import math
import os
import hashlib
import logging
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import chromadb
from chromadb.errors import NotFoundError


logger = logging.getLogger(__name__) 
MIN_RELEVANCE = 0.15


class RagConfigurationError(RuntimeError):
    """Raised when semantic retrieval cannot be configured or completed."""


def _api_key() -> str:
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RagConfigurationError("Set LLM_API_KEY or OPENAI_API_KEY in .env before using the remote provider.")
    return key


def provider_configured() -> bool:
    return bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed with the configured provider, retaining local retrieval on provider failures."""
    if not texts:
        return []
    if not provider_configured():
        return [_local_embed(text) for text in texts]
    payload = {"model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"), "input": texts}
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    try:
        request = Request(
            f"{base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=35) as response:  # noqa: S310 - operator-configured API URL
            result = json.loads(response.read().decode("utf-8"))
        vectors = [item["embedding"] for item in sorted(result["data"], key=lambda item: item["index"])]
        dimensions = {len(vector) for vector in vectors if isinstance(vector, list)}
        if (
            len(vectors) != len(texts)
            or not all(
                isinstance(vector, list)
                and vector
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in vector
                )
                for vector in vectors
            )
            or len(dimensions) != 1
        ):
            raise ValueError("response contained incomplete embedding data")
        return vectors
    except HTTPError as exc:
        logger.warning(
            "Embedding provider returned HTTP %s; using local embeddings.",
            exc.code,
        )
        return [_local_embed(text) for text in texts]
    except (URLError, OSError) as exc:
        reason = getattr(exc, "reason", None)
        error_type = type(reason).__name__ if reason is not None else type(exc).__name__
        logger.warning(
            "Embedding provider request failed (%s); using local embeddings.",
            error_type,
        )
        return [_local_embed(text) for text in texts]
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        # Project creation and evidence retrieval stay usable when a configured
        # provider rejects a model, credential, or request.
        logger.warning(
            "Embedding provider returned invalid vector data (%s); using local embeddings.",
            type(exc).__name__,
        )
        return [_local_embed(text) for text in texts]


def _local_embed(text: str, dimensions: int = 384) -> list[float]:
    """Create stable local vectors so the prototype remains usable without a provider key."""
    vector = [0.0] * dimensions
    for token in re.findall(r"[a-z0-9]{2,}", text.lower()):
        bucket = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big") % dimensions
        vector[bucket] += 1.0
    magnitude = sum(value * value for value in vector) ** 0.5
    return [value / magnitude for value in vector] if magnitude else vector


class VectorStore:
    """Persistent ChromaDB store using vectors from an OpenAI-compatible embedding endpoint."""

    def __init__(self) -> None:
        root = Path(os.getenv("VECTOR_DB_PATH", Path(os.getenv("APP_DATA_DIR", "data")) / "chroma"))
        root.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(root))

    @staticmethod
    def _collection_name(project_id: str) -> str:
        return f"healthreq_{project_id}"

    @staticmethod
    def configuration() -> dict[str, str]:
        """Return safe-to-display settings without exposing provider credentials."""
        return {
            "vector_database": "ChromaDB",
            "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") if provider_configured() else "Local hashed embeddings (384 dimensions)",
            "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini") if provider_configured() else "Evidence-only fallback",
            "collection_scope": "One persistent collection per project",
        }

    def index(self, project_id: str, documents: list[dict]) -> int:
        ids: list[str] = []
        texts: list[str] = []
        metadata: list[dict[str, str]] = []
        seen_chunks: set[tuple[str, str]] = set()
        for document in documents:
            for chunk in document.get("chunks", []):
                normalized_text = " ".join(chunk.get("text", "").split())
                if not normalized_text:
                    continue
                chunk_key = (document["id"], normalized_text.casefold())
                if chunk_key in seen_chunks:
                    continue
                seen_chunks.add(chunk_key)
                ids.append(f"{document['id']}:{chunk['id']}")
                texts.append(normalized_text)
                metadata.append({"document_id": document["id"], "document_name": document["name"], "chunk_id": chunk["id"]})
        if not texts:
            raise RagConfigurationError("No text chunks were available for semantic indexing.")
        try:
            collection = self.client.get_or_create_collection(
                self._collection_name(project_id), metadata={"hnsw:space": "cosine"}
            )
            collection.upsert(ids=ids, documents=texts, metadatas=metadata, embeddings=_embed(texts))
        except RagConfigurationError:
            raise
        except Exception as exc:
            raise RagConfigurationError("The vector database could not index the project sources.") from exc
        return len(texts)

    def retrieve(self, project_id: str, query: str, limit: int = 6) -> list[dict]:
        if limit <= 0 or not query.strip():
            return []
        try:
            collection = self.client.get_collection(self._collection_name(project_id))
        except NotFoundError as exc:
            raise RagConfigurationError("The project's vector collection is unavailable. Recreate the project to index its sources.") from exc
        count = collection.count()
        if count == 0:
            return []
        try:
            result = collection.query(
                query_embeddings=_embed([query]),
                n_results=min(limit, count),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RagConfigurationError("The vector database could not retrieve project evidence.") from exc
        documents = result.get("documents", [[]])[0]
        metadata = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        matches = []
        for text, item, distance in zip(documents, metadata, distances, strict=True):
            relevance = max(0.0, min(1.0, 1.0 - float(distance)))
            if relevance < MIN_RELEVANCE:
                continue
            matches.append({
                "id": item["chunk_id"],
                "text": text,
                "document_id": item["document_id"],
                "document_name": item["document_name"],
                "relevance": round(relevance, 2),
            })
        return matches
