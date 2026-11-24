from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.documents import chunk_text, extract_text, mask_sensitive
from app.services.rag import _local_embed

DATA_DIR = Path(os.getenv("APP_DATA_DIR", ROOT / "data"))
CHROMA_DIR = Path(os.getenv("VECTOR_DB_PATH", DATA_DIR / "chroma"))
LEGACY_KEYWORDS = (
    "loan",
    "mortgage",
    "bank",
    "credit score",
    "credit assessment",
    "financial institution",
    "customer onboarding",
    "kyc",
    "investment",
    "portfolio",
    "account opening",
)


def is_legacy_financial_text(content: str) -> bool:
    lowered = content.lower()
    return any(keyword in lowered for keyword in LEGACY_KEYWORDS)


def remove_legacy_project_jsons() -> list[str]:
    removed: list[str] = []
    for path in sorted(DATA_DIR.glob("*.json")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if is_legacy_financial_text(text):
            path.unlink(missing_ok=True)
            removed.append(path.name)
    return removed


def remove_legacy_collections() -> list[str]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    removed: list[str] = []
    for collection in client.list_collections():
        lowered = collection.name.lower()
        if lowered.startswith("finreq_") or any(keyword in lowered for keyword in ("loan", "mortgage", "bank", "credit", "financial")):
            client.delete_collection(collection.name)
            removed.append(collection.name)
    return removed


def build_healthcare_demo_collection(collection_name: str = "healthreq_demo_healthcare") -> int:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    source_files = [ROOT / "sample_data" / "digital_onboarding_policy.txt"]
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []

    for source_file in source_files:
        text = mask_sensitive(extract_text(source_file.name, source_file.read_bytes()))
        for chunk in chunk_text(text, size=700):
            ids.append(f"{source_file.stem}:{chunk.id}")
            documents.append(chunk.text)
            metadatas.append({"document_id": source_file.stem, "document_name": source_file.name, "chunk_id": chunk.id})

    if not documents:
        raise RuntimeError("No healthcare evidence documents were found to reindex.")

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=[_local_embed(text) for text in documents],
    )
    return collection.count()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    removed_jsons = remove_legacy_project_jsons()
    removed_collections = remove_legacy_collections()
    count = build_healthcare_demo_collection()

    print("Healthcare RAG rebuild complete.")
    print(f"Removed JSON exports: {len(removed_jsons)}")
    if removed_jsons:
        print("  " + ", ".join(removed_jsons))
    print(f"Removed Chroma collections: {len(removed_collections)}")
    if removed_collections:
        print("  " + ", ".join(removed_collections))
    print(f"Indexed healthcare demo chunks: {count}")
    print(f"Demo collection: healthreq_demo_healthcare")
    print("The app still keeps its current architecture and workflow; only the RAG evidence store has been refreshed for healthcare.")


if __name__ == "__main__":
    main()
