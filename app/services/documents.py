from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from xml.etree import ElementTree

from pypdf import PdfReader
from pypdf.errors import PdfReadError


SENSITIVE_PATTERNS = [
    (re.compile(r"(?i)\b(?:mrn|medical record number|patient id)\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{2,19}\b"), "[MASKED_MRN]"),
    (re.compile(r"(?i)\b(?:dob|date of birth)\s*[:#-]?\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"), "[MASKED_DOB]"),
    (re.compile(r"(?i)\b(?:patient(?:\s+name)?|name\s+of\s+patient|name)\s*[:#=-]\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"), "[MASKED_PATIENT_NAME]"),
    (re.compile(r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"), "[MASKED_PATIENT_NAME]"),
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"), "[MASKED_PHONE]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[MASKED_EMAIL]"),
]


@dataclass
class Chunk:
    id: str
    text: str


class DocumentExtractionError(ValueError):
    """Raised when an uploaded document cannot be parsed safely."""


def mask_sensitive(text: str) -> str:
    for pattern, replacement in SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def extract_text(filename: str, content: bytes) -> str:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else "txt"
    if suffix in {"txt", "md", "csv"}:
        return content.decode("utf-8", errors="replace")
    if suffix == "pdf":
        try:
            return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
        except (PdfReadError, OSError, ValueError) as exc:
            raise DocumentExtractionError("The PDF could not be read. Check that it is a valid, text-based PDF.") from exc
    if suffix == "docx":
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                root = ElementTree.fromstring(archive.read("word/document.xml"))
        except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError) as exc:
            raise DocumentExtractionError("The DOCX could not be read. Check that it is a valid Word document.") from exc
        return "\n".join(node.text or "" for node in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
    raise ValueError("Supported formats are .txt, .md, .csv, .pdf, and .docx")


def chunk_text(text: str, size: int = 700) -> list[Chunk]:
    if size < 1:
        raise ValueError("Chunk size must be a positive integer.")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    normalized = [line for line in lines if line]
    if not normalized:
        return []
    sentences = [
        sentence.strip()
        for line in normalized
        for sentence in re.split(r"(?<=[.!?])\s+", line)
        if sentence.strip()
    ]
    chunks: list[Chunk] = []
    current = ""
    for sentence in sentences:
        pieces: list[str] = []
        if len(sentence) > size:
            words = sentence.split()
            piece = ""
            for word in words:
                if len(word) > size:
                    if piece:
                        pieces.append(piece)
                        piece = ""
                    pieces.extend(word[index:index + size] for index in range(0, len(word), size))
                    continue
                candidate = f"{piece} {word}".strip()
                if piece and len(candidate) > size:
                    pieces.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                pieces.append(piece)
        else:
            pieces = [sentence]
        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > size:
                chunks.append(Chunk(id=f"chunk-{len(chunks) + 1:03d}", text=current))
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(Chunk(id=f"chunk-{len(chunks) + 1:03d}", text=current))
    return chunks


def retrieve(query: str, documents: list[dict], limit: int = 6) -> list[dict]:
    if limit <= 0 or not query.strip():
        return []
    terms = {term.lower() for term in re.findall(r"[a-zA-Z]{3,}", query)}
    if not terms:
        return []
    ranked: list[tuple[float, dict]] = []
    for document in documents:
        for chunk in document.get("chunks", []):
            words = set(re.findall(r"[a-zA-Z]{3,}", chunk["text"].lower()))
            overlap = len(terms & words)
            score = overlap / max(1, len(terms))
            if score > 0:
                ranked.append((score, {**chunk, "document_id": document["id"], "document_name": document["name"], "relevance": round(score, 2)}))
    return [item for _, item in sorted(ranked, key=lambda pair: pair[0], reverse=True)[:limit]]
