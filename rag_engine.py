from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import ollama
from pypdf import PdfReader

DEFAULT_INDEX_DIR = Path("data/got_index")

CHUNKS_FILE = "chunks.json"
VECTORS_FILE = "vectors.npy"
METADATA_FILE = "metadata.json"
MIN_CHUNK_CHARS = 80

SYSTEM_PROMPT = """You are a helpful Game of Thrones book assistant.
Use only the provided context from the books and do not use outside knowledge.
Default style: natural conversational tone, short and smooth.
Unless the user asks for detail, answer in 2-5 sentences and avoid bullet points.
If the answer is not in the context, explicitly say you do not know from the provided books.
If evidence in the context conflicts (for example, alive in one passage and dead in another), explain that it depends on timeline and cite both.
Do not speculate or add details that are not directly supported by the retrieved context.
Keep claims tightly grounded and cite sources as [1], [2], etc., matching the numbered context blocks."""


@dataclass
class LoadedIndex:
    vectors: np.ndarray
    chunks: list[dict]
    metadata: dict
    bm25: "BM25Index | None" = None


@dataclass
class BM25Index:
    postings: dict[str, list[tuple[int, int]]]
    doc_lengths: np.ndarray
    avg_doc_length: float
    doc_count: int


def find_pdf_file(search_dir: Path) -> Path | None:
    pdfs = sorted(search_dir.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def extract_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []
    for page_num, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        cleaned = normalize_text(extracted)
        if cleaned:
            pages.append({"page": page_num, "text": cleaned})
    return pages


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            split_point = text.rfind(" ", max(start, end - 120), end)
            if split_point > start:
                end = split_point
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> list[dict]:
    chunk_id = 0
    chunks: list[dict] = []
    for page in pages:
        for piece in _split_text(page["text"], chunk_size, chunk_overlap):
            if len(piece) < min_chunk_chars:
                continue
            chunks.append(
                {
                    "id": chunk_id,
                    "page": page["page"],
                    "text": piece,
                }
            )
            chunk_id += 1
    return chunks


OLLAMA_CONNECTIVITY_ERROR = (
    "Could not connect to Ollama. Start Ollama locally and ensure the model is pulled."
)


def _embed_batch(client: ollama.Client, model: str, batch: list[str]) -> list[list[float]]:
    if hasattr(client, "embed"):
        try:
            response = client.embed(model=model, input=batch)
            return response["embeddings"]
        except Exception as exc:
            message = str(exc).lower()
            if "connect" in message or "refused" in message or "timed out" in message:
                raise RuntimeError(OLLAMA_CONNECTIVITY_ERROR) from exc

    embeddings = []
    for text in batch:
        try:
            response = client.embeddings(model=model, prompt=text)
        except Exception as exc:
            raise RuntimeError(OLLAMA_CONNECTIVITY_ERROR) from exc
        embeddings.append(response["embedding"])
    return embeddings


def embed_texts(
    texts: list[str], model: str, batch_size: int = 24, client: ollama.Client | None = None
) -> np.ndarray:
    if not texts:
        raise ValueError("No texts provided for embedding.")

    client = client or ollama.Client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(_embed_batch(client, model, batch))

    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_bm25_index(chunks: list[dict]) -> BM25Index:
    postings: dict[str, list[tuple[int, int]]] = {}
    doc_lengths = np.zeros(len(chunks), dtype=np.float32)

    for doc_id, chunk in enumerate(chunks):
        tokens = tokenize_text(chunk["text"])
        term_freq = Counter(tokens)
        doc_lengths[doc_id] = float(sum(term_freq.values()))
        for term, tf in term_freq.items():
            postings.setdefault(term, []).append((doc_id, int(tf)))

    avg_doc_length = float(doc_lengths.mean()) if len(doc_lengths) else 1.0
    return BM25Index(
        postings=postings,
        doc_lengths=doc_lengths,
        avg_doc_length=max(avg_doc_length, 1.0),
        doc_count=len(chunks),
    )


def build_index(
    pdf_path: Path,
    index_dir: Path = DEFAULT_INDEX_DIR,
    embedding_model: str = "nomic-embed-text",
    chunk_size: int = 1200,
    chunk_overlap: int = 180,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
    batch_size: int = 24,
    progress: Callable[[str], None] | None = None,
) -> dict:
    progress = progress or (lambda _: None)

    progress(f"Reading PDF: {pdf_path}")
    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError("No readable text found in the PDF.")

    progress("Chunking text")
    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=min_chunk_chars,
    )
    if not chunks:
        raise ValueError("No chunks were created. Adjust chunk settings.")

    progress(f"Embedding {len(chunks)} chunks with model '{embedding_model}'")
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts, model=embedding_model, batch_size=batch_size)

    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    np.save(index_dir / VECTORS_FILE, vectors)

    metadata = {
        "source_pdf": str(pdf_path),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "min_chunk_chars": min_chunk_chars,
        "chunk_count": len(chunks),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (index_dir / METADATA_FILE).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    progress(f"Index saved to: {index_dir}")
    return metadata


def load_index(index_dir: Path = DEFAULT_INDEX_DIR) -> LoadedIndex:
    chunks_path = index_dir / CHUNKS_FILE
    vectors_path = index_dir / VECTORS_FILE
    metadata_path = index_dir / METADATA_FILE

    if not chunks_path.exists() or not vectors_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Index not found in {index_dir}. Run ingest.py first to build the index."
        )

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    vectors = np.load(vectors_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    bm25 = build_bm25_index(chunks)
    return LoadedIndex(vectors=vectors, chunks=chunks, metadata=metadata, bm25=bm25)


def retrieve_chunks(
    question: str,
    index: LoadedIndex,
    embedding_model: str,
    top_k: int = 8,
) -> list[dict]:
    query_variants = expand_queries(question)

    semantic_sets: list[np.ndarray] = []
    bm25_sets: list[np.ndarray] = []
    lexical_sets: list[np.ndarray] = []
    status_sets: list[np.ndarray] = []
    final_sets: list[np.ndarray] = []

    for query in query_variants:
        query_vector = embed_texts([query], model=embedding_model)[0]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            semantic_raw = index.vectors @ query_vector
        semantic_raw = np.nan_to_num(semantic_raw, nan=-1.0, posinf=1.0, neginf=-1.0)
        semantic_norm = normalize_scores(semantic_raw)

        bm25_raw = compute_bm25_scores(query, index.bm25)
        if bm25_raw.size == 0:
            bm25_raw = np.zeros(len(index.chunks), dtype=np.float32)
        bm25_norm = normalize_scores(bm25_raw)

        lexical = keyword_match_scores(query, index.chunks)
        status = status_match_scores(query, index.chunks)
        if any(term in STATUS_TERMS for term in tokenize_text(query)):
            combined = 0.45 * semantic_norm + 0.25 * bm25_norm + 0.10 * lexical + 0.20 * status
        else:
            combined = 0.6 * semantic_norm + 0.3 * bm25_norm + 0.1 * lexical
        semantic_sets.append(semantic_norm)
        bm25_sets.append(bm25_norm)
        lexical_sets.append(lexical)
        status_sets.append(status)
        final_sets.append(combined)

    semantic_scores = np.max(np.vstack(semantic_sets), axis=0)
    bm25_scores = np.max(np.vstack(bm25_sets), axis=0)
    lexical_scores = np.max(np.vstack(lexical_sets), axis=0)
    status_scores = np.max(np.vstack(status_sets), axis=0)
    final_scores = np.max(np.vstack(final_sets), axis=0)

    top_k = max(1, min(top_k, len(index.chunks)))
    if len(final_sets) > 1:
        selected: list[int] = []
        for variant_scores in final_sets:
            idx = int(np.argmax(variant_scores))
            if idx not in selected:
                selected.append(idx)
        for idx in np.argsort(final_scores)[::-1]:
            idx_int = int(idx)
            if idx_int in selected:
                continue
            selected.append(idx_int)
            if len(selected) >= top_k:
                break
        top_indices = np.array(selected[:top_k], dtype=np.int64)
    else:
        top_indices = np.argsort(final_scores)[::-1][:top_k]

    results: list[dict] = []
    for idx in top_indices:
        chunk = index.chunks[int(idx)]
        results.append(
            {
                "id": chunk["id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(final_scores[idx]),
                "semantic_score": float(semantic_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "lexical_score": float(lexical_scores[idx]),
                "status_score": float(status_scores[idx]),
            }
        )
    return results


STOP_WORDS = {
    "about",
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "do",
    "does",
    "did",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "describe",
    "explain",
    "can",
    "could",
    "would",
    "should",
    "tell",
    "that",
    "the",
    "to",
    "us",
    "what",
    "who",
    "with",
    "you",
}

STATUS_TERMS = {"alive", "dead", "death", "killed", "slain", "murdered", "dies", "died"}


def normalize_query(question: str) -> str:
    # Basic typo normalization for common GOT entity queries.
    question = question.lower()
    question = re.sub(r"\bjohn snow\b", "jon snow", question)
    return re.sub(r"\s+", " ", question).strip()


def expand_queries(question: str) -> list[str]:
    normalized = normalize_query(question)
    lower = normalized.lower()
    queries = [normalized]

    if any(term in lower for term in STATUS_TERMS):
        subject_terms = [
            token for token in tokenize_text(lower) if token not in STOP_WORDS and token not in STATUS_TERMS
        ]
        if subject_terms:
            subject = " ".join(subject_terms[:4])
            queries.append(f"{subject} dead killed murdered slain died")
            queries.append(f"{subject} alive survives living")
            for token in subject_terms[:2]:
                queries.append(f"{token} dead killed murdered slain died")
                queries.append(f"{token} alive survives living")
        queries.append(f"{normalized} fate timeline")

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)
    return deduped


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    cleaned = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    minimum = float(cleaned.min())
    maximum = float(cleaned.max())
    if maximum - minimum < 1e-9:
        return np.zeros_like(cleaned, dtype=np.float32)
    return (cleaned - minimum) / (maximum - minimum)


def _query_terms(question: str) -> list[str]:
    return [
        token
        for token in tokenize_text(question)
        if len(token) >= 2 and token not in STOP_WORDS
    ]


def compute_bm25_scores(question: str, bm25: BM25Index | None) -> np.ndarray:
    if bm25 is None or bm25.doc_count == 0:
        return np.zeros(0, dtype=np.float32)

    terms = _query_terms(question)
    if not terms:
        return np.zeros(bm25.doc_count, dtype=np.float32)

    scores = np.zeros(bm25.doc_count, dtype=np.float32)
    k1 = 1.5
    b = 0.75

    for term in terms:
        postings = bm25.postings.get(term)
        if not postings:
            continue
        df = len(postings)
        idf = np.log(1.0 + (bm25.doc_count - df + 0.5) / (df + 0.5))
        for doc_id, tf in postings:
            doc_length = float(bm25.doc_lengths[doc_id])
            denom = tf + k1 * (1 - b + b * doc_length / bm25.avg_doc_length)
            if denom == 0:
                continue
            scores[doc_id] += float(idf) * ((tf * (k1 + 1)) / denom)
    return scores


def status_match_scores(question: str, chunks: list[dict]) -> np.ndarray:
    query_tokens = tokenize_text(question)
    status_terms = [token for token in query_tokens if token in STATUS_TERMS]
    subject_terms = [
        token
        for token in query_tokens
        if token not in STATUS_TERMS and token not in STOP_WORDS and len(token) >= 2
    ]

    if not status_terms or not subject_terms:
        return np.zeros(len(chunks), dtype=np.float32)

    scores = np.zeros(len(chunks), dtype=np.float32)
    for idx, chunk in enumerate(chunks):
        text = chunk["text"].lower()
        subject_hits = sum(1 for term in subject_terms if re.search(rf"\b{re.escape(term)}\b", text))
        status_hits = sum(1 for term in status_terms if re.search(rf"\b{re.escape(term)}\b", text))
        if subject_hits == 0 or status_hits == 0:
            continue

        proximity = 0.0
        for subject in subject_terms[:2]:
            for status_term in status_terms:
                near_pattern = (
                    rf"\b{re.escape(subject)}\b.{{0,40}}\b{re.escape(status_term)}\b|"
                    rf"\b{re.escape(status_term)}\b.{{0,40}}\b{re.escape(subject)}\b"
                )
                if re.search(near_pattern, text):
                    proximity = 0.4
                    break
            if proximity:
                break

        base = 0.6 * (subject_hits / len(subject_terms)) + 0.4 * (status_hits / len(status_terms))
        scores[idx] = min(1.0, base + proximity)
    return scores


def keyword_match_scores(question: str, chunks: list[dict]) -> np.ndarray:
    question_lower = question.lower().strip()
    terms = [token for token in _query_terms(question_lower) if len(token) >= 3]
    if not terms:
        return np.zeros(len(chunks), dtype=np.float32)

    scores = np.zeros(len(chunks), dtype=np.float32)
    for idx, chunk in enumerate(chunks):
        text_lower = chunk["text"].lower()
        hits = sum(1 for token in terms if re.search(rf"\b{re.escape(token)}\b", text_lower))
        if hits:
            scores[idx] = hits / len(terms)
        if question_lower in text_lower:
            scores[idx] = min(1.0, scores[idx] + 0.3)
    return scores


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (page {chunk['page']})\n{chunk['text']}")
    return "\n\n".join(blocks)


def _prepare_history(chat_history: list[dict] | None, max_messages: int = 8) -> list[dict]:
    if not chat_history:
        return []

    prepared: list[dict] = []
    for message in chat_history:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        prepared.append({"role": role, "content": content})
    return prepared[-max_messages:]


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    chat_model: str,
    chat_history: list[dict] | None = None,
) -> str:
    client = ollama.Client()
    context = build_context(retrieved_chunks)

    prompt = (
        "Context passages from the books:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer based only on the context. "
        "If the question depends on timeline, explicitly say so and include both early and later evidence with citations. "
        "If context is insufficient for a claim, say that clearly instead of guessing. "
        "Keep the answer concise and conversational unless the user asks for deep detail."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_prepare_history(chat_history))
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat(
            model=chat_model,
            messages=messages,
            options={"temperature": 0.35, "num_predict": 280},
        )
    except Exception as exc:
        raise RuntimeError(OLLAMA_CONNECTIVITY_ERROR) from exc
    return response["message"]["content"].strip()


def answer_with_rag(
    question: str,
    index: LoadedIndex,
    chat_model: str,
    top_k: int = 8,
    chat_history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    embedding_model = index.metadata["embedding_model"]
    hits = retrieve_chunks(
        question=question,
        index=index,
        embedding_model=embedding_model,
        top_k=top_k,
    )
    answer = generate_answer(
        question=question,
        retrieved_chunks=hits,
        chat_model=chat_model,
        chat_history=chat_history,
    )
    return answer, hits
