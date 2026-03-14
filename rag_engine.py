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
MIN_CHUNK_CHARS = 120

BOOK_TITLES = [
    "A Game of Thrones",
    "A Clash of Kings",
    "A Storm of Swords",
    "A Feast for Crows",
    "A Dance with Dragons",
]

SYSTEM_PROMPT = """You are a helpful Game of Thrones book assistant.
Use only the provided context from the books and do not use outside knowledge.
Treat conversation history only as disambiguation help; every factual claim must be grounded in the current context blocks.
This assistant covers only the five books in the provided PDF, not TV adaptation or outside canon.
Default style: natural conversational tone, clear and moderately detailed.
Unless the user explicitly asks for a very short reply, answer in 5-8 sentences and avoid bullet points.
If the answer is not in the context, explicitly say you do not know from the provided books.
If evidence in the context conflicts, explain that it depends on timeline and cite both.
Prefer the most direct definitional or event passages over incidental mentions.
Do not infer hidden motives, off-page causes, or responsibility unless the context explicitly states them.
Never convert ages, durations, or chronology into relative present-day phrasing. If the text says someone is fifteen, say they are described as fifteen or fifteen years old in that passage.
Do not upgrade a location/title phrase into a more specific family claim. For example, "bastard of Winterfell" does not by itself justify naming a parent.
Do not use implication language such as "this suggests" or "this implies" for unsupported claims. State only what the evidence explicitly says.
Do not speculate or add details that are not directly supported by the retrieved context.
Keep claims tightly grounded and cite sources as [1], [2], etc., matching the numbered context blocks."""

OLLAMA_CONNECTIVITY_ERROR = (
    "Could not connect to Ollama. Start Ollama locally and ensure the model is pulled."
)

PAGE_SEPARATOR = "\n\n"
FOLLOW_UP_PRONOUNS = {
    "he",
    "she",
    "him",
    "her",
    "his",
    "hers",
    "they",
    "them",
    "their",
    "it",
    "its",
    "that",
    "this",
    "those",
    "these",
    "there",
    "one",
    "ones",
}
FOLLOW_UP_PREFIXES = (
    "and ",
    "what about",
    "how about",
    "why did",
    "why was",
    "why is",
    "what happened",
    "where did",
    "where was",
    "when did",
    "did he",
    "did she",
    "did they",
    "was he",
    "was she",
    "were they",
)


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


def detect_book_title(text: str) -> str | None:
    lower = text.lower()
    for title in BOOK_TITLES:
        title_lower = title.lower()
        if title_lower not in lower:
            continue
        if (
            lower.startswith(title_lower)
            or "publishing history" in lower
            or "title page" in lower
            or "master - table of contents" in lower
            or len(tokenize_text(text)) <= 80
        ):
            return title
    return None


def _looks_like_contents_page(text: str) -> bool:
    lower = text.lower()
    return (
        lower.startswith("contents")
        or "master - table of contents" in lower
        or "cover title page copyright" in lower
        or lower.count("chapter ") >= 8
        or ("chapter 1" in lower and "chapter 2" in lower and "chapter 3" in lower)
        or (
            "appendix" in lower
            and lower.count("house ") >= 3
            and len(re.findall(r"[.!?]", text)) == 0
        )
        or (
            "acknowledgments" in lower
            and lower.count("house ") >= 2
            and len(re.findall(r"[.!?]", text)) == 0
        )
    )


def _looks_like_publishing_page(text: str) -> bool:
    lower = text.lower()
    markers = (
        "publishing history",
        "all rights reserved",
        "published in the united states",
        "library of congress",
        "ebook isbn",
        "bantam",
        "random house",
        "this is a work of fiction",
        "are works of fiction",
        "division of random house",
        "published by bantam",
    )
    return any(marker in lower for marker in markers)


def _looks_like_marketing_page(text: str) -> bool:
    lower = text.lower()
    markers = (
        "praise for",
        "graphic novel",
        "preview of",
        "time magazine",
        "american tolkien",
        "by george r. r. martin",
    )
    return any(marker in lower for marker in markers)


def _looks_like_dedication_page(text: str) -> bool:
    lower = text.lower().strip()
    token_count = len(tokenize_text(text))
    if lower.startswith("this one is for") or "written in crayon" in lower:
        return True
    if token_count > 120:
        return False
    return lower.startswith(("for ", "to "))


def _looks_like_named_chapter_page(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    return bool(
        re.match(
            r"^[A-Z]\s+(?:THE\s+)?[A-Z][A-Z'’&.-]+(?:\s+[A-Z][A-Z'’&.-]+){0,4}\s+[a-z]",
            compact,
        )
    )


def _is_story_opening_page(text: str) -> bool:
    lower = text.lower()
    return (
        lower.startswith("prologue")
        or lower.startswith("epilogue")
        or (
            bool(re.match(r"^chapter\s+[0-9ivxlcdm]+\b", lower))
            and lower.count("chapter ") <= 2
        )
        or _looks_like_named_chapter_page(text)
    )


def classify_page(text: str, keep_appendices: bool = True) -> tuple[bool, str | None]:
    lower = text.lower().strip()

    if _is_story_opening_page(text):
        return True, None
    if lower.startswith("appendix"):
        return keep_appendices, None if keep_appendices else "appendix"
    if _looks_like_contents_page(text):
        return False, "table_of_contents"
    if _looks_like_publishing_page(text):
        return False, "publishing"
    if _looks_like_marketing_page(text):
        return False, "marketing"
    if lower.startswith("acknowledgments"):
        return False, "acknowledgments"
    if lower.startswith("about the author"):
        return False, "about_author"
    if "george r. r. martin" in lower and len(tokenize_text(text)) <= 12:
        return False, "author_note"
    if "click here to view the maps in greater detail" in lower:
        return False, "map_link"
    if lower.startswith(("a note on chronology", "a cavil on chronology")):
        return False, "chronology_note"
    if _looks_like_dedication_page(text):
        return False, "dedication"
    return True, None


def annotate_pages(raw_pages: list[dict], keep_appendices: bool = True) -> list[dict]:
    annotated: list[dict] = []
    current_book = "Unknown"
    current_section = "story"

    for page in raw_pages:
        text = page["text"]
        lower = text.lower().strip()
        book_title = detect_book_title(text)
        new_book = bool(book_title and book_title != current_book)

        if new_book and book_title:
            current_book = book_title

        is_relevant, skip_reason = classify_page(text, keep_appendices=keep_appendices)

        if new_book and not _is_story_opening_page(text):
            current_section = "front_matter"

        if _is_story_opening_page(text):
            current_section = "story"
        elif lower.startswith("appendix"):
            current_section = "appendix"
        elif lower.startswith("acknowledgments") or lower.startswith("about the author"):
            current_section = "back_matter"
        elif is_relevant and current_section == "front_matter":
            current_section = "story"
        elif is_relevant and current_section not in {"appendix", "back_matter"}:
            current_section = "story"

        annotated.append(
            {
                "page": page["page"],
                "text": text,
                "book": current_book,
                "section_type": current_section,
                "is_relevant": is_relevant,
                "skip_reason": skip_reason,
            }
        )
    return annotated


def filter_pages(
    raw_pages: list[dict],
    filter_irrelevant_pages: bool = True,
    keep_appendices: bool = True,
) -> tuple[list[dict], dict]:
    annotated_pages = annotate_pages(raw_pages, keep_appendices=keep_appendices)
    skip_counts: Counter[str] = Counter()
    kept_pages: list[dict] = []

    for page in annotated_pages:
        if filter_irrelevant_pages and not page["is_relevant"]:
            skip_counts[page["skip_reason"] or "other"] += 1
            continue
        kept_pages.append(
            {
                "page": page["page"],
                "text": page["text"],
                "book": page["book"],
                "section_type": page.get("section_type", "story"),
            }
        )

    stats = {
        "total_pages": len(raw_pages),
        "kept_pages": len(kept_pages),
        "skipped_pages": len(raw_pages) - len(kept_pages),
        "skipped_by_reason": dict(sorted(skip_counts.items())),
        "filtering_enabled": filter_irrelevant_pages,
        "keep_appendices": keep_appendices,
    }
    return kept_pages, stats


def _iter_chunk_windows(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    if len(text) <= chunk_size:
        return [(0, len(text))]

    windows: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            search_start = max(start + chunk_size // 2, end - 240)
            split_point = max(
                text.rfind("\n\n", search_start, end),
                text.rfind(". ", search_start, end),
                text.rfind("? ", search_start, end),
                text.rfind("! ", search_start, end),
                text.rfind(" ", max(start, end - 120), end),
            )
            if split_point > start:
                end = split_point + 1
        windows.append((start, end))
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return windows


def _group_pages_by_book(pages: list[dict]) -> list[list[dict]]:
    if not pages:
        return []

    groups: list[list[dict]] = []
    current_group: list[dict] = [pages[0]]

    for page in pages[1:]:
        if page.get("book") != current_group[-1].get("book"):
            groups.append(current_group)
            current_group = [page]
            continue
        current_group.append(page)

    groups.append(current_group)
    return groups


def _combine_pages(pages: list[dict]) -> tuple[str, list[dict]]:
    combined_parts: list[str] = []
    page_spans: list[dict] = []
    cursor = 0

    for idx, page in enumerate(pages):
        if idx:
            combined_parts.append(PAGE_SEPARATOR)
            cursor += len(PAGE_SEPARATOR)

        text = page["text"].strip()
        start = cursor
        combined_parts.append(text)
        cursor += len(text)
        page_spans.append(
            {
                "page": page["page"],
                "start": start,
                "end": cursor,
                "section_type": page.get("section_type", "story"),
            }
        )

    return "".join(combined_parts), page_spans


def _pages_for_span(page_spans: list[dict], start: int, end: int) -> tuple[int, int]:
    page_start: int | None = None
    page_end: int | None = None

    for span in page_spans:
        if span["end"] <= start:
            continue
        if span["start"] >= end:
            break
        if page_start is None:
            page_start = span["page"]
        page_end = span["page"]

    if page_start is None or page_end is None:
        fallback = page_spans[0]["page"]
        return fallback, fallback
    return page_start, page_end


def _section_for_span(page_spans: list[dict], start: int, end: int) -> str:
    section_counts: Counter[str] = Counter()
    for span in page_spans:
        if span["end"] <= start:
            continue
        if span["start"] >= end:
            break
        section_counts[span.get("section_type", "story")] += 1
    if not section_counts:
        return "story"
    return section_counts.most_common(1)[0][0]


def chunk_pages(
    pages: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> list[dict]:
    chunk_id = 0
    chunks: list[dict] = []

    for page_group in _group_pages_by_book(pages):
        combined_text, page_spans = _combine_pages(page_group)
        book = page_group[0].get("book", "Unknown")

        for start, end in _iter_chunk_windows(combined_text, chunk_size, chunk_overlap):
            piece = combined_text[start:end].strip()
            if len(piece) < min_chunk_chars:
                continue

            page_start, page_end = _pages_for_span(page_spans, start, end)
            section_type = _section_for_span(page_spans, start, end)
            chunks.append(
                {
                    "id": chunk_id,
                    "page": page_start,
                    "page_start": page_start,
                    "page_end": page_end,
                    "book": book,
                    "section_type": section_type,
                    "text": piece,
                }
            )
            chunk_id += 1

    return chunks


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
    chunk_size: int = 1400,
    chunk_overlap: int = 220,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
    batch_size: int = 24,
    filter_irrelevant_pages: bool = True,
    keep_appendices: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict:
    progress = progress or (lambda _: None)

    progress(f"Reading PDF: {pdf_path}")
    raw_pages = extract_pages(pdf_path)
    if not raw_pages:
        raise ValueError("No readable text found in the PDF.")

    progress("Filtering and annotating pages")
    pages, page_stats = filter_pages(
        raw_pages,
        filter_irrelevant_pages=filter_irrelevant_pages,
        keep_appendices=keep_appendices,
    )
    if not pages:
        raise ValueError("No relevant pages remained after filtering.")

    progress(
        "Keeping "
        f"{page_stats['kept_pages']} of {page_stats['total_pages']} extracted pages"
    )
    progress("Chunking text across page boundaries")
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
        "page_filtering": filter_irrelevant_pages,
        "keep_appendices": keep_appendices,
        "page_stats": page_stats,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (index_dir / METADATA_FILE).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    progress(f"Index saved to: {index_dir}")
    return metadata


def _normalize_loaded_chunks(chunks: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for idx, chunk in enumerate(chunks):
        page_start = int(chunk.get("page_start", chunk.get("page", 0)))
        page_end = int(chunk.get("page_end", page_start))
        normalized.append(
            {
                **chunk,
                "id": int(chunk.get("id", idx)),
                "page": page_start,
                "page_start": page_start,
                "page_end": page_end,
                "book": chunk.get("book") or "Unknown",
                "section_type": chunk.get("section_type") or "story",
            }
        )
    return normalized


def load_index(index_dir: Path = DEFAULT_INDEX_DIR) -> LoadedIndex:
    chunks_path = index_dir / CHUNKS_FILE
    vectors_path = index_dir / VECTORS_FILE
    metadata_path = index_dir / METADATA_FILE

    if not chunks_path.exists() or not vectors_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Index not found in {index_dir}. Run ingest.py first to build the index."
        )

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = _normalize_loaded_chunks(json.load(f))
    vectors = np.load(vectors_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    bm25 = build_bm25_index(chunks)
    return LoadedIndex(vectors=vectors, chunks=chunks, metadata=metadata, bm25=bm25)


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
    "happen",
    "happened",
    "happens",
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
    "when",
    "where",
    "why",
    "who",
    "with",
    "you",
}

STATUS_TERMS = {
    "alive",
    "assassinated",
    "betrayed",
    "dead",
    "death",
    "executed",
    "execution",
    "killed",
    "slain",
    "murdered",
    "dies",
    "died",
    "stabbed",
    "survive",
    "survives",
    "survived",
    "wounded",
}

QUERY_NORMALIZATIONS = {
    r"\bjohn snow\b": "jon snow",
    r"\bdaenarys\b": "daenerys",
    r"\bdany\b": "daenerys",
    r"\bcersi\b": "cersei",
    r"\bjoffery\b": "joffrey",
    r"\bkhaleesi\b": "daenerys",
    r"\bned stark\b": "eddard stark",
    r"\blittlefinger\b": "petyr baelish",
    r"\bthe hound\b": "sandor clegane",
    r"\bthe mountain\b": "gregor clegane",
    r"\bthe imp\b": "tyrion lannister",
}

IDENTITY_DESCRIPTOR_TERMS = {
    "son",
    "daughter",
    "bastard",
    "brother",
    "sister",
    "lord",
    "lady",
    "king",
    "queen",
    "prince",
    "princess",
    "maester",
    "commander",
    "warden",
    "captain",
    "steward",
    "khaleesi",
    "mother",
    "father",
    "wife",
    "husband",
    "heir",
    "widow",
    "widower",
}

EVENT_TERMS = STATUS_TERMS | {
    "behead",
    "beheaded",
    "burned",
    "burnt",
    "capture",
    "captured",
    "confess",
    "confessed",
    "escape",
    "escaped",
    "hang",
    "hanged",
    "hung",
    "imprison",
    "imprisoned",
    "poison",
    "poisoned",
    "slaughter",
    "slaughtered",
}

DECISIVE_EVENT_TERMS = {
    "beheaded",
    "dead",
    "death",
    "dies",
    "died",
    "executed",
    "execution",
    "killed",
    "murdered",
    "slain",
}

HYPOTHETICAL_MARKERS = (
    "would ",
    "could ",
    "might ",
    "if ",
    "perhaps",
    "maybe",
    "dream",
    "dreamed",
    "thought",
    "wondered",
    "seemed",
    "supposed",
    "rumor",
    "rumored",
    "they say",
    "it is said",
    "as if",
    "imagined",
    "hope",
    "hoped",
)

SUBJECT_ALIAS_OVERRIDES = {
    "cersei lannister": ["queen cersei", "cersei"],
    "daenerys targaryen": ["daenerys", "daenerys stormborn", "khaleesi"],
    "eddard stark": ["ned stark", "lord eddard"],
    "gregor clegane": ["the mountain"],
    "jon snow": ["lord snow"],
    "petyr baelish": ["littlefinger"],
    "robb stark": ["young wolf"],
    "sandor clegane": ["the hound"],
    "tyrion lannister": ["the imp", "tyrion"],
}


def normalize_query(question: str) -> str:
    question = question.lower()
    for pattern, replacement in QUERY_NORMALIZATIONS.items():
        question = re.sub(pattern, replacement, question)
    return re.sub(r"\s+", " ", question).strip()


def _query_terms(question: str) -> list[str]:
    return [
        token for token in tokenize_text(question) if len(token) >= 2 and token not in STOP_WORDS
    ]


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = re.sub(r"\s+", " ", query).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def extract_focus_entity(text: str) -> str | None:
    candidates = re.findall(r"\b[A-Z][a-z']+(?:\s+[A-Z][a-z']+){0,2}\b", text)
    ignored = {
        "Game",
        "Thrones",
        "Song",
        "Ice",
        "Fire",
        "Question",
        "Answer",
        "Context",
    }
    for candidate in reversed(candidates):
        words = candidate.split()
        if all(word in ignored for word in words):
            continue
        return candidate

    fallback_terms = [token for token in _query_terms(text) if token not in FOLLOW_UP_PRONOUNS]
    if not fallback_terms:
        return None
    return " ".join(fallback_terms[:3])


def contextualize_queries(question: str, chat_history: list[dict] | None = None) -> list[str]:
    queries = [question]
    if not chat_history:
        return queries

    lower = question.lower().strip()
    tokens = tokenize_text(lower)
    follow_up = (
        len(_query_terms(question)) <= 5
        or any(token in FOLLOW_UP_PRONOUNS for token in tokens)
        or any(lower.startswith(prefix) for prefix in FOLLOW_UP_PREFIXES)
    )
    if not follow_up:
        return queries

    recent_user_messages = [
        (message.get("content") or "").strip()
        for message in chat_history
        if message.get("role") == "user" and (message.get("content") or "").strip()
    ]
    recent_assistant_messages = [
        (message.get("content") or "").strip()
        for message in chat_history
        if message.get("role") == "assistant" and (message.get("content") or "").strip()
    ]

    if recent_user_messages:
        last_user = recent_user_messages[-1]
        queries.append(f"{last_user} {question}")
        entity = extract_focus_entity(last_user)
        if entity:
            queries.append(f"{entity} {question}")

    if recent_assistant_messages:
        entity = extract_focus_entity(recent_assistant_messages[-1])
        if entity:
            queries.append(f"{entity} {question}")

    return _dedupe_queries(queries)


def expand_queries(question: str) -> list[str]:
    normalized = normalize_query(question)
    lower = normalized.lower()
    terms = _query_terms(lower)
    queries = [normalized]

    if 2 <= len(terms) <= 5:
        queries.append(" ".join(terms))

    if any(term in STATUS_TERMS for term in tokenize_text(lower)):
        subject_terms = [token for token in terms if token not in STATUS_TERMS]
        if subject_terms:
            subject = " ".join(subject_terms[:4])
            queries.append(f"{subject} dead killed murdered slain died")
            queries.append(f"{subject} alive survives living wounded stabbed betrayed")
            queries.append(f"{subject} fate timeline")

    if _is_identity_question(normalized) and terms:
        subject = " ".join(terms[:4])
        queries.append(subject)
        queries.append(f"{subject} identity family role")
        queries.append(f"{subject} family parent house role bastard son daughter")

    if lower.startswith(("what happened to ", "what happens to ")) and terms:
        subject = " ".join(terms[:4])
        queries.append(f"{subject} fate outcome")
        queries.append(f"{subject} killed executed murdered stabbed betrayed dies")

    return _dedupe_queries(queries)


def build_query_variants(question: str, chat_history: list[dict] | None = None) -> list[str]:
    queries: list[str] = []
    for query in contextualize_queries(question, chat_history):
        queries.extend(expand_queries(query))
    return _dedupe_queries(queries)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    cleaned = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    minimum = float(cleaned.min())
    maximum = float(cleaned.max())
    if maximum - minimum < 1e-9:
        return np.zeros_like(cleaned, dtype=np.float32)
    return (cleaned - minimum) / (maximum - minimum)


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


def _query_phrases(question: str) -> list[tuple[str, int]]:
    terms = _query_terms(question)
    if len(terms) < 2:
        return []

    phrases: list[tuple[str, int]] = []
    seen: set[str] = set()
    max_n = min(4, len(terms))
    for size in range(max_n, 1, -1):
        for start in range(0, len(terms) - size + 1):
            phrase = " ".join(terms[start : start + size])
            if phrase in seen or len(phrase) < 5:
                continue
            seen.add(phrase)
            phrases.append((phrase, size))
    return phrases[:8]


def phrase_match_scores(question: str, chunks: list[dict]) -> np.ndarray:
    phrases = _query_phrases(question)
    if not phrases:
        return np.zeros(len(chunks), dtype=np.float32)

    scores = np.zeros(len(chunks), dtype=np.float32)
    for idx, chunk in enumerate(chunks):
        text_lower = chunk["text"].lower()
        best = 0.0
        for phrase, size in phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", text_lower):
                best = max(best, min(1.0, 0.35 + 0.15 * size))
        scores[idx] = best
    return scores


def proximity_match_scores(question: str, chunks: list[dict]) -> np.ndarray:
    terms = _query_terms(question)[:5]
    if len(terms) < 2:
        return np.zeros(len(chunks), dtype=np.float32)

    scores = np.zeros(len(chunks), dtype=np.float32)
    for idx, chunk in enumerate(chunks):
        positions: list[tuple[int, str]] = []
        text_lower = chunk["text"].lower()
        for term in terms:
            positions.extend(
                (match.start(), term)
                for match in re.finditer(rf"\b{re.escape(term)}\b", text_lower)
            )
        if len(positions) < 2:
            continue

        positions.sort()
        counts: Counter[str] = Counter()
        unique_terms = 0
        left = 0
        best = 0.0

        for right, (position, term) in enumerate(positions):
            counts[term] += 1
            if counts[term] == 1:
                unique_terms += 1

            while position - positions[left][0] > 120:
                left_term = positions[left][1]
                counts[left_term] -= 1
                if counts[left_term] == 0:
                    unique_terms -= 1
                left += 1

            best = max(best, unique_terms / len(terms))

        if best > 0 and len(positions) >= len(terms):
            best = min(1.0, best + 0.15)
        scores[idx] = best
    return scores


def _is_identity_question(question: str) -> bool:
    lower = question.lower().strip()
    return lower.startswith(
        ("who is ", "who was ", "what is ", "what was ", "tell me about ", "describe ")
    )


def _is_fate_question(question: str) -> bool:
    lower = normalize_query(question)
    return lower.startswith(("what happened to ", "what happens to ", "how did ")) or any(
        term in STATUS_TERMS for term in tokenize_text(lower)
    )


def _extract_subject(question: str) -> str | None:
    cleaned = question.strip().rstrip("?.! ")
    patterns = (
        r"^(?:who|what)\s+(?:is|was)\s+(.+)$",
        r"^(?:tell me about|describe)\s+(.+)$",
        r"^(?:what happened to|what happens to)\s+(.+)$",
        r"^how did\s+(.+?)\s+die$",
        r"^how was\s+(.+?)\s+killed$",
        r"^is\s+(.+?)\s+dead$",
    )
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        subject = match.group(1).strip(" ,")
        subject = re.sub(
            r"\b(?:in|from)\s+the\s+(?:books|novels|series|story)\b$",
            "",
            subject,
            flags=re.IGNORECASE,
        )
        subject = re.sub(r"\s+", " ", subject).strip(" ,")
        if subject:
            return subject
    return None


def _subject_phrase(question: str) -> str | None:
    subject = _extract_subject(question)
    if subject:
        return normalize_query(subject)
    terms = _query_terms(question)
    if not terms:
        return None
    return " ".join(terms[: min(4, len(terms))])


def _subject_aliases(question: str) -> tuple[list[str], list[str]]:
    canonical = _subject_phrase(question)
    raw_subject = _extract_subject(question)
    strong: list[str] = []
    weak: list[str] = []
    seen: set[str] = set()

    def add(alias: str, bucket: list[str]) -> None:
        cleaned = re.sub(r"\s+", " ", alias.lower()).strip(" ,.?!")
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        bucket.append(cleaned)

    if canonical:
        add(canonical, strong)

    if raw_subject:
        normalized_raw = normalize_query(raw_subject)
        bucket = strong if len(normalized_raw.split()) >= 2 else weak
        add(raw_subject, bucket)

    if canonical:
        for pattern, replacement in QUERY_NORMALIZATIONS.items():
            if replacement != canonical:
                continue
            alias = pattern.replace(r"\b", "").replace("\\", "")
            add(alias, strong if len(alias.split()) >= 2 else weak)
        for alias in SUBJECT_ALIAS_OVERRIDES.get(canonical, []):
            add(alias, strong)

    return strong, weak


def _best_alias_score(text: str, strong_aliases: list[str], weak_aliases: list[str]) -> float:
    lower = text.lower()
    best = 0.0

    for alias in strong_aliases:
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            score = 0.96
            if lower.startswith(alias):
                score = 1.0
            elif re.search(rf"(?:^|[,:;(\[])\s*{re.escape(alias)}\b", lower):
                score = 0.99
            best = max(best, score)

    for alias in weak_aliases:
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            best = max(best, 0.74)

    return best


def _has_hypothetical_language(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in HYPOTHETICAL_MARKERS)


def direct_subject_match_scores(question: str, chunks: list[dict]) -> np.ndarray:
    strong_aliases, weak_aliases = _subject_aliases(question)
    if not strong_aliases and not weak_aliases:
        return np.zeros(len(chunks), dtype=np.float32)

    scores = np.zeros(len(chunks), dtype=np.float32)
    for idx, chunk in enumerate(chunks):
        text_lower = chunk["text"].lower()
        scores[idx] = _best_alias_score(text_lower, strong_aliases, weak_aliases)
    return scores


def identity_match_scores(question: str, chunks: list[dict]) -> np.ndarray:
    if not _is_identity_question(question):
        return np.zeros(len(chunks), dtype=np.float32)

    strong_aliases, weak_aliases = _subject_aliases(question)
    if not strong_aliases and not weak_aliases:
        return np.zeros(len(chunks), dtype=np.float32)

    descriptor_pattern = "|".join(sorted(IDENTITY_DESCRIPTOR_TERMS))

    scores = np.zeros(len(chunks), dtype=np.float32)
    for idx, chunk in enumerate(chunks):
        text_lower = chunk["text"].lower()
        alias_score = _best_alias_score(text_lower, strong_aliases, weak_aliases)
        if alias_score <= 0:
            continue

        for alias in strong_aliases + weak_aliases:
            appositive = re.compile(
                rf"\b{re.escape(alias)}\b(?:,|:|—|-).{{0,100}}\b({descriptor_pattern})\b"
            )
            copula = re.compile(rf"\b{re.escape(alias)}\b.{{0,40}}\b(is|was)\b")
            backward = re.compile(rf"\b(is|was)\b.{{0,40}}\b{re.escape(alias)}\b")
            if appositive.search(text_lower):
                scores[idx] = 1.0
                break
            if copula.search(text_lower) or backward.search(text_lower):
                scores[idx] = max(scores[idx], 0.82)

        if scores[idx] == 0 and any(
            re.search(rf"\b{term}\b", text_lower) for term in IDENTITY_DESCRIPTOR_TERMS
        ):
            scores[idx] = max(scores[idx], 0.56 + 0.28 * alias_score)
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


def profile_match_scores(question: str, chunks: list[dict]) -> np.ndarray:
    if not _is_identity_question(question):
        return np.zeros(len(chunks), dtype=np.float32)

    strong_aliases, weak_aliases = _subject_aliases(question)
    if not strong_aliases and not weak_aliases:
        return np.zeros(len(chunks), dtype=np.float32)

    descriptor_pattern = "|".join(sorted(IDENTITY_DESCRIPTOR_TERMS))
    scores = np.zeros(len(chunks), dtype=np.float32)
    for idx, chunk in enumerate(chunks):
        text_lower = chunk["text"].lower()
        alias_score = _best_alias_score(text_lower, strong_aliases, weak_aliases)
        if alias_score < 0.74:
            continue

        best = 0.0
        for alias in strong_aliases + weak_aliases:
            if re.search(
                rf"\b{re.escape(alias)}\b[^.!?\n]{{0,140}}\b({descriptor_pattern})\b",
                text_lower,
            ):
                best = 1.0
                break
            if re.search(
                rf"\b({descriptor_pattern})\b[^.!?\n]{{0,100}}\b{re.escape(alias)}\b",
                text_lower,
            ):
                best = max(best, 0.9)
        if best == 0.0 and chunk.get("section_type") == "appendix":
            best = 0.78 + 0.16 * alias_score
        scores[idx] = min(best, 1.0)
    return scores


def event_match_scores(question: str, chunks: list[dict]) -> np.ndarray:
    if not _is_fate_question(question):
        return np.zeros(len(chunks), dtype=np.float32)

    strong_aliases, weak_aliases = _subject_aliases(question)
    if not strong_aliases and not weak_aliases:
        return np.zeros(len(chunks), dtype=np.float32)

    event_pattern = "|".join(sorted(EVENT_TERMS))
    scores = np.zeros(len(chunks), dtype=np.float32)
    for idx, chunk in enumerate(chunks):
        text_lower = chunk["text"].lower()
        alias_score = _best_alias_score(text_lower, strong_aliases, weak_aliases)
        if alias_score < 0.74:
            continue

        best = 0.0
        for alias in strong_aliases + weak_aliases:
            for match in re.finditer(rf"\b{re.escape(alias)}\b", text_lower):
                window = _trim_window(chunk["text"], match.start(), radius=125).lower()
                event_hits = sum(
                    1 for term in EVENT_TERMS if re.search(rf"\b{re.escape(term)}\b", window)
                )
                if event_hits == 0:
                    continue
                decisive_hits = sum(
                    1
                    for term in DECISIVE_EVENT_TERMS
                    if re.search(rf"\b{re.escape(term)}\b", window)
                )
                local_score = 0.72 + 0.06 * min(event_hits, 3) + 0.1 * min(decisive_hits, 2)
                if _has_hypothetical_language(window):
                    local_score -= 0.25
                best = max(best, local_score)

        scores[idx] = min(best, 1.0)
    return scores


def source_type_match_scores(question: str, chunks: list[dict]) -> np.ndarray:
    scores = np.zeros(len(chunks), dtype=np.float32)
    identity = _is_identity_question(question)
    fate = _is_fate_question(question)

    for idx, chunk in enumerate(chunks):
        section_type = chunk.get("section_type", "story")
        if section_type == "front_matter":
            scores[idx] = 0.08
        elif section_type == "back_matter":
            scores[idx] = 0.22
        elif identity:
            scores[idx] = 1.0 if section_type == "appendix" else 0.82
        elif fate:
            scores[idx] = 1.0 if section_type == "story" else 0.7
        else:
            scores[idx] = 1.0 if section_type == "story" else 0.78
    return scores


def split_text_units(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", normalized)
        if part.strip()
    ]
    if len(sentences) > 1:
        return sentences

    parts = [
        part.strip(" ,;:-")
        for part in re.split(r"\s+[—-](?=\w)|\s+[—-]\s+|,\s+[—-]\s+|•", normalized)
        if part.strip(" ,;:-")
    ]
    return parts or [normalized]


def _trim_window(text: str, center: int, radius: int = 110) -> str:
    start = max(0, center - radius)
    end = min(len(text), center + radius)
    while start > 0 and text[start] not in ".!?;:\n":
        start -= 1
    while end < len(text) and text[end - 1] not in ".!?;:\n":
        end += 1
        if end >= len(text):
            end = len(text)
            break
    return text[start:end].strip(" \n,;:-")


def _unit_score(question: str, chunk: dict, text: str) -> float:
    lower = text.lower()
    query_terms = _query_terms(question)
    if not query_terms:
        return 0.0

    strong_aliases, weak_aliases = _subject_aliases(question)
    alias_score = _best_alias_score(lower, strong_aliases, weak_aliases)
    term_hits = sum(1 for term in query_terms if re.search(rf"\b{re.escape(term)}\b", lower))
    score = 0.45 * (term_hits / len(query_terms))

    if (strong_aliases or weak_aliases) and (_is_identity_question(question) or _is_fate_question(question)):
        min_alias_score = 0.74 if strong_aliases else 0.62
        if alias_score < min_alias_score:
            return 0.0
        score += 0.8 * alias_score

    if _is_identity_question(question):
        descriptor_hits = sum(
            1 for term in IDENTITY_DESCRIPTOR_TERMS if re.search(rf"\b{re.escape(term)}\b", lower)
        )
        if descriptor_hits == 0 and chunk.get("section_type") != "appendix":
            return 0.0
        score += min(0.6, 0.12 * descriptor_hits)
        alias_positions = [
            match.start()
            for alias in strong_aliases + weak_aliases
            for match in re.finditer(rf"\b{re.escape(alias)}\b", lower)
        ]
        if alias_positions:
            first_alias = min(alias_positions)
            if first_alias <= 40:
                score += 0.28
            elif first_alias <= 90:
                score += 0.14
        if any(
            re.search(
                rf"\b{re.escape(alias)}\b[^.!?\n]{{0,60}}\b"
                r"(called|bastard|sworn brother|son|daughter|queen|king|khaleesi|mother|widow)\b",
                lower,
            )
            for alias in strong_aliases + weak_aliases
        ):
            score += 0.24
        if chunk.get("section_type") == "appendix":
            score += 0.16

    if _is_fate_question(question):
        event_hits = sum(1 for term in EVENT_TERMS if re.search(rf"\b{re.escape(term)}\b", lower))
        if event_hits == 0:
            return 0.0
        score += min(0.7, 0.12 * event_hits)
        if _has_hypothetical_language(lower):
            score -= 0.4
        if chunk.get("section_type") == "story":
            score += 0.1

    for phrase, size in _query_phrases(question):
        if re.search(rf"\b{re.escape(phrase)}\b", lower):
            score += 0.08 * size

    if chunk.get("section_type") == "front_matter":
        score -= 0.3

    return max(0.0, min(score, 3.0))


def _is_duplicate_snippet(normalized_text: str, seen_texts: list[str]) -> bool:
    for existing in seen_texts:
        if normalized_text == existing or normalized_text in existing or existing in normalized_text:
            return True
    return False


def extract_evidence(question: str, hits: list[dict], max_units: int = 8) -> list[dict]:
    query_terms = _query_terms(question)
    strong_aliases, weak_aliases = _subject_aliases(question)
    identity = _is_identity_question(question)
    fate = _is_fate_question(question)
    candidates: list[dict] = []
    seen_texts: list[str] = []

    for hit in hits:
        hit_text = hit["text"]
        snippet_pool = split_text_units(hit_text)
        if identity or fate:
            snippet_pool = [snippet for snippet in snippet_pool if len(snippet) <= 320]
        for alias in strong_aliases + weak_aliases + query_terms[:3]:
            if not alias:
                continue
            for match in re.finditer(rf"\b{re.escape(alias)}\b", hit_text.lower()):
                snippet_pool.append(
                    _trim_window(hit_text, match.start(), radius=120 if fate else 105)
                )

        for snippet in snippet_pool:
            cleaned = re.sub(r"\s+", " ", snippet).strip(" ,;:-")
            if len(cleaned) < 40:
                continue
            lower = cleaned.lower()
            alias_score = _best_alias_score(lower, strong_aliases, weak_aliases)
            if (identity or fate) and (strong_aliases or weak_aliases):
                min_alias_score = 0.74 if strong_aliases else 0.62
                if alias_score < min_alias_score:
                    continue
            if identity and not any(
                re.search(rf"\b{term}\b", lower) for term in IDENTITY_DESCRIPTOR_TERMS
            ):
                if hit.get("section_type") != "appendix":
                    continue
            if fate:
                if not any(re.search(rf"\b{term}\b", lower) for term in EVENT_TERMS):
                    continue
                if _has_hypothetical_language(lower):
                    continue
            score = _unit_score(question, hit, cleaned)
            if score <= 0:
                continue
            normalized = re.sub(r"[^a-z0-9]+", " ", lower).strip()
            if not normalized or _is_duplicate_snippet(normalized, seen_texts):
                continue
            seen_texts.append(normalized)
            candidates.append(
                {
                    "text": cleaned,
                    "source": format_source_label(hit),
                    "score": score,
                    "alias_score": alias_score,
                    "section_type": hit.get("section_type", "story"),
                }
            )
            if len(candidates) >= 72:
                break
        if len(candidates) >= 72:
            break

    candidates.sort(
        key=lambda item: (
            item["score"],
            item["alias_score"],
            1.0 if item["section_type"] == "story" and _is_fate_question(question) else 0.0,
            1.0 if item["section_type"] == "appendix" and _is_identity_question(question) else 0.0,
            len(item["text"]),
        ),
        reverse=True,
    )

    selected: list[dict] = []
    source_counts: Counter[str] = Counter()
    per_source_limit = 1 if (identity or fate) else 2
    for candidate in candidates:
        if source_counts[candidate["source"]] >= per_source_limit:
            continue
        selected.append(
            {
                "text": candidate["text"],
                "source": candidate["source"],
                "score": candidate["score"],
                "section_type": candidate["section_type"],
            }
        )
        source_counts[candidate["source"]] += 1
        if len(selected) >= max_units:
            break
    return selected


def retrieve_chunks(
    question: str,
    index: LoadedIndex,
    embedding_model: str,
    top_k: int = 8,
    chat_history: list[dict] | None = None,
) -> list[dict]:
    query_variants = build_query_variants(question, chat_history)

    semantic_sets: list[np.ndarray] = []
    bm25_sets: list[np.ndarray] = []
    lexical_sets: list[np.ndarray] = []
    phrase_sets: list[np.ndarray] = []
    proximity_sets: list[np.ndarray] = []
    direct_sets: list[np.ndarray] = []
    identity_sets: list[np.ndarray] = []
    profile_sets: list[np.ndarray] = []
    status_sets: list[np.ndarray] = []
    event_sets: list[np.ndarray] = []
    source_type_sets: list[np.ndarray] = []
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
        phrase = phrase_match_scores(query, index.chunks)
        proximity = proximity_match_scores(query, index.chunks)
        direct = direct_subject_match_scores(query, index.chunks)
        identity = identity_match_scores(query, index.chunks)
        profile = profile_match_scores(query, index.chunks)
        status = status_match_scores(query, index.chunks)
        event = event_match_scores(query, index.chunks)
        source_type = source_type_match_scores(query, index.chunks)

        if _is_fate_question(query):
            combined = (
                0.20 * semantic_norm
                + 0.14 * bm25_norm
                + 0.08 * lexical
                + 0.12 * phrase
                + 0.08 * proximity
                + 0.12 * status
                + 0.16 * direct
                + 0.24 * event
                + 0.06 * source_type
            )
        elif _is_identity_question(query):
            combined = (
                0.16 * semantic_norm
                + 0.12 * bm25_norm
                + 0.08 * lexical
                + 0.14 * phrase
                + 0.08 * proximity
                + 0.12 * direct
                + 0.12 * identity
                + 0.22 * profile
                + 0.08 * source_type
            )
        else:
            combined = (
                0.34 * semantic_norm
                + 0.20 * bm25_norm
                + 0.10 * lexical
                + 0.16 * phrase
                + 0.12 * proximity
                + 0.06 * direct
                + 0.02 * source_type
            )

        semantic_sets.append(semantic_norm)
        bm25_sets.append(bm25_norm)
        lexical_sets.append(lexical)
        phrase_sets.append(phrase)
        proximity_sets.append(proximity)
        direct_sets.append(direct)
        identity_sets.append(identity)
        profile_sets.append(profile)
        status_sets.append(status)
        event_sets.append(event)
        source_type_sets.append(source_type)
        final_sets.append(combined)

    semantic_scores = np.max(np.vstack(semantic_sets), axis=0)
    bm25_scores = np.max(np.vstack(bm25_sets), axis=0)
    lexical_scores = np.max(np.vstack(lexical_sets), axis=0)
    phrase_scores = np.max(np.vstack(phrase_sets), axis=0)
    proximity_scores = np.max(np.vstack(proximity_sets), axis=0)
    direct_scores = np.max(np.vstack(direct_sets), axis=0)
    identity_scores = np.max(np.vstack(identity_sets), axis=0)
    profile_scores = np.max(np.vstack(profile_sets), axis=0)
    status_scores = np.max(np.vstack(status_sets), axis=0)
    event_scores = np.max(np.vstack(event_sets), axis=0)
    source_type_scores = np.max(np.vstack(source_type_sets), axis=0)
    final_scores = np.max(np.vstack(final_sets), axis=0)

    top_k = max(1, min(top_k, len(index.chunks)))
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

    results: list[dict] = []
    for idx in selected[:top_k]:
        chunk = index.chunks[int(idx)]
        results.append(
            {
                "id": chunk["id"],
                "page": chunk["page"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "book": chunk.get("book", "Unknown"),
                "text": chunk["text"],
                "score": float(final_scores[idx]),
                "semantic_score": float(semantic_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "lexical_score": float(lexical_scores[idx]),
                "phrase_score": float(phrase_scores[idx]),
                "proximity_score": float(proximity_scores[idx]),
                "direct_subject_score": float(direct_scores[idx]),
                "identity_score": float(identity_scores[idx]),
                "profile_score": float(profile_scores[idx]),
                "status_score": float(status_scores[idx]),
                "event_score": float(event_scores[idx]),
                "source_type_score": float(source_type_scores[idx]),
                "section_type": chunk.get("section_type", "story"),
            }
        )

    if _is_identity_question(question):
        results.sort(
            key=lambda hit: (
                hit["profile_score"],
                hit["direct_subject_score"],
                hit["identity_score"],
                hit["source_type_score"],
                hit["phrase_score"],
                hit["score"],
            ),
            reverse=True,
        )
    elif _is_fate_question(question):
        results.sort(
            key=lambda hit: (
                hit["event_score"],
                hit["direct_subject_score"],
                hit["status_score"],
                hit["source_type_score"],
                hit["phrase_score"],
                hit["score"],
            ),
            reverse=True,
        )
    return results


def format_source_label(chunk: dict) -> str:
    page_start = int(chunk.get("page_start", chunk.get("page", 0)))
    page_end = int(chunk.get("page_end", page_start))
    if page_start == page_end:
        page_label = f"page {page_start}"
    else:
        page_label = f"pages {page_start}-{page_end}"

    book = chunk.get("book")
    if book and book != "Unknown":
        return f"{book}, {page_label}"
    return page_label


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] ({format_source_label(chunk)})\n{chunk['text']}")
    return "\n\n".join(blocks)


def build_evidence_block(evidence: list[dict]) -> str:
    blocks = []
    for idx, item in enumerate(evidence, start=1):
        blocks.append(f"[E{idx}] ({item['source']}) {item['text']}")
    return "\n".join(blocks)


def _display_subject(question: str) -> str:
    subject = _extract_subject(question) or _subject_phrase(question) or "This character"
    return " ".join(part.capitalize() for part in subject.split())


def _fact_fragment(text: str, aliases: list[str], max_words: int = 28) -> str:
    lower = text.lower()
    fragment = re.sub(r"\s+", " ", text).strip(" ,;:-")
    for alias in aliases:
        match = re.search(rf"\b{re.escape(alias)}\b", lower)
        if not match:
            continue
        fragment = _trim_window(text, match.start(), radius=100)
        fragment = re.sub(r"\s+", " ", fragment).strip(" ,;:-")
        break

    words = fragment.split()
    if len(words) > max_words:
        fragment = " ".join(words[:max_words]).rstrip(" ,;:-")
    if fragment and fragment[0].islower():
        fragment = fragment[0].upper() + fragment[1:]
    return fragment


def generate_extract_answer(question: str, evidence: list[dict]) -> str | None:
    identity = _is_identity_question(question)
    fate = _is_fate_question(question)
    if not evidence or not (identity or fate):
        return None

    strong_aliases, weak_aliases = _subject_aliases(question)
    aliases = strong_aliases + weak_aliases
    if not aliases:
        return None

    fragments: list[tuple[int, str]] = []
    seen: set[str] = set()
    for idx, item in enumerate(evidence, start=1):
        fragment = _fact_fragment(
            item["text"],
            aliases,
            max_words=30 if identity else 24,
        )
        normalized = re.sub(r"[^a-z0-9]+", " ", fragment.lower()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        fragments.append((idx, fragment))

    if not fragments:
        return None

    sentences: list[str] = []
    subject = _display_subject(question)

    if identity:
        first_idx, first_fragment = fragments[0]
        sentences.append(
            f"{subject} is described in one retrieved entry as {first_fragment} [E{first_idx}]."
        )
        for idx, fragment in fragments[1:4]:
            sentences.append(f"Another entry adds {fragment} [E{idx}].")
    elif fate:
        first_idx, first_fragment = fragments[0]
        sentences.append(f"The clearest retrieved passage says {first_fragment} [E{first_idx}].")
        for idx, fragment in fragments[1:4]:
            sentences.append(f"Another passage adds {fragment} [E{idx}].")

    return postprocess_answer(" ".join(sentences))


def select_context_chunks(question: str, hits: list[dict], max_chunks: int = 6) -> list[dict]:
    if _is_identity_question(question):
        focused = [
            hit
            for hit in hits
            if hit.get("profile_score", 0.0) >= 0.78
            or (
                hit.get("direct_subject_score", 0.0) >= 0.9
                and hit.get("identity_score", 0.0) >= 0.72
            )
        ]
        if focused:
            return focused[: max(2, min(max_chunks, len(focused)))]

    if _is_fate_question(question):
        focused = [
            hit
            for hit in hits
            if hit.get("event_score", 0.0) >= 0.78
            or (
                hit.get("direct_subject_score", 0.0) >= 0.9
                and hit.get("status_score", 0.0) >= 0.55
            )
        ]
        if focused:
            return focused[: max(2, min(max_chunks, len(focused)))]

    return hits[:max_chunks]


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


def postprocess_answer(answer: str) -> str:
    answer = re.sub(
        r"\bborn ([a-z0-9-]+) years ago\b",
        r"described as \1 years old in the cited passage",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"\ba bastard of ([a-z0-9-]+) years\b",
        r"a bastard described as \1 years old",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"\ba bastard described as ([a-z0-9-]+) years old\b",
        r"a \1-year-old bastard",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(r"\bThis suggests that[^.]*\.\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\bThis implies that[^.]*\.\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(
        r"\bHowever, it is (?:worth )?noting that[^.]*\.\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"\bit is not until later in the series[^.]*\.\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"\b(?:It is|It's|It’s) worth noting(?: that)?[^.]*\.\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(r"\bNote:\s.*$", "", answer, flags=re.IGNORECASE)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer)
        if sentence.strip()
    ]
    cited_sentences = [
        sentence for sentence in sentences if re.search(r"\[(?:E)?\d+\]", sentence)
    ]
    if cited_sentences:
        answer = " ".join(cited_sentences)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    chat_model: str,
    chat_history: list[dict] | None = None,
) -> str:
    client = ollama.Client()
    if _is_identity_question(question):
        evidence_limit = 4
    elif _is_fate_question(question):
        evidence_limit = 5
    else:
        evidence_limit = 6
    evidence = extract_evidence(question, retrieved_chunks, max_units=evidence_limit)
    evidence_block = build_evidence_block(evidence)

    prompt = (
        "High-signal evidence snippets:\n"
        f"{evidence_block or 'None'}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer using only the evidence snippets above. "
        "If a claim is not present in those snippets, say you do not know from the provided books. "
        "Prefer the most direct identity or event evidence and ignore incidental mentions or hypotheticals. "
        "Every factual sentence must include an inline citation such as [E1] or [E2]. "
        "If a sentence cannot be cited, do not include it. "
        "If the question depends on timeline, explicitly say so and cite both early and later evidence. "
        "Never convert an age into a phrase like 'born X years ago'; keep age and chronology phrasing close to the source text. "
        "Do not infer motives, hidden causes, exact parentage, locations of imprisonment, heirs, later-book revelations, or TV-series details unless an evidence snippet states them directly. "
        "If context is insufficient for a claim, say that clearly instead of guessing. "
        "If the snippets are appendix-style character entries, rewrite them into natural prose without adding new facts. "
        "Write a fuller answer by default, usually one solid paragraph of about 6-10 sentences unless the user asks for brevity."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_prepare_history(chat_history))
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat(
            model=chat_model,
            messages=messages,
            options={"temperature": 0.0, "num_predict": 650, "repeat_penalty": 1.08},
        )
    except Exception as exc:
        raise RuntimeError(OLLAMA_CONNECTIVITY_ERROR) from exc
    return postprocess_answer(response["message"]["content"].strip())


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
        chat_history=chat_history,
    )
    context_hits = select_context_chunks(question, hits, max_chunks=min(top_k, 6))
    answer = generate_answer(
        question=question,
        retrieved_chunks=context_hits,
        chat_model=chat_model,
        chat_history=chat_history,
    )
    return answer, hits
