from __future__ import annotations

import argparse
from pathlib import Path

from rag_engine import DEFAULT_INDEX_DIR, build_index, find_pdf_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local RAG index from the Game of Thrones PDF using Ollama embeddings."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Path to the source PDF. If omitted, the first PDF in the current directory is used.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Output directory for the index files.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=180,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=80,
        help="Skip chunks smaller than this character count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf or find_pdf_file(Path.cwd())
    if not pdf_path:
        raise FileNotFoundError("No PDF found. Pass --pdf path/to/file.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    metadata = build_index(
        pdf_path=pdf_path,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_chars,
        batch_size=args.batch_size,
        progress=print,
    )

    print("\nDone.")
    print(f"Chunks: {metadata['chunk_count']}")
    print(f"Index directory: {args.index_dir}")


if __name__ == "__main__":
    main()
