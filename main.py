from __future__ import annotations

import argparse
from pathlib import Path

from rag_engine import answer_with_rag, load_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Terminal chatbot over the Game of Thrones RAG index."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/got_index"),
        help="Directory containing vectors.npy, chunks.json and metadata.json",
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default="llama3.1:8b",
        help="Ollama chat model to use for answering questions",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many chunks to retrieve per question",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = load_index(args.index_dir)
    embedding_model = index.metadata["embedding_model"]

    print("Game of Thrones RAG chatbot")
    print(f"Index: {args.index_dir}")
    print(f"Embedding model: {embedding_model}")
    print(f"Chat model: {args.chat_model}")
    print("Type 'exit' to quit.\n")

    history: list[dict] = []
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        answer, hits = answer_with_rag(
            question=question,
            index=index,
            chat_model=args.chat_model,
            top_k=args.top_k,
            chat_history=history[-8:],
        )
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        print(f"\nAssistant: {answer}\n")
        print("Sources:")
        for i, hit in enumerate(hits, start=1):
            print(f"[{i}] page {hit['page']} (score={hit['score']:.3f})")
        print()


if __name__ == "__main__":
    main()
