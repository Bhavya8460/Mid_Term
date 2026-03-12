from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_engine import DEFAULT_INDEX_DIR, answer_with_rag, build_index, find_pdf_file, load_index

st.set_page_config(page_title="GOT RAG Chatbot", page_icon="🐉", layout="wide")


@st.cache_resource(show_spinner=False)
def get_loaded_index(index_dir_str: str):
    return load_index(Path(index_dir_str))


def _model_history(messages: list[dict], max_messages: int = 8) -> list[dict]:
    history: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        if content.startswith("Ask me anything about the Game of Thrones books"):
            continue
        history.append({"role": role, "content": content})
    return history[-max_messages:]


def _build_index_ui(pdf_path: Path, index_dir: Path, embedding_model: str) -> None:
    with st.spinner("Building index. This can take a while on first run..."):
        build_index(
            pdf_path=pdf_path,
            index_dir=index_dir,
            embedding_model=embedding_model,
            progress=lambda message: st.write(message),
        )
    st.success("Index built successfully.")


def main() -> None:
    st.title("Game of Thrones RAG Chatbot")
    st.caption("Ask questions about the books. Answers are grounded in your PDF.")

    default_pdf = find_pdf_file(Path.cwd())
    default_pdf_str = str(default_pdf) if default_pdf else ""

    with st.sidebar:
        st.header("Settings")
        pdf_path_input = st.text_input("PDF path", value=default_pdf_str)
        index_dir_input = st.text_input("Index directory", value=str(DEFAULT_INDEX_DIR))
        embedding_model = st.text_input("Embedding model", value="nomic-embed-text")
        chat_model = st.text_input("Chat model", value="llama3.1:8b")
        top_k = st.slider("Top K chunks", min_value=4, max_value=20, value=8)
        rebuild = st.button("Build / Rebuild Index")

    pdf_path = Path(pdf_path_input) if pdf_path_input else None
    index_dir = Path(index_dir_input)

    if rebuild:
        if not pdf_path or not pdf_path.exists():
            st.error("Please provide a valid PDF path before building the index.")
            return
        _build_index_ui(pdf_path=pdf_path, index_dir=index_dir, embedding_model=embedding_model)
        get_loaded_index.clear()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask me anything about the Game of Thrones books after the index is built."
                ),
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Your question")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            index = get_loaded_index(str(index_dir.resolve()))
            history = _model_history(st.session_state.messages[:-1], max_messages=8)
            answer, hits = answer_with_rag(
                question=question,
                index=index,
                chat_model=chat_model,
                top_k=top_k,
                chat_history=history,
            )
            st.markdown(answer)
            with st.expander("Retrieved sources"):
                for i, hit in enumerate(hits, start=1):
                    st.markdown(
                        f"**[{i}] page {hit['page']}** "
                        f"(score={hit['score']:.3f}, dense={hit['semantic_score']:.3f}, "
                        f"bm25={hit['bm25_score']:.3f}, keyword={hit['lexical_score']:.3f}, "
                        f"status={hit['status_score']:.3f})"
                    )
                    st.write(hit["text"])
        except Exception as exc:
            st.error(
                "Could not answer the question. Make sure Ollama is running and the index exists."
            )
            st.exception(exc)
            answer = "I couldn't answer due to a runtime issue."

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
