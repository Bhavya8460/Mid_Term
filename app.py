from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from rag_engine import (
    DEFAULT_INDEX_DIR,
    answer_with_rag,
    build_index,
    find_pdf_file,
    format_source_label,
    load_index,
)

st.set_page_config(page_title="GOT RAG Chatbot", page_icon="🐉", layout="wide")

BACKGROUND_IMAGE = Path(__file__).with_name("Background.jpg")


@st.cache_resource(show_spinner=False)
def get_loaded_index(index_dir_str: str):
    return load_index(Path(index_dir_str))


@st.cache_data(show_spinner=False)
def _image_to_base64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def _apply_theme() -> None:
    background_style = "background: linear-gradient(180deg, #f3ddda 0%, #d4e5e8 34%, #0f1320 100%);"
    if BACKGROUND_IMAGE.exists():
        encoded = _image_to_base64(str(BACKGROUND_IMAGE))
        background_style = (
            "background-image: linear-gradient("
            "180deg, rgba(243, 221, 218, 0.16) 0%, rgba(212, 229, 232, 0.10) 28%, "
            "rgba(11, 15, 25, 0.76) 62%, rgba(8, 11, 19, 0.92) 100%"
            f"), url('data:image/jpeg;base64,{encoded}');"
            "background-size: cover;"
            "background-position: center top;"
            "background-attachment: fixed;"
        )

    st.markdown(
        f"""
        <style>
        :root {{
            --stark-ink: #f7f5f2;
            --stark-mist: #d9eaec;
            --stark-steel: #9eb6be;
            --stark-panel: rgba(9, 13, 22, 0.74);
            --stark-panel-strong: rgba(9, 13, 22, 0.88);
            --stark-line: rgba(220, 234, 236, 0.22);
            --stark-accent: #d7ecef;
            --stark-rose: #efd8d1;
            --stark-shadow: 0 24px 60px rgba(0, 0, 0, 0.32);
        }}

        .stApp {{
            {background_style}
            color: var(--stark-ink);
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stToolbar"] {{
            right: 1rem;
        }}

        [data-testid="stAppViewContainer"] > .main {{
            background: linear-gradient(
                180deg,
                rgba(8, 11, 19, 0.02) 0%,
                rgba(8, 11, 19, 0.18) 12%,
                rgba(8, 11, 19, 0.42) 26%,
                rgba(8, 11, 19, 0.68) 42%,
                rgba(8, 11, 19, 0.88) 100%
            );
        }}

        .block-container {{
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 6rem;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(9, 13, 22, 0.90) 0%, rgba(9, 13, 22, 0.96) 100%);
            border-right: 1px solid var(--stark-line);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--stark-ink);
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
        [data-testid="stSidebar"] .stCheckbox p,
        [data-testid="stSidebar"] .stSlider p,
        [data-testid="stSidebar"] small {{
            color: rgba(247, 245, 242, 0.84) !important;
        }}

        [data-testid="stSidebar"] .stTextInput > div > div,
        [data-testid="stSidebar"] .stNumberInput > div > div,
        [data-testid="stSidebar"] .stSelectbox > div > div,
        [data-testid="stSidebar"] div[data-baseweb="input"] > div,
        [data-testid="stSidebar"] div[data-baseweb="base-input"] > div,
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(215, 236, 239, 0.18) !important;
        }}

        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] div[contenteditable="true"],
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] input,
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
        [data-testid="stSidebar"] div[data-baseweb="input"] input,
        [data-testid="stSidebar"] div[data-baseweb="base-input"] input {{
            color: #fffaf7 !important;
            -webkit-text-fill-color: #fffaf7 !important;
            caret-color: #fffaf7 !important;
            background: transparent !important;
        }}

        [data-testid="stSidebar"] .stTextInput input::placeholder,
        [data-testid="stSidebar"] .stNumberInput input::placeholder,
        [data-testid="stSidebar"] textarea::placeholder,
        [data-testid="stSidebar"] div[data-baseweb="input"] input::placeholder {{
            color: rgba(247, 245, 242, 0.44) !important;
            -webkit-text-fill-color: rgba(247, 245, 242, 0.44) !important;
        }}

        [data-testid="stSidebar"] .stCheckbox svg,
        [data-testid="stSidebar"] .stSlider svg,
        [data-testid="stSidebar"] button {{
            color: #fffaf7 !important;
        }}

        h1, h2, h3 {{
            color: var(--stark-ink) !important;
            letter-spacing: 0.02em;
        }}

        [data-testid="stMarkdownContainer"] p,
        .stCaption,
        label {{
            color: rgba(247, 245, 242, 0.88);
        }}

        .hero-shell {{
            position: relative;
            overflow: hidden;
            margin: 0 0 1.4rem 0;
            padding: 2rem 2rem 1.7rem 2rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 26px;
            background:
                linear-gradient(135deg, rgba(11, 15, 25, 0.80) 0%, rgba(11, 15, 25, 0.60) 44%, rgba(215, 236, 239, 0.10) 100%);
            box-shadow: var(--stark-shadow);
            backdrop-filter: blur(12px);
        }}

        .hero-shell::after {{
            content: "";
            position: absolute;
            inset: auto -8% -32% auto;
            width: 240px;
            height: 240px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(239, 216, 209, 0.30) 0%, rgba(239, 216, 209, 0) 70%);
            pointer-events: none;
        }}

        .hero-kicker {{
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 0.9rem;
            padding: 0.42rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(215, 236, 239, 0.24);
            background: rgba(215, 236, 239, 0.08);
            color: var(--stark-mist);
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }}

        .hero-title {{
            max-width: 12ch;
            margin: 0;
            font-size: clamp(2.4rem, 4.8vw, 4.7rem);
            line-height: 0.96;
            font-weight: 800;
            color: #fffaf7;
        }}

        .hero-copy {{
            max-width: 48rem;
            margin: 1rem 0 0 0;
            font-size: 1rem;
            line-height: 1.65;
            color: rgba(247, 245, 242, 0.86);
        }}

        .hero-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1.4rem;
        }}

        .hero-stat {{
            padding: 1rem 1.05rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}

        .hero-stat strong {{
            display: block;
            margin-bottom: 0.2rem;
            font-size: 1.02rem;
            color: #fffaf7;
        }}

        .hero-stat span {{
            color: rgba(247, 245, 242, 0.72);
            font-size: 0.9rem;
        }}

        .stAlert,
        [data-testid="stExpander"],
        [data-testid="stChatMessage"],
        [data-testid="stForm"],
        .stTextInput > div > div,
        .stNumberInput > div > div,
        .stSelectbox > div > div,
        .stSlider,
        [data-testid="stFileUploader"],
        .stButton > button,
        [data-testid="stChatInput"] {{
            border-radius: 20px;
        }}

        .stAlert {{
            background: rgba(9, 13, 22, 0.76);
            border: 1px solid var(--stark-line);
            box-shadow: var(--stark-shadow);
        }}

        [data-testid="stExpander"] {{
            background: rgba(9, 13, 22, 0.72);
            border: 1px solid var(--stark-line);
            box-shadow: var(--stark-shadow);
            overflow: hidden;
        }}

        [data-testid="stChatMessage"] {{
            background: rgba(9, 13, 22, 0.70);
            border: 1px solid var(--stark-line);
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.22);
            padding: 0.9rem 1rem;
            backdrop-filter: blur(10px);
        }}

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{
            color: rgba(247, 245, 242, 0.94);
        }}

        .stTextInput > div > div,
        .stNumberInput > div > div,
        .stSelectbox > div > div {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(215, 236, 239, 0.18);
        }}

        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {{
            color: var(--stark-ink);
        }}

        .stButton > button {{
            width: 100%;
            border: 1px solid rgba(215, 236, 239, 0.24);
            background: linear-gradient(135deg, rgba(215, 236, 239, 0.16) 0%, rgba(239, 216, 209, 0.14) 100%);
            color: #fffaf7;
            font-weight: 700;
            transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
        }}

        .stButton > button:hover {{
            border-color: rgba(239, 216, 209, 0.42);
            background: linear-gradient(135deg, rgba(215, 236, 239, 0.22) 0%, rgba(239, 216, 209, 0.22) 100%);
            transform: translateY(-1px);
        }}

        [data-testid="stChatInput"] {{
            background: rgba(9, 13, 22, 0.82);
            border: 1px solid var(--stark-line);
            box-shadow: var(--stark-shadow);
            backdrop-filter: blur(12px);
        }}

        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] > div > div,
        [data-testid="stChatInput"] div[data-baseweb="base-input"],
        [data-testid="stChatInput"] div[data-baseweb="textarea"] {{
            background: transparent !important;
        }}

        [data-testid="stChatInput"] input,
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] div[contenteditable="true"] {{
            color: #fffaf7 !important;
            -webkit-text-fill-color: #fffaf7 !important;
            caret-color: #fffaf7 !important;
            background: transparent !important;
        }}

        [data-testid="stChatInput"] input::placeholder,
        [data-testid="stChatInput"] textarea::placeholder {{
            color: rgba(247, 245, 242, 0.52) !important;
        }}

        [data-testid="stChatInput"] button {{
            color: #fffaf7 !important;
        }}

        @media (max-width: 900px) {{
            .hero-shell {{
                padding: 1.45rem 1.2rem 1.25rem 1.2rem;
            }}

            .hero-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-kicker">Winter Archive</div>
            <h1 class="hero-title">Ask the books, not the noise.</h1>
            <p class="hero-copy">
                This chatbot searches the five-book Game of Thrones corpus with filtered pages,
                stronger retrieval, and grounded citations. The battlefield stays in the background.
                The answers stay readable.
            </p>
            <div class="hero-grid">
                <div class="hero-stat">
                    <strong>Filtered Source</strong>
                    <span>Removes title, contents, copyright, dedication, and author pages.</span>
                </div>
                <div class="hero-stat">
                    <strong>Hybrid Retrieval</strong>
                    <span>Combines embeddings, BM25, phrase scoring, and follow-up-aware queries.</span>
                </div>
                <div class="hero-stat">
                    <strong>Grounded Answers</strong>
                    <span>Returns concise responses with citations back to the book and page range.</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _default_messages() -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": (
                "Ask me about the Game of Thrones books. "
                "For the best results, rebuild the index with page filtering enabled."
            ),
        }
    ]


def _model_history(messages: list[dict], max_messages: int = 8) -> list[dict]:
    history: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        if content.startswith("Ask me about the Game of Thrones books"):
            continue
        history.append({"role": role, "content": content})
    return history[-max_messages:]


def _build_index_ui(
    pdf_path: Path,
    index_dir: Path,
    embedding_model: str,
    filter_irrelevant_pages: bool,
    keep_appendices: bool,
) -> None:
    with st.spinner("Building index. This can take a while on first run..."):
        build_index(
            pdf_path=pdf_path,
            index_dir=index_dir,
            embedding_model=embedding_model,
            filter_irrelevant_pages=filter_irrelevant_pages,
            keep_appendices=keep_appendices,
            progress=lambda message: st.write(message),
        )
    st.success("Index built successfully.")


def _show_index_summary(index) -> None:
    metadata = index.metadata
    page_stats = metadata.get("page_stats", {})

    st.caption(f"Built: {metadata.get('created_at', 'unknown')}")
    st.write(f"Chunks: {metadata.get('chunk_count', len(index.chunks))}")
    if page_stats:
        st.write(
            f"Pages kept: {page_stats.get('kept_pages', 0)} / {page_stats.get('total_pages', 0)}"
        )
        st.write(f"Pages skipped: {page_stats.get('skipped_pages', 0)}")
    st.write(f"Embedding model: {metadata.get('embedding_model', 'unknown')}")
    st.write(f"Page filtering: {'on' if metadata.get('page_filtering') else 'off'}")
    st.write(f"Appendices kept: {'yes' if metadata.get('keep_appendices', True) else 'no'}")


def _show_rebuild_notice(index) -> None:
    metadata = index.metadata
    if "page_stats" not in metadata:
        st.info("This index was built with the older pipeline. Rebuild it to remove front matter.")
        return
    if not metadata.get("page_filtering", False):
        st.info("This index keeps front/back matter. Rebuild with page filtering enabled.")


def _render_sources(hits: list[dict]) -> None:
    with st.expander("Retrieved sources"):
        for i, hit in enumerate(hits, start=1):
            st.markdown(
                f"**[{i}] {format_source_label(hit)}** "
                f"(score={hit['score']:.3f}, dense={hit['semantic_score']:.3f}, "
                f"bm25={hit['bm25_score']:.3f}, keyword={hit['lexical_score']:.3f}, "
                f"phrase={hit['phrase_score']:.3f}, proximity={hit['proximity_score']:.3f}, "
                f"section={hit.get('section_type', 'story')})"
            )
            if hit.get("identity_score", 0.0) > 0:
                st.caption(f"Identity boost: {hit['identity_score']:.3f}")
            if hit.get("profile_score", 0.0) > 0:
                st.caption(f"Profile boost: {hit['profile_score']:.3f}")
            if hit.get("status_score", 0.0) > 0:
                st.caption(f"Status boost: {hit['status_score']:.3f}")
            if hit.get("event_score", 0.0) > 0:
                st.caption(f"Event boost: {hit['event_score']:.3f}")
            if hit.get("direct_subject_score", 0.0) > 0:
                st.caption(f"Subject match: {hit['direct_subject_score']:.3f}")
            if hit.get("source_type_score", 0.0) > 0:
                st.caption(f"Section boost: {hit['source_type_score']:.3f}")
            st.write(hit["text"])


def main() -> None:
    _apply_theme()
    _render_hero()

    default_pdf = find_pdf_file(Path.cwd())
    default_pdf_str = str(default_pdf) if default_pdf else ""

    with st.sidebar:
        st.header("Settings")
        pdf_path_input = st.text_input("PDF path", value=default_pdf_str)
        index_dir_input = st.text_input("Index directory", value=str(DEFAULT_INDEX_DIR))
        embedding_model = st.text_input("Embedding model", value="nomic-embed-text")
        chat_model = st.text_input("Chat model", value="llama3.1:8b")
        top_k = st.slider("Top K chunks", min_value=4, max_value=20, value=12)
        filter_irrelevant_pages = st.checkbox("Filter non-story pages", value=True)
        keep_appendices = st.checkbox("Keep appendices", value=True)
        rebuild = st.button("Build / Rebuild Index", use_container_width=True)
        clear_chat = st.button("Clear chat", use_container_width=True)

    pdf_path = Path(pdf_path_input) if pdf_path_input else None
    index_dir = Path(index_dir_input)

    if clear_chat:
        st.session_state.messages = _default_messages()

    if rebuild:
        if not pdf_path or not pdf_path.exists():
            st.error("Please provide a valid PDF path before building the index.")
            return
        _build_index_ui(
            pdf_path=pdf_path,
            index_dir=index_dir,
            embedding_model=embedding_model,
            filter_irrelevant_pages=filter_irrelevant_pages,
            keep_appendices=keep_appendices,
        )
        get_loaded_index.clear()

    loaded_index = None
    if index_dir.exists():
        try:
            loaded_index = get_loaded_index(str(index_dir.resolve()))
        except Exception:
            loaded_index = None

    with st.sidebar:
        st.divider()
        st.subheader("Index Summary")
        if loaded_index is not None:
            _show_index_summary(loaded_index)
        else:
            st.caption("Build the index to see summary data.")

    if "messages" not in st.session_state:
        st.session_state.messages = _default_messages()

    if loaded_index is not None:
        _show_rebuild_notice(loaded_index)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about the books")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            index = loaded_index or get_loaded_index(str(index_dir.resolve()))
            history = _model_history(st.session_state.messages[:-1], max_messages=8)
            answer, hits = answer_with_rag(
                question=question,
                index=index,
                chat_model=chat_model,
                top_k=top_k,
                chat_history=history,
            )
            st.markdown(answer)
            _render_sources(hits)
        except Exception as exc:
            st.error(
                "Could not answer the question. Make sure Ollama is running and the index exists."
            )
            st.exception(exc)
            answer = "I couldn't answer due to a runtime issue."

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
