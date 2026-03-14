# Game of Thrones RAG Chatbot (Local + Ollama)

This project builds a local chatbot over the combined *A Song of Ice and Fire* PDF using:

- Ollama for embeddings and generation
- Local vector retrieval with NumPy + BM25 hybrid scoring
- Streamlit for the chat UI

## What improved

- Front matter filtering removes cover pages, title pages, copyright pages, contents pages, map links, dedications, acknowledgments, and author pages.
- Cross-page chunking builds cleaner context windows instead of hard-stopping every chunk at a single page.
- Retrieval is stronger for follow-up questions, character identity questions, and fate/death queries.
- Index metadata now records how many pages were kept vs skipped.
- The UI shows index stats, rebuild controls, and clearer source labels with book + page range.

## 1) Prerequisites

Install and run Ollama, then pull models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Build the index

Default rebuild:

```bash
python ingest.py
```

Useful options:

```bash
python ingest.py --pdf "A Game of Thrones 5-Book Bundle ... .pdf"
python ingest.py --disable-page-filter
python ingest.py --drop-appendices
```

The default build:

- filters irrelevant pages
- keeps appendices
- uses larger cross-page chunks for better answers

## 4) Run the chatbot UI

```bash
streamlit run app.py
```

In the sidebar you can:

- rebuild the index
- toggle page filtering
- choose whether appendices stay in the corpus
- clear chat history
- inspect index stats

## Optional: terminal chatbot

```bash
python main.py
```

## Files

- `rag_engine.py`: ingestion, filtering, chunking, retrieval, prompting
- `ingest.py`: CLI for building the index
- `app.py`: Streamlit chatbot UI
- `main.py`: terminal chatbot
- `data/got_index/`: generated index files (`vectors.npy`, `chunks.json`, `metadata.json`)
