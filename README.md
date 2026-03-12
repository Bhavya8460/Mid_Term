# Game of Thrones RAG Chatbot (Local + Ollama)

This project builds a chatbot over your combined *Game of Thrones* books PDF using:

- Ollama for embeddings and generation
- Local vector retrieval (NumPy cosine similarity)
- Streamlit UI for chat

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

## 3) Build the RAG index from your PDF

If your PDF is in the project folder, this is enough:

```bash
python ingest.py
```

Or provide a path explicitly:

```bash
python ingest.py --pdf "A Game of Thrones 5-Book Bundle ... .pdf"
```

## 4) Run the chatbot UI

```bash
streamlit run app.py
```

In the sidebar, you can rebuild the index, change models, and adjust retrieval depth.

If answers look unrelated (for example only appendix headings), rebuild the index after updating code:

```bash
python ingest.py --min-chunk-chars 80
```

For better answer quality on a large corpus, set a higher retrieval depth in the UI (for example `Top K chunks = 8` to `12`).

The chatbot keeps short conversation memory (recent turns) so follow-up questions feel more natural.

## Optional: terminal chatbot

```bash
python main.py
```

## Files

- `ingest.py`: PDF extraction, chunking, embeddings, and index build
- `app.py`: Streamlit chat application
- `main.py`: terminal chat alternative
- `rag_engine.py`: shared RAG logic
- `data/got_index/`: generated index files (`vectors.npy`, `chunks.json`, `metadata.json`)
