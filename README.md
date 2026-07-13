# Multimodal RAG — "Attention Is All You Need"

A retrieval-augmented generation (RAG) pipeline that answers questions about the
Transformer paper (Vaswani et al., 2017) using **text, headings, tables, and
figures/diagrams together** — not just the text layer of the PDF.

Most RAG pipelines over PDFs silently drop anything that isn't plain text:
architecture diagrams, attention-weight plots, and result tables are usually
invisible to a text extractor. This project routes every page through
Gemini's vision capability to capture that content as text descriptions,
then embeds it alongside the body text in one shared vector store — so a
question like *"what BLEU score did the base model get?"* can be answered
from a table, and *"describe the encoder-decoder architecture"* can be
answered from a diagram, with the same retrieval pipeline.

## Features

- **Automatic source retrieval** — downloads the paper from arXiv if it isn't already present locally.
- **Structured text extraction** — separates body text from headings using font-size/weight heuristics (PyMuPDF).
- **Vision-based figure/table extraction** — rasterizes every page and asks Gemini to describe diagrams/charts or transcribe tables into markdown, since vector-drawn figures don't exist as extractable embedded images.
- **One shared embedding space** — text chunks, headings, and vision-derived descriptions are all embedded with the same model into a single Chroma collection.
- **Hybrid retrieval** — BM25 (sparse) + dense vector search, combined via a weighted ensemble.
- **Grounded generation** — answers are generated strictly from retrieved context, with Gemini instructed to say when it doesn't know.
- **Cross-modality demo** — three built-in queries that each pull from a different content type (table, figure, text).
- **Full traceability** — every query, its retrieved context, and the final answer are logged to `rag_session_log.jsonl`.
- **Cost controls** — request throttling and a configurable token budget that halts execution before you blow through a quota.

## Architecture

```
                    ┌─────────────────────┐
                    │   Attention.pdf      │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
        PyMuPDF text extraction     Page rasterization
         (body text / headings)      + Gemini vision
                 │                  (tables / figures / charts)
                 └─────────────┬─────────────┘
                               │
                    Shared embedding model
                     (sentence-transformers)
                               │
                        Chroma vector store
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
          BM25 retriever              Dense retriever
                 └─────────────┬─────────────┘
                        Ensemble retriever
                               │
                      Gemini grounded answer
```

## Requirements

- Python 3.10+
- A Google API key with access to the Gemini API

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your-api-key-here
```

The key is loaded from the environment at runtime — it is never hardcoded.

## Usage

```bash
python multimodal_rag.py
```

On first run, the script will:

1. Download the paper (if not already present in `docs/`).
2. Extract and vision-describe all content into a Chroma index.
3. Run three demo queries — one each against a table, a figure, and body text.
4. Drop into an interactive prompt where you can ask your own questions.

Type `exit` to quit. A running token-usage total is printed after each call
and logged at session end.

## Output

Each query produces:

- Console output with the retrieved chunks' content type/page and the final answer.
- An appended record in `rag_session_log.jsonl`:

```json
{
  "query": "...",
  "retrieved_context": [
    {"content_type": "table", "page": 8, "text": "..."}
  ],
  "answer": "..."
}
```

## Configuration knobs

| Variable | Purpose |
|---|---|
| `TOKEN_BUDGET` | Hard ceiling on cumulative tokens before the script exits |
| `RENDER_DPI` | Resolution used when rasterizing pages for vision calls |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Text splitter settings |
| `HEADING_FONT_THRESHOLD` | Font-size cutoff used to classify a line as a heading |

## Project structure

```
.
├── multimodal_rag.py       # main pipeline
├── requirements.txt
├── docs/                   # downloaded source PDF
├── chroma_multimodal_db/   # persisted vector store
└── rag_session_log.jsonl   # query/context/answer log
```

## Notes & limitations

- Vision-based figure/table detection relies on prompted heuristics rather
  than a dedicated layout model — verify `HEADING_FONT_THRESHOLD` and the
  vision prompt against your own PDF if reusing this on a different paper.
- Confirm the Gemini model strings used for text and vision generation are
  current and support the required input modality before running.
