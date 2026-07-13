"""
Multimodal RAG over "Attention Is All You Need" (Vaswani et al., 2017)

Pipeline:
  1. Download/locate the paper PDF.
  2. Extract body text + headings via PyMuPDF text layout (font-size heuristic).
  3. Rasterize every page and send it through Gemini's vision endpoint to get
     rich descriptions of any figure / diagram / chart / table on that page
     (vector-drawn diagrams, like the encoder-decoder architecture, don't show
     up as embedded raster images, so page rasterization + vision is the
     reliable way to capture them).
  4. Chunk + embed everything (text, headings, table/figure/chart
     descriptions) into ONE shared Chroma vector store using ONE embedding
     model, so retrieval can pull relevant content regardless of modality.
  5. Hybrid retrieval (BM25 + dense) + Gemini grounded generation.
  6. Demo queries that each hit a different modality (table / figure / text),
     with every query + retrieved context + answer logged to disk.

Requires: GOOGLE_API_KEY set in the environment (or a .env file).
"""

import os
import io
import json
import time

from dotenv import load_dotenv
import requests
import fitz  # PyMuPDF

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Add it to your environment or a .env file "
        "(never hardcode it in source)."
    )
client = genai.Client(api_key=API_KEY)

# NOTE: confirm these model strings against Google's current model list —
# whatever you use for VISION_MODEL must accept image input (multimodal).
TEXT_MODEL = "gemini-3.1-flash-lite"
VISION_MODEL = "gemini-3.1-flash-lite"

PDF_DIR = "docs"
PDF_PATH = os.path.join(PDF_DIR, "attention_is_all_you_need.pdf")
PDF_URL = "https://arxiv.org/pdf/1706.03762"

PERSIST_DIR = "chroma_multimodal_db"
LOG_PATH = "rag_session_log.jsonl"

RENDER_DPI = 150            # page rasterization resolution for vision calls
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
HEADING_FONT_THRESHOLD = 11.5  # tune against your PDF; body text is usually ~10-10.5pt

TOKEN_BUDGET = 100_000
usage_totals = {"prompt": 0, "output": 0, "total": 0}
MIN_INTERVAL_SECONDS = 1.0
_last_call_time = 0.0


def _throttle():
    """Enforce a minimum gap between API calls (helps with free-tier RPM limits)."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_INTERVAL_SECONDS:
        time.sleep(MIN_INTERVAL_SECONDS - elapsed)
    _last_call_time = time.time()


def _track_usage(response):
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return
    usage_totals["prompt"] += usage.prompt_token_count or 0
    usage_totals["output"] += usage.candidates_token_count or 0
    usage_totals["total"] += usage.total_token_count or 0
    print(
        f"[tokens] this call: prompt={usage.prompt_token_count} "
        f"output={usage.candidates_token_count} | running total={usage_totals['total']}"
    )
    if usage_totals["total"] >= TOKEN_BUDGET:
        print(f"\n!!! Token budget of {TOKEN_BUDGET} reached. Stopping further calls. !!!")
        raise SystemExit(0)


# ---------------------------------------------------------------------------
# 1. Get the source PDF
# ---------------------------------------------------------------------------
def ensure_pdf():
    os.makedirs(PDF_DIR, exist_ok=True)
    if not os.path.exists(PDF_PATH):
        print(f"Downloading 'Attention Is All You Need' to {PDF_PATH} ...")
        r = requests.get(PDF_URL, timeout=60)
        r.raise_for_status()
        with open(PDF_PATH, "wb") as f:
            f.write(r.content)
    return PDF_PATH


# ---------------------------------------------------------------------------
# 2. Text + heading extraction (PyMuPDF, font-size heuristic)
# ---------------------------------------------------------------------------
def extract_text_and_headings(fitz_doc):
    text_docs, heading_docs = [], []

    for page_index, page in enumerate(fitz_doc, start=1):
        page_dict = page.get_text("dict")
        body_lines = []

        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = "".join(s["text"] for s in spans).strip()
                if not line_text:
                    continue

                max_size = max(s["size"] for s in spans)
                is_bold = any("Bold" in s.get("font", "") for s in spans)

                if max_size >= HEADING_FONT_THRESHOLD and (is_bold or max_size > HEADING_FONT_THRESHOLD + 1):
                    heading_docs.append(
                        Document(
                            page_content=line_text,
                            metadata={"page": page_index, "content_type": "heading", "source": "attention_paper"},
                        )
                    )
                else:
                    body_lines.append(line_text)

        if body_lines:
            text_docs.append(
                Document(
                    page_content="\n".join(body_lines),
                    metadata={"page": page_index, "content_type": "text", "source": "attention_paper"},
                )
            )

    return text_docs, heading_docs


# ---------------------------------------------------------------------------
# 3. Visual (figure / diagram / chart / table) extraction via Gemini vision
# ---------------------------------------------------------------------------
VISION_PROMPT = (
    "You are looking at one page from the paper 'Attention Is All You Need'. "
    "If this page contains a figure, architecture diagram, or chart (e.g. attention-"
    "weight plots), describe it in detail: what it depicts, its structure/components, "
    "axis labels, and any visible trends or values — enough detail that someone could "
    "answer questions about it without seeing the image. "
    "If this page contains a data table, transcribe it as a markdown table, preserving "
    "every row and column value exactly. "
    "A page can contain more than one of these; describe each separately. "
    "If the page has none of the above (pure body text, references, etc.), respond with "
    "exactly: NONE"
)


def render_page_png(page, dpi=RENDER_DPI):
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def classify_visual_description(description):
    lower = description.lower()
    if "| ---" in description or "|--" in description or lower.startswith("table"):
        return "table"
    if any(k in lower for k in ("chart", "plot", "graph")):
        return "chart"
    return "figure"


def describe_visuals_on_page(image_bytes):
    _throttle()
    response = client.models.generate_content(
        model=VISION_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            VISION_PROMPT,
        ],
        config=types.GenerateContentConfig(max_output_tokens=700, temperature=0.1),
    )
    _track_usage(response)
    return response.text.strip()


def extract_visuals(fitz_doc):
    visual_docs = []
    for page_index, page in enumerate(fitz_doc, start=1):
        image_bytes = render_page_png(page)
        try:
            description = describe_visuals_on_page(image_bytes)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [warn] vision call failed on page {page_index}: {e}")
            continue

        if description and description.strip().upper() != "NONE":
            content_type = classify_visual_description(description)
            visual_docs.append(
                Document(
                    page_content=description,
                    metadata={"page": page_index, "content_type": content_type, "source": "attention_paper"},
                )
            )
            print(f"  page {page_index}: found {content_type}")

    return visual_docs


# ---------------------------------------------------------------------------
# 4. Build the shared document set + vector store
# ---------------------------------------------------------------------------
def build_documents(pdf_path):
    fitz_doc = fitz.open(pdf_path)
    try:
        print("Extracting body text and headings ...")
        text_docs, heading_docs = extract_text_and_headings(fitz_doc)

        print("Extracting figures/diagrams/charts/tables via Gemini vision ...")
        visual_docs = extract_visuals(fitz_doc)
    finally:
        fitz_doc.close()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    text_chunks = splitter.split_documents(text_docs)

    all_docs = text_chunks + heading_docs + visual_docs
    print(
        f"Built {len(text_chunks)} text chunks, {len(heading_docs)} headings, "
        f"{len(visual_docs)} figure/chart/table descriptions -> {len(all_docs)} total documents"
    )
    return all_docs


def initialize_rag_system():
    print("--- Initializing multimodal index (this only happens once) ---")
    pdf_path = ensure_pdf()
    all_docs = build_documents(pdf_path)

    # Same embedding model for every content type -> one shared embedding space.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(all_docs, embeddings, persist_directory=PERSIST_DIR)

    dense_ret = db.as_retriever(search_kwargs={"k": 4})
    sparse_ret = BM25Retriever.from_documents(all_docs)
    sparse_ret.k = 4

    ensemble_retriever = EnsembleRetriever(retrievers=[sparse_ret, dense_ret], weights=[0.3, 0.7])
    return ensemble_retriever


# ---------------------------------------------------------------------------
# 5. Retrieval + grounded generation
# ---------------------------------------------------------------------------
def get_answer(ensemble_retriever, query):
    docs = ensemble_retriever.invoke(query)

    context_parts = []
    for d in docs:
        tag = f"[{d.metadata.get('content_type', 'text').upper()} - page {d.metadata.get('page', '?')}]"
        context_parts.append(f"{tag}\n{d.page_content}")
    context_text = "\n\n".join(context_parts)

    prompt = f"""Use the provided context to answer the query accurately.
The context may include body text, headings, transcribed tables, and descriptions
of figures/diagrams/charts. If the answer is not in the context, state that you do not know.

Context:
{context_text}

Query: {query}"""

    _throttle()
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=512, temperature=0.2),
    )
    _track_usage(response)

    answer = response.text

    # Persist query + retrieved context + answer so the end-to-end run is inspectable.
    record = {
        "query": query,
        "retrieved_context": [
            {
                "content_type": d.metadata.get("content_type"),
                "page": d.metadata.get("page"),
                "text": d.page_content,
            }
            for d in docs
        ],
        "answer": answer,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return docs, answer


# ---------------------------------------------------------------------------
# 6. Demo queries across modalities
# ---------------------------------------------------------------------------
DEMO_QUERIES = [
    # Table
    "According to the results table, what BLEU score did the base Transformer "
    "model achieve on the WMT 2014 English-to-German translation task?",
    # Figure
    "Describe the overall encoder-decoder architecture of the Transformer as "
    "shown in the paper's architecture diagram.",
    # Text
    "What is multi-head attention and why does the paper use it instead of a "
    "single attention function?",
]


def run_demo(ensemble_retriever):
    print("\n=== Running demo queries across modalities (table / figure / text) ===")
    for q in DEMO_QUERIES:
        print(f"\n> Query: {q}")
        try:
            docs, answer = get_answer(ensemble_retriever, q)
        except SystemExit:
            break
        sources = [(d.metadata.get("content_type"), d.metadata.get("page")) for d in docs]
        print(f"--- Retrieved from (content_type, page): {sources} ---")
        print(f"--- Answer ---\n{answer}")


# ---------------------------------------------------------------------------
# Execution flow
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    retriever = initialize_rag_system()

    try:
        run_demo(retriever)
    except SystemExit:
        pass

    print(f"\n=== System Ready (Type 'exit' to quit) — session log: {LOG_PATH} ===")
    while True:
        try:
            q = input("\n> Query: ")
        except EOFError:
            break
        if q.lower() == "exit":
            break

        try:
            context, answer = get_answer(retriever, q)
            print(f"\n--- Answer ---\n{answer}")
        except SystemExit:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)

    print(
        f"\n=== Session totals: prompt={usage_totals['prompt']} "
        f"output={usage_totals['output']} total={usage_totals['total']} ==="
    )