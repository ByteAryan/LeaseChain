"""
DocuMind — A Context-Aware Semantic Search Engine
==================================================
Upload a PDF or TXT file, split it into overlapping semantic chunks, embed every
chunk with a local open-source transformer, index the vectors with FAISS, and
search the document by *meaning* instead of by exact keywords.

Pipeline (good for explaining during an evaluation)
---------------------------------------------------
1. EXTRACTION   : PDF/TXT  -> raw text (PyPDF2).
2. CHUNKING     : raw text -> overlapping, sentence-aware chunks (~150 words).
3. EMBEDDING    : chunk    -> 384-dim dense vector (all-MiniLM-L6-v2).
4. INDEXING     : vectors  -> FAISS IndexFlatL2 (exact nearest-neighbour search).
5. RETRIEVAL    : query    -> same embedding model -> top-K nearest chunks.
6. SCORING      : FAISS L2 distance -> cosine similarity -> confidence (%).
7. VISUALISING  : PCA flattens the 384-dim vectors to 2-D for the "meaning map".

Run with:  streamlit run app.py
"""

# --------------------------------------------------------------------------- #
# Environment tweaks (must happen BEFORE importing faiss / torch / transformers)
# --------------------------------------------------------------------------- #
import os

# Prevents a common "OMP: Error #15" crash on macOS/Windows when FAISS and
# PyTorch each ship their own OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Silences HuggingFace tokenizer fork/parallelism warnings in Streamlit reruns.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
import html
import io
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import faiss
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 maps every text to a 384-dimensional vector
MAX_HISTORY_ITEMS = 10  # how many past queries are displayed in the sidebar
MAX_MAP_POINTS = 1500  # background dots drawn on the meaning map (keeps the SVG light)

# Confidence bands (cosine similarity expressed as %). MiniLM scores for
# genuinely relevant passages typically fall between 35% and 70%.
STRONG_THRESHOLD = 55.0
MODERATE_THRESHOLD = 35.0


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Unit:
    """Smallest piece of text we chunk with (normally one sentence)."""

    text: str
    page: int
    words: int


@dataclass
class Chunk:
    """A chunk is the unit that gets embedded, indexed and returned in search."""

    chunk_id: int
    text: str
    page: int
    words: int


# --------------------------------------------------------------------------- #
# Step 1 — Text extraction
# --------------------------------------------------------------------------- #
def _normalize_whitespace(text: str) -> str:
    """Collapse newlines/tabs/multiple spaces so downstream splitting is clean."""
    return re.sub(r"\s+", " ", text).strip()


def extract_pages(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """
    Convert an uploaded file into a list of (page_number, text) tuples.
    TXT files are treated as a single "page".
    """
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))

        # Many PDFs are "encrypted" with an empty user password; try to unlock.
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise ValueError("This PDF is password-protected.") from exc

        pages: List[Tuple[int, str]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""  # extract_text() can return None
            cleaned = _normalize_whitespace(raw)
            if cleaned:
                pages.append((page_number, cleaned))
    else:
        try:
            raw = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw = file_bytes.decode("latin-1", errors="ignore")
        cleaned = _normalize_whitespace(raw)
        pages = [(1, cleaned)] if cleaned else []

    if not pages:
        raise ValueError(
            "No extractable text found. If this is a scanned PDF, it needs OCR first."
        )
    return pages


# --------------------------------------------------------------------------- #
# Step 2 — Chunking
# --------------------------------------------------------------------------- #
# WHY CHUNK?  Embedding models have a limited context window (256 word-pieces
# for MiniLM) and a single vector for a whole book would blur every topic
# together. Splitting into focused passages gives each vector ONE clear meaning,
# which makes retrieval precise.
#
# WHY OVERLAP?  If an idea straddles a chunk boundary, overlap guarantees that
# at least one chunk contains the full context.
#
# WHY SENTENCE-AWARE?  Cutting mid-sentence produces fragments with distorted
# meaning; we only cut between sentences.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def to_units(pages: List[Tuple[int, str]], max_words: int) -> List[Unit]:
    """Split page text into sentence units; hard-split any sentence that is too long."""
    units: List[Unit] = []
    for page_number, text in pages:
        for sentence in _SENTENCE_SPLIT.split(text):
            words = sentence.split()
            if not words:
                continue
            if len(words) <= max_words:
                units.append(Unit(" ".join(words), page_number, len(words)))
            else:
                # Pathological case (e.g. a table or a run-on line with no punctuation).
                for start in range(0, len(words), max_words):
                    piece = words[start : start + max_words]
                    units.append(Unit(" ".join(piece), page_number, len(piece)))
    return units


def build_chunks(units: List[Unit], chunk_size: int, overlap: int) -> List[Chunk]:
    """
    Greedily pack sentences into chunks of ~`chunk_size` words. When a chunk is
    full, the trailing sentences (up to `overlap` words) are carried over into
    the next chunk -- a "sliding window" with sentence-level granularity.
    """
    chunks: List[Chunk] = []
    current: List[Unit] = []
    current_words = 0

    def emit(buffer: List[Unit]) -> None:
        text = " ".join(u.text for u in buffer)
        chunks.append(
            Chunk(
                chunk_id=len(chunks),
                text=text,
                page=buffer[0].page,  # page where the chunk begins
                words=len(text.split()),
            )
        )

    for unit in units:
        if current and current_words + unit.words > chunk_size:
            emit(current)
            # Build the overlap window from the end of the chunk we just emitted.
            carry: List[Unit] = []
            carried_words = 0
            for previous in reversed(current):
                if carried_words + previous.words > overlap:
                    break
                carry.insert(0, previous)
                carried_words += previous.words
            current, current_words = carry, carried_words

        current.append(unit)
        current_words += unit.words

    if current:
        emit(current)
    return chunks


# --------------------------------------------------------------------------- #
# Step 3 — Embeddings
# --------------------------------------------------------------------------- #
# An EMBEDDING is a list of numbers (here 384) that encodes the *meaning* of a
# text. Texts with similar meaning land close together in this vector space, so
# "automobile maintenance" and "fixing a car" end up as near neighbours even
# though they share no keywords.
@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    """Load the transformer once per server process (cached across reruns)."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Encode texts into L2-normalised float32 vectors (unit length).

    Normalising is the key trick: for unit vectors, Euclidean (L2) distance and
    cosine similarity are mathematically linked, so we can recover a cosine
    score from FAISS's L2 distance (see `l2_to_similarity`).
    """
    vectors = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(vectors.astype("float32"))  # FAISS needs float32


# --------------------------------------------------------------------------- #
# Step 4 — Vector indexing (FAISS)
# --------------------------------------------------------------------------- #
# FAISS (Facebook AI Similarity Search) stores vectors and finds nearest
# neighbours very quickly. IndexFlatL2 performs an *exact* brute-force search
# using squared Euclidean distance -- perfect accuracy, and still milliseconds
# for tens of thousands of chunks. (For millions of vectors you would switch to
# an approximate index such as IVF or HNSW.)
def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# --------------------------------------------------------------------------- #
# Step 5/6 — Retrieval and score conversion
# --------------------------------------------------------------------------- #
def l2_to_similarity(squared_l2: float) -> float:
    """
    Convert a FAISS distance into a cosine similarity in [0, 1].

    FAISS returns the SQUARED L2 distance d = ||a - b||^2. For unit vectors:
        ||a - b||^2 = 2 - 2 * cos(a, b)   =>   cos(a, b) = 1 - d / 2
    Identical meaning -> d = 0 -> similarity 1.0. Unrelated text -> d ~ 2 -> ~0.
    """
    return float(np.clip(1.0 - squared_l2 / 2.0, 0.0, 1.0))


def semantic_search(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    chunks: List[Chunk],
    top_k: int,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Embed the query with the SAME model and fetch its nearest chunks.
    Returns (ranked results, query vector). The vector is reused by the map.
    """
    query_vector = embed_texts(model, [query])
    k = min(top_k, index.ntotal)
    distances, ids = index.search(query_vector, k)

    results: List[Dict] = []
    for rank, (dist, idx) in enumerate(zip(distances[0], ids[0]), start=1):
        if idx < 0:  # FAISS pads with -1 when fewer than k vectors exist
            continue
        similarity = l2_to_similarity(float(dist))
        results.append(
            {
                "rank": rank,
                "chunk": chunks[int(idx)],
                "chunk_index": int(idx),
                "distance": float(dist),
                "confidence": similarity * 100.0,
            }
        )
    return results, query_vector


# --------------------------------------------------------------------------- #
# Step 7 — Flattening the vector space for the "meaning map"
# --------------------------------------------------------------------------- #
# PCA (Principal Component Analysis) finds the two directions along which the
# 384-dim vectors vary the most and projects every vector onto them. The result
# is a 2-D picture where passages with similar meaning tend to sit together.
# It is an approximation (384 -> 2 dimensions loses information), so the map is
# for intuition; the confidence scores always come from the full vectors.
def compute_projection(embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    basis = np.zeros((2, embeddings.shape[1]), dtype="float32")
    if len(embeddings) >= 2:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)  # rows = principal axes
        basis[: min(2, vt.shape[0])] = vt[:2]
    return {"mean": mean, "basis": basis, "coords": centered @ basis.T}


def normalize_for_map(
    doc: Dict, query_vector: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Scale passage (and optional query) coordinates into the unit square."""
    points = doc["coords"]
    query_xy = None
    if query_vector is not None:
        query_xy = ((query_vector[0] - doc["mean"]) @ doc["basis"].T).reshape(1, 2)
    everything = points if query_xy is None else np.vstack([points, query_xy])

    low, high = everything.min(axis=0), everything.max(axis=0)
    span = float(max((high - low).max(), 1e-9))
    offset = (1.0 - (high - low) / span) / 2.0  # centre the smaller axis

    def scale(p: np.ndarray) -> np.ndarray:
        return (p - low) / span + offset

    return scale(points), (scale(query_xy)[0] if query_xy is not None else None)


# --------------------------------------------------------------------------- #
# Sample document (lets anyone try the app in one click, no file needed)
# --------------------------------------------------------------------------- #
SAMPLE_NAME = "sample-guide.txt"
SAMPLE_QUERIES = [
    "Why do I forget things when I'm tired?",
    "Affordable ways to run a house on renewable power",
    "Growing food with very little space",
]
SAMPLE_TEXT = (
    "While we sleep, the brain replays the day and decides which experiences to keep. "
    "Deep sleep strengthens facts and names, while dreaming sleep links new ideas to old ones. "
    "People who cut their nights short often struggle to recall details the next morning, because the brain "
    "never finished filing them away. Short naps help a little, but a regular bedtime does far more. "
    "Dim lights, a cool room and a quiet phone in the hour before bed all make the process work better. "
    "Students who rest properly before an exam usually beat classmates who stay up cramming, even if they studied fewer hours.\n\n"
    "Rooftop solar panels turn sunlight into electricity that a household can use directly or sell back to the grid. "
    "A typical system pays for itself within six to ten years, and many governments offer rebates that shorten the wait. "
    "Adding a home battery stores the afternoon surplus for the evening, when families use the most power. "
    "Panels need very little care beyond an occasional rinse, and most come with warranties that last twenty-five years. "
    "Homes with good insulation and efficient appliances need fewer panels, which lowers the upfront cost.\n\n"
    "Great coffee starts with fresh beans ground just before brewing. "
    "A grind that is too fine makes the cup bitter, while one that is too coarse tastes thin and sour. "
    "Water just below boiling extracts the flavour best, and a ratio of about sixty grams of coffee per litre is a reliable starting point. "
    "Pour-over methods reward patience, since a slow, steady stream wets all the grounds evenly. "
    "Rinsing the paper filter and warming the mug beforehand keeps the temperature stable and the taste clean.\n\n"
    "You do not need a garden to grow your own food. Herbs such as basil, mint and chives thrive on a sunny windowsill. "
    "Balconies can hold tomatoes, peppers and salad leaves in deep containers with good drainage. "
    "Most vegetables want about six hours of light, so watch where the sun falls before choosing a spot. "
    "Water when the top of the soil feels dry rather than on a fixed schedule, because overwatering kills more pot plants than neglect. "
    "A small bin of kitchen scraps and worms can supply free fertiliser all year.\n\n"
    "A budget is simply a plan for your money that you make before the month begins. "
    "Start by writing down what comes in and what goes out for a few weeks, because most people underestimate small daily purchases. "
    "A popular rule splits income into needs, wants and savings, with roughly half, thirty percent and twenty percent. "
    "Set up an automatic transfer on payday so savings happen before you can spend the money. "
    "Aim to build an emergency fund that covers three months of essentials, and cancel subscriptions you no longer use.\n\n"
    "Space telescopes see the universe without the blur of Earth's atmosphere. "
    "They have revealed thousands of planets orbiting other stars, some of them in the zone where liquid water could exist. "
    "Robotic rovers on Mars drill into rock to search for signs that microbes once lived there. "
    "Each mission costs billions and takes years to plan, yet the technology developed along the way often finds uses in medicine and communication. "
    "Sending people to other worlds remains the hardest challenge, because of radiation, isolation and the long journey.\n\n"
    "Training for a first marathon works best when mileage grows slowly. "
    "Most plans add no more than ten percent to the weekly distance, with an easier week every fourth week to recover. "
    "The long run on the weekend teaches your body to burn fuel efficiently and your mind to stay patient. "
    "Rest days are when muscles actually rebuild, so skipping them leads to injury rather than speed. "
    "Practise drinking and eating during long runs so that nothing about race day is a surprise.\n\n"
    "Machine learning lets computers find patterns in examples instead of following hand-written rules. "
    "A model is shown thousands of labelled photos, for instance, and gradually adjusts its internal numbers until it can name new photos correctly. "
    "If it memorises the training examples instead of learning the general pattern, it fails on fresh data, a problem called overfitting. "
    "Keeping some data aside for testing reveals whether the model truly learned. "
    "More varied examples usually help more than a more complicated model."
)


# --------------------------------------------------------------------------- #
# Document processing pipeline
# --------------------------------------------------------------------------- #
def pick_suggestions(chunks: List[Chunk], embeddings: np.ndarray, n: int = 3) -> List[str]:
    """
    Offer a few clickable starter queries for any document.
    Farthest-point sampling picks passages that are far apart in meaning (so the
    suggestions cover different topics); each passage's opening words become a query.
    """
    count = len(chunks)
    chosen = [0]
    if count > 1:
        distance = np.linalg.norm(embeddings - embeddings[0], axis=1)
        while len(chosen) < min(n, count):
            nxt = int(np.argmax(distance))
            chosen.append(nxt)
            distance = np.minimum(distance, np.linalg.norm(embeddings - embeddings[nxt], axis=1))
    suggestions = []
    for idx in chosen:
        first_sentence = _SENTENCE_SPLIT.split(chunks[idx].text)[0]
        words = first_sentence.split()[:10]
        phrase = " ".join(words).strip(" .,;:!?-—")
        if phrase and phrase not in suggestions:
            suggestions.append(phrase)
    return suggestions


def process_document(
    file_bytes: bytes,
    filename: str,
    chunk_size: int,
    overlap: int,
    progress: Optional[Callable[[str], None]] = None,
    suggestions: Optional[List[str]] = None,
) -> Dict:
    """Run extraction -> chunking -> embedding -> indexing and collect analytics."""
    started = time.perf_counter()
    say = progress or (lambda _msg: None)

    say("Reading your file…")
    pages = extract_pages(file_bytes, filename)
    total_words = sum(len(text.split()) for _, text in pages)

    say("Splitting the text into overlapping passages…")
    units = to_units(pages, max_words=chunk_size)
    chunks = build_chunks(units, chunk_size, overlap)
    if not chunks:
        raise ValueError("The document produced no text chunks.")

    say(f"Turning {len(chunks):,} passages into vectors (the first run also downloads the model)…")
    model = load_model()
    embeddings = embed_texts(model, [c.text for c in chunks])

    say("Building the search index and the meaning map…")
    index = build_index(embeddings)
    projection = compute_projection(embeddings)

    word_counts = [c.words for c in chunks]
    return {
        "filename": filename,
        "is_pdf": filename.lower().endswith(".pdf"),
        "pages": len(pages),
        "total_words": total_words,
        "chunks": chunks,
        "index": index,
        "word_counts": word_counts,
        "avg_chunk": float(np.mean(word_counts)),
        "min_chunk": int(np.min(word_counts)),
        "max_chunk": int(np.max(word_counts)),
        "build_seconds": time.perf_counter() - started,
        "suggestions": suggestions or pick_suggestions(chunks, embeddings),
        **projection,
    }


# --------------------------------------------------------------------------- #
# Design system (dark)
# --------------------------------------------------------------------------- #
# Palette: midnight (page), slate (surfaces), ink (text), periwinkle (actions)
# and three bright signal colours that encode match strength (teal / amber / rose).
# Type: Bricolage Grotesque for headings and numbers, Public Sans for reading.
# The meaning map is the one expressive element; everything else stays quiet.
PANEL = "#161C2D"  # surface colour, reused inside SVG where CSS variables don't apply

STYLE_BASE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Public+Sans:wght@400;500;600&display=swap');

:root{
  color-scheme:dark;
  --bg:#0E1320; --paper:#161C2D; --paper2:#1C2338; --ink:#E7EBF5; --ink2:#9CA7C4; --line:#2A3250;
  --blue:#8E9BFF; --blue-strong:#A9B3FF; --blue-soft:rgba(142,155,255,.15);
  --teal:#2DBFAB; --teal-ink:#5EEAD4; --amber:#F5B942; --amber-ink:#FBCB6B;
  --rose:#F0647F; --rose-ink:#FF8FA5;
  --display:'Bricolage Grotesque','Public Sans',system-ui,sans-serif;
  --body:'Public Sans',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
}

/* ---- Streamlit chrome ---- */
.stApp{background:var(--bg);color:var(--ink);font-family:var(--body);isolation:isolate}
#MainMenu,footer{visibility:hidden}
[data-testid="stAppDeployButton"],[data-testid="stDeployButton"],.stDeployButton{display:none!important}
[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"] *,[data-testid="stSidebarCollapseButton"] *,[data-testid="stExpandSidebarButton"] *{color:var(--ink2)!important}
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1200px}
[data-testid="stSidebar"]{background:#111729;border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container{padding-top:1.6rem}
h1,h2,h3,h4,h5,h6{font-family:var(--display)!important;color:var(--ink)!important;letter-spacing:-.01em}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
label,[data-testid="stWidgetLabel"] p{color:var(--ink);font-family:var(--body)}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:var(--ink2)!important}
a{color:var(--blue)}
:focus-visible{outline:2px solid var(--blue)!important;outline-offset:2px}
::selection{background:rgba(142,155,255,.35)}
*{scrollbar-color:#3A4466 transparent}

/* ---- Inputs ---- */
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextInput"] [data-baseweb="base-input"]{background:var(--paper)!important;border-radius:14px!important}
[data-testid="stTextInput"] [data-baseweb="input"]{
  border:1.5px solid var(--line)!important;box-shadow:0 14px 30px -18px rgba(0,0,0,.7)}
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within{
  border-color:var(--blue)!important;box-shadow:0 0 0 4px rgba(142,155,255,.2)}
[data-testid="stTextInput"] input{
  font-family:var(--body);font-size:1.1rem;padding:1rem 1.1rem;color:var(--ink)!important;
  -webkit-text-fill-color:var(--ink)}
[data-testid="stTextInput"] input::placeholder{color:#7C87A6;-webkit-text-fill-color:#7C87A6}
[data-testid="stFileUploaderDropzone"]{background:var(--paper);border:1.5px dashed #46507A;border-radius:12px}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--blue);background:var(--paper2)}
[data-testid="stFileUploaderDropzone"] small,[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p{color:var(--ink2)!important}
[data-testid="stFileUploaderDropzone"] button{
  background:var(--blue-soft);color:var(--ink);border:1px solid var(--line);border-radius:8px}
[data-testid="stFileUploaderFile"] *{color:var(--ink)!important}
[data-testid="stSlider"] label p{font-weight:500}
[data-testid="stSliderThumbValue"]{color:var(--ink)!important}
[data-testid="stTickBarMin"],[data-testid="stTickBarMax"]{color:var(--ink2)!important}
[data-testid="stTooltipIcon"] svg{color:var(--ink2)}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:12px;background:var(--paper)}
[data-testid="stExpander"] summary p,[data-testid="stExpander"] summary span{color:var(--ink)!important;font-weight:500}
[data-testid="stExpander"] summary svg{color:var(--ink2)}
[data-testid="stSpinner"] *{color:var(--ink2)!important}
[data-baseweb="tab-list"]{gap:.5rem}
button[data-baseweb="tab"]{background:transparent!important}
button[data-baseweb="tab"] p{color:var(--ink2);font-weight:600;font-size:1rem}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--ink)}
[data-baseweb="tab-highlight"]{background:var(--blue)!important}
[data-baseweb="tab-border"]{background:var(--line)!important}

/* ---- Buttons (only actionable things get hover motion) ---- */
.stButton>button{
  width:100%;border:1px solid var(--line);background:var(--paper2);color:var(--ink);
  border-radius:10px;font-weight:500;transition:border-color .15s,background .15s,color .15s}
.stButton>button p{color:inherit}
.stButton>button:hover{border-color:var(--blue);background:var(--blue-soft);color:var(--blue-strong)}
[data-testid="stSidebar"] .stButton>button{justify-content:flex-start;text-align:left;padding:.45rem .8rem}
.stButton>button[kind="primary"],.stDownloadButton>button{
  background:var(--blue);color:#0E1320;border:1px solid var(--blue);border-radius:10px;font-weight:600}
.stButton>button[kind="primary"] p,.stDownloadButton>button p{color:#0E1320}
.stButton>button[kind="primary"]:hover,.stDownloadButton>button:hover{
  background:var(--blue-strong);border-color:var(--blue-strong);color:#0E1320}

/* ---- Header ---- */
.dm-head{display:flex;align-items:center;gap:14px;margin-bottom:1.6rem}
.dm-wordmark{font-family:var(--display);font-weight:800;font-size:2.5rem;letter-spacing:-.035em;line-height:1;color:var(--ink)}
.dm-tagline{color:var(--ink2);font-size:1.05rem;margin-top:.35rem}
.dm-brand{display:flex;align-items:center;gap:10px;margin-bottom:1.2rem}
.dm-brand b{font-family:var(--display);font-size:1.25rem;letter-spacing:-.02em;color:var(--ink)}
.dm-sec{font-family:var(--display);font-weight:700;font-size:1.4rem;letter-spacing:-.02em;margin:1.4rem 0 .7rem;color:var(--ink)}
.dm-sidesec{font-family:var(--display);font-weight:700;font-size:1.02rem;margin:1.2rem 0 .3rem;color:var(--ink)}
.dm-hint{color:var(--ink2);font-size:.9rem;line-height:1.5;margin:.2rem 0 .6rem}

/* ---- Banners ---- */
.dm-banner{display:flex;gap:.85rem;align-items:flex-start;padding:.9rem 1.15rem;border-radius:12px;
  border:1px solid;margin:.4rem 0 1.2rem;line-height:1.5}
.dm-banner b{display:block;font-weight:600}
.dm-ico{flex:none;width:22px;height:22px;border-radius:50%;display:grid;place-items:center;
  font-size:.8rem;font-weight:700;color:#0E1320;margin-top:1px}
.dm-success{background:rgba(45,191,171,.12);border-color:rgba(45,191,171,.45);color:#C4F5EC}.dm-success .dm-ico{background:var(--teal)}
.dm-info{background:rgba(142,155,255,.13);border-color:rgba(142,155,255,.45);color:#DDE1FF}.dm-info .dm-ico{background:var(--blue)}
.dm-warn{background:rgba(245,185,66,.12);border-color:rgba(245,185,66,.45);color:#FDE6B0}.dm-warn .dm-ico{background:var(--amber)}
.dm-error{background:rgba(240,100,127,.12);border-color:rgba(240,100,127,.45);color:#FFD0DA}.dm-error .dm-ico{background:var(--rose)}

/* ---- Panels and analytics ---- */
.dm-panel{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:1.2rem 1.4rem}
.dm-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:0}
.dm-stat{padding:0 1.3rem;border-left:1px solid var(--line)}
.dm-stat:first-child{padding-left:0;border-left:none}
.dm-stat b{display:block;font-family:var(--display);font-size:2.1rem;font-weight:700;letter-spacing:-.03em;line-height:1.1;color:var(--ink)}
.dm-stat span{color:var(--ink2);font-size:.9rem}
.dm-histtitle{font-weight:600;font-size:.98rem;margin:0 0 .2rem;color:var(--ink)}
.dm-hist{display:flex;align-items:flex-end;gap:8px;height:170px}
.dm-hcol{flex:1;height:100%;display:flex;flex-direction:column}
.dm-hplot{flex:1;display:flex;align-items:flex-end;padding-top:20px}
.dm-hbar{position:relative;width:100%;background:var(--blue);border-radius:5px 5px 0 0;min-height:3px}
.dm-hval{position:absolute;top:-19px;left:50%;transform:translateX(-50%);font-size:.78rem;font-weight:600;color:var(--ink2)}
.dm-hlab{height:20px;text-align:center;font-size:.78rem;color:var(--ink2);padding-top:5px}
.dm-note{color:var(--ink2);font-size:.92rem;margin-top:.9rem;line-height:1.6}
.dm-summary{color:var(--ink2);margin:0 0 .3rem;font-size:.98rem}
.dm-summary b{color:var(--ink);font-weight:600}

/* ---- Results ---- */
.dm-results{background:var(--paper);border:1px solid var(--line);border-radius:14px;overflow:hidden}
.dm-row{display:flex;gap:1rem;padding:1.15rem 1.3rem;border-bottom:1px solid var(--line);transition:background .15s}
.dm-row:last-child{border-bottom:none}
.dm-row:hover{background:var(--paper2)}
.dm-mark{flex:none;width:30px;height:30px;border-radius:50%;display:grid;place-items:center;
  font-family:var(--display);font-weight:700;font-size:.95rem}
.dm-row-body{flex:1;min-width:0}
.dm-meta{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.84rem;color:var(--ink2)}
.dm-verdict{font-weight:600}
.dm-text{margin-top:.55rem;line-height:1.7;color:var(--ink);font-size:1rem;max-width:68ch;overflow-wrap:anywhere}
.dm-det{margin-top:.7rem;font-size:.86rem;color:var(--ink2);line-height:1.55}
.dm-det summary{cursor:pointer;color:var(--blue);font-weight:500}
.dm-score{flex:none;width:92px;text-align:right}
.dm-score b{display:block;font-family:var(--display);font-size:1.7rem;font-weight:700;letter-spacing:-.03em;line-height:1.1;color:var(--ink)}
.dm-meter{height:6px;border-radius:3px;background:rgba(255,255,255,.12);margin-top:.45rem;overflow:hidden}
.dm-meter i{display:block;height:100%;border-radius:3px}
.dm-score small{color:var(--ink2);font-size:.78rem}

/* ---- Meaning map: the one expressive element ---- */
[data-testid="stColumn"]:has(.dm-map-anchor),[data-testid="column"]:has(.dm-map-anchor){
  position:sticky;top:3.6rem;align-self:flex-start}
.dm-map-title{font-family:var(--display);font-weight:700;font-size:1.15rem;letter-spacing:-.015em;color:var(--ink)}
.dm-map-sub{color:var(--ink2);font-size:.88rem;line-height:1.5;margin:.25rem 0 .6rem}
.dm-map{width:100%;height:auto;display:block;font-family:var(--body)}
.dm-bg circle{fill:var(--ink);fill-opacity:.28}
.dm-link{stroke:var(--ink);stroke-opacity:.45;stroke-width:1.2;stroke-dasharray:3 4;animation:dmFade .6s ease .55s backwards}
.dm-hit{transform-box:fill-box;transform-origin:center;transition:transform .15s;
  animation:dmFly .95s cubic-bezier(.2,.9,.25,1) backwards}
.dm-num{font-family:var(--display);font-weight:700;font-size:13px;text-anchor:middle;dominant-baseline:central}
.dm-q-dot{fill:var(--ink)}
.dm-q-ring{fill:none;stroke:var(--ink);stroke-opacity:.4;stroke-width:1.5}
.dm-pulse{fill:none;stroke:var(--ink);stroke-width:2;transform-box:fill-box;transform-origin:center;
  animation:dmPulse 1.3s ease-out 1 both}
.dm-q-lab{font-size:12.5px;font-weight:600;fill:var(--ink);stroke:var(--paper);stroke-width:4px;paint-order:stroke}
.dm-legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.84rem;color:var(--ink2);margin-top:.6rem}
.dm-legend span{display:inline-flex;align-items:center;gap:6px}
.dm-legend i{width:10px;height:10px;border-radius:50%;display:inline-block}
@keyframes dmFly{from{transform:translate(var(--dx),var(--dy));opacity:0}}
@keyframes dmFade{from{opacity:0}}
@keyframes dmPulse{from{transform:scale(1);opacity:.7}to{transform:scale(5);opacity:0}}

/* ---- Landing ---- */
.dm-landing{display:grid;grid-template-columns:1.05fr 1fr;gap:2rem;align-items:start;margin-top:.6rem}
.dm-steps{list-style:none;margin:0;padding:0}
.dm-steps li{display:flex;gap:1rem;padding-bottom:1.35rem;position:relative}
.dm-steps li:not(:last-child)::after{content:"";position:absolute;left:15px;top:34px;bottom:2px;width:2px;background:var(--line)}
.dm-n{flex:none;width:32px;height:32px;border-radius:50%;background:var(--ink);color:var(--bg);display:grid;place-items:center;
  font-family:var(--display);font-weight:700}
.dm-steps b{display:block;font-weight:600;margin-top:4px;color:var(--ink)}
.dm-steps span{color:var(--ink2);line-height:1.55;font-size:.95rem}
.dm-demo{margin-top:1.6rem;display:grid;gap:.7rem;color:var(--ink2);line-height:1.6}
.dm-demo .dm-q{font-weight:600;color:var(--ink)}
.dm-cmp{display:grid;grid-template-columns:1fr 1fr;gap:.7rem}
.dm-cmp div{border-radius:10px;padding:.7rem .9rem;font-size:.92rem;line-height:1.45}
.dm-cmp .no{background:var(--paper2);color:var(--ink2)}
.dm-cmp .yes{background:rgba(45,191,171,.13);color:#C4F5EC}
.dm-cmp b{display:block;font-weight:600;color:inherit}
.dm-tips{margin:.5rem 0 0;padding-left:1.1rem;color:var(--ink2);line-height:1.75}
.dm-footer{color:var(--ink2);font-size:.85rem;margin-top:2.6rem;padding-top:1rem;border-top:1px solid var(--line)}

@media (max-width:900px){
  .dm-landing{grid-template-columns:1fr}
  .dm-stats{grid-template-columns:1fr 1fr;row-gap:1.2rem}
  .dm-stat:nth-child(3){padding-left:0;border-left:none}
  .dm-row{flex-wrap:wrap}.dm-score{width:100%;text-align:left}
}
@media (prefers-reduced-motion:reduce){
  .dm-hit,.dm-link,.dm-pulse{animation:none!important}
}
</style>
"""


def hover_links_css() -> str:
    """
    Two-way hover link between a result row and its dot on the map, using the CSS
    :has() selector (no JavaScript needed). Hovering either one highlights the other.
    """
    rules = []
    for rank in range(1, 11):
        rules.append(
            f'.stApp:has(.dm-row[data-r="{rank}"]:hover) .dm-hit[data-r="{rank}"]{{transform:scale(1.45)}}'
            f'.stApp:has(.dm-hit[data-r="{rank}"]:hover) .dm-row[data-r="{rank}"]{{background:var(--blue-soft)}}'
        )
    return "<style>" + "".join(rules) + "</style>"


def inject_css() -> None:
    st.markdown(STYLE_BASE + hover_links_css(), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# HTML component builders
# (every builder returns ONE unindented line: Markdown would treat indented or
#  blank-line-separated HTML as code instead of rendering it)
# --------------------------------------------------------------------------- #
def logo_svg(size: int) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 32 32" aria-hidden="true">'
        '<circle cx="16" cy="16" r="15" fill="#E7EBF5"/>'
        '<path d="M16 16 8 10M16 16l7-7M16 16l6 7" stroke="#0E1320" stroke-opacity=".4" stroke-width="1"/>'
        '<circle cx="8" cy="10" r="2.6" fill="#0B8577"/><circle cx="23" cy="9" r="2.6" fill="#E0A020"/>'
        '<circle cx="22" cy="23" r="2.6" fill="#D1405B"/><circle cx="16" cy="16" r="3.2" fill="#0E1320"/></svg>'
    )


def confidence_style(confidence: float) -> Tuple[str, str, str, str]:
    """Map a confidence % to (marker fill, text colour on fill, text ink, label)."""
    if confidence >= STRONG_THRESHOLD:
        return "#2DBFAB", "#0E1320", "#5EEAD4", "Strong match"
    if confidence >= MODERATE_THRESHOLD:
        return "#F5B942", "#0E1320", "#FBCB6B", "Related"
    return "#F0647F", "#0E1320", "#FF8FA5", "Weak match"


def banner(kind: str, title: str, body: str) -> None:
    icons = {"success": "✓", "info": "i", "warn": "!", "error": "×"}
    st.markdown(
        f'<div class="dm-banner dm-{kind}"><div class="dm-ico">{icons[kind]}</div>'
        f"<div><b>{title}</b>{body}</div></div>",
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="dm-sec">{title}</div>', unsafe_allow_html=True)


def histogram_html(word_counts: List[int]) -> str:
    """Chunk-size distribution drawn as plain HTML/CSS bars (no chart library)."""
    n_bins = min(12, max(len(set(word_counts)), 1))
    counts, edges = np.histogram(word_counts, bins=n_bins)
    peak = max(int(counts.max()), 1)
    columns = []
    for i, count in enumerate(counts):
        height = (count / peak) * 100 if count else 0
        low, high = int(edges[i]), int(edges[i + 1])
        columns.append(
            f'<div class="dm-hcol" title="{low} to {high} words: {int(count)} passages">'
            f'<div class="dm-hplot"><div class="dm-hbar" style="height:{height:.1f}%">'
            f'<span class="dm-hval">{int(count)}</span></div></div>'
            f'<div class="dm-hlab">{low}</div></div>'
        )
    return f'<div class="dm-hist">{"".join(columns)}</div>'


def build_map_svg(
    points: np.ndarray,
    query_xy: Optional[np.ndarray],
    hits: List[Dict],
    size: int = 440,
) -> str:
    """
    Draw the meaning map as inline SVG.
    `points` and `query_xy` are in the unit square; each hit dict carries
    rank, xy, fill, on (text colour) and title.
    """
    pad = 34
    span = size - 2 * pad

    def px(p: np.ndarray) -> Tuple[float, float]:
        return pad + float(p[0]) * span, pad + float(p[1]) * span

    # Background dots: every passage in the document (capped for performance).
    if len(points) > MAX_MAP_POINTS:
        chosen = np.random.default_rng(0).choice(len(points), MAX_MAP_POINTS, replace=False)
        points = points[chosen]
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6"/>' for x, y in (px(p) for p in points)
    )

    parts = [
        f'<svg class="dm-map" viewBox="0 0 {size} {size}" role="img" '
        f'aria-label="Map of document passages arranged by meaning">',
        f'<g class="dm-bg">{dots}</g>',
    ]

    if query_xy is not None:
        qx, qy = px(query_xy)
        for hit in hits:  # connector lines first so dots sit on top
            hx, hy = px(hit["xy"])
            parts.append(f'<line class="dm-link" x1="{qx:.1f}" y1="{qy:.1f}" x2="{hx:.1f}" y2="{hy:.1f}"/>')
        for hit in hits:
            hx, hy = px(hit["xy"])
            rank = hit["rank"]
            parts.append(
                f'<g class="dm-hit" data-r="{rank}" style="--dx:{qx - hx:.1f}px;--dy:{qy - hy:.1f}px;'
                f'animation-delay:{(rank - 1) * 70}ms">'
                f'<title>{html.escape(hit["title"])}</title>'
                f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="13" fill="{hit["fill"]}" stroke="{PANEL}" stroke-width="2"/>'
                f'<text class="dm-num" x="{hx:.1f}" y="{hy:.1f}" fill="{hit["on"]}">{rank}</text></g>'
            )
        on_right = qx > size * 0.6
        label_x = qx - 20 if on_right else qx + 20
        anchor = "end" if on_right else "start"
        parts.append(
            f'<circle class="dm-pulse" cx="{qx:.1f}" cy="{qy:.1f}" r="8"/>'
            f'<circle class="dm-q-ring" cx="{qx:.1f}" cy="{qy:.1f}" r="14"/>'
            f'<circle class="dm-q-dot" cx="{qx:.1f}" cy="{qy:.1f}" r="7"/>'
            f'<text class="dm-q-lab" x="{label_x:.1f}" y="{qy + 4:.1f}" text-anchor="{anchor}">Your question</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def map_panel(svg: str, subtitle: str, show_legend: bool) -> str:
    legend = (
        '<div class="dm-legend">'
        '<span><i style="background:#2DBFAB"></i>Strong match</span>'
        '<span><i style="background:#F5B942"></i>Related</span>'
        '<span><i style="background:#F0647F"></i>Weak match</span></div>'
        if show_legend
        else ""
    )
    return (
        '<div class="dm-map-anchor"></div><div class="dm-panel">'
        '<div class="dm-map-title">Meaning map</div>'
        f'<div class="dm-map-sub">{subtitle}</div>{svg}{legend}</div>'
    )


def results_html(hits: List[Dict], show_page: bool) -> str:
    """One panel holding every ranked result: rank marker, snippet, confidence."""
    rows = []
    for hit in hits:
        chunk: Chunk = hit["chunk"]
        fill, on, ink, label = confidence_style(hit["confidence"])
        meta = [f'<span class="dm-verdict" style="color:{ink}">{label}</span>']
        if show_page:
            meta.append(f"<span>Page {chunk.page}</span>")
        meta.append(f"<span>Passage {chunk.chunk_id + 1}</span>")
        meta.append(f"<span>{chunk.words} words</span>")
        cosine = hit["confidence"] / 100.0
        rows.append(
            f'<div class="dm-row" data-r="{hit["rank"]}">'
            f'<div class="dm-mark" style="background:{fill};color:{on}">{hit["rank"]}</div>'
            f'<div class="dm-row-body"><div class="dm-meta">{"".join(meta)}</div>'
            # html.escape stops document text from breaking (or injecting into) the markup.
            f'<div class="dm-text">{html.escape(chunk.text)}</div>'
            f'<details class="dm-det"><summary>How was this score calculated?</summary>'
            f'FAISS returned a squared L2 distance of {hit["distance"]:.4f}. '
            f'Cosine similarity = 1 − {hit["distance"]:.4f} ÷ 2 = {cosine:.4f}, shown as {hit["confidence"]:.1f}%.'
            f"</details></div>"
            f'<div class="dm-score"><b>{hit["confidence"]:.0f}%</b>'
            f'<div class="dm-meter"><i style="width:{min(hit["confidence"], 100):.1f}%;background:{fill}"></i></div>'
            f"<small>confidence</small></div></div>"
        )
    return f'<div class="dm-results">{"".join(rows)}</div>'


# --------------------------------------------------------------------------- #
# Page sections
# --------------------------------------------------------------------------- #
def render_header() -> None:
    st.markdown(
        f'<div class="dm-head">{logo_svg(44)}'
        '<div><div class="dm-wordmark">DocuMind</div>'
        '<div class="dm-tagline">Ask questions about a document in plain language. Find answers by meaning, not keywords.</div></div></div>',
        unsafe_allow_html=True,
    )


def sample_map_svg() -> str:
    """A static, seeded example map shown on the landing page before any upload."""
    rng = np.random.default_rng(7)
    centres = np.array([[0.26, 0.30], [0.72, 0.24], [0.52, 0.76]])
    points = np.vstack([c + rng.normal(0, 0.07, size=(38, 2)) for c in centres])
    points = np.clip(points, 0.04, 0.96)
    query = np.array([0.66, 0.36])
    nearest = np.argsort(np.linalg.norm(points - query, axis=1))[:4]
    confidences = [74.0, 66.0, 47.0, 31.0]
    hits = []
    for rank, (idx, conf) in enumerate(zip(nearest, confidences), start=1):
        fill, on, _, _ = confidence_style(conf)
        hits.append(
            {"rank": rank, "xy": points[idx], "fill": fill, "on": on,
             "title": f"Example passage ranked {rank}, {conf:.0f}% confidence"}
        )
    return build_map_svg(np.delete(points, nearest, axis=0), query, hits)


def start_sample() -> None:
    """Callback: load the built-in sample document."""
    st.session_state.use_sample = True
    st.session_state.query_input = ""


def stop_sample() -> None:
    """Callback: unload the sample document and return to the start screen."""
    st.session_state.use_sample = False
    st.session_state.query_input = ""


def render_landing() -> None:
    banner(
        "info",
        "Start by choosing a document",
        "Upload a PDF or TXT file from the sidebar, or try the built-in sample below. "
        "Everything runs on your computer, so your files stay private.",
    )
    c1, _ = st.columns([2, 3])
    with c1:
        st.button(
            "Try a sample document", key="sample_main", type="primary", on_click=start_sample,
            help="Loads a short guide covering eight different topics so you can try a search straight away.",
        )

    steps = [
        ("Split", "The text is cut into overlapping passages of about 150 words, always at the end of a sentence."),
        ("Embed", "A local model turns each passage into 384 numbers that capture what it means."),
        ("Index", "FAISS stores those vectors so the closest ones can be found in milliseconds."),
        ("Search", "Your question gets the same treatment, and the nearest passages are ranked for you."),
    ]
    items = "".join(
        f'<li><div class="dm-n">{i}</div><div><b>{title}</b><span>{text}</span></div></li>'
        for i, (title, text) in enumerate(steps, start=1)
    )
    left = (
        f'<div><div class="dm-sec" style="margin-top:0">How it works</div><ul class="dm-steps">{items}</ul>'
        '<div class="dm-demo"><div class="dm-sec" style="font-size:1.1rem;margin:.4rem 0 0">Why meaning beats keywords</div>'
        '<div>Question: <span class="dm-q">“fixing my car”</span><br>'
        'Passage: “Automobile repair guide: diagnosing engine and brake faults.”</div>'
        '<div class="dm-cmp"><div class="no"><b>Keyword search</b>No shared words, so the passage is missed.</div>'
        '<div class="yes"><b>DocuMind</b>71% close in meaning, so it ranks first.</div></div></div></div>'
    )
    right = map_panel(
        sample_map_svg(),
        "An example of what you will see: every dot is a passage, and the numbered ones are the best answers to a question.",
        show_legend=True,
    )
    st.markdown(f'<div class="dm-landing">{left}<div>{right}</div></div>', unsafe_allow_html=True)


def render_stats_strip(doc: Dict) -> None:
    """Headline numbers shown above the search box."""
    stats = [
        (f"{doc['total_words']:,}", "words processed"),
        (f"{len(doc['chunks']):,}", "vector chunks created"),
        (f"{doc['avg_chunk']:.0f}", "words per chunk on average"),
        (f"{doc['build_seconds']:.1f} s", "to read and index"),
    ]
    stat_html = "".join(f'<div class="dm-stat"><b>{v}</b><span>{label}</span></div>' for v, label in stats)
    st.markdown(f'<div class="dm-panel"><div class="dm-stats">{stat_html}</div></div>', unsafe_allow_html=True)


def render_insights(doc: Dict) -> None:
    """Analytics tab: chunk-size distribution and plain-language file facts."""
    pages = f"{doc['pages']} page{'s' if doc['pages'] != 1 else ''} contained text. " if doc["is_pdf"] else ""
    note = (
        f"<b>{html.escape(doc['filename'])}</b>: {pages}Chunks range from {doc['min_chunk']} to {doc['max_chunk']} words, "
        f"and each one is stored as a {EMBEDDING_DIM}-dimensional vector in FAISS. "
        "Most chunks should sit near your chosen chunk size. A few shorter ones at the left are normal, "
        "because the last chunk of a page or file is rarely full."
    )
    st.markdown(
        '<div class="dm-panel"><div class="dm-histtitle">Chunk size distribution (words per chunk)</div>'
        f'{histogram_html(doc["word_counts"])}<div class="dm-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def render_tips() -> None:
    st.markdown(
        '<div class="dm-panel"><div class="dm-map-title">Ask in your own words</div>'
        '<ul class="dm-tips"><li>Describe an idea instead of a keyword, like “ways to cut costs”.</li>'
        "<li>Ask a full question, like “what causes inflation?”.</li>"
        "<li>Passages are ranked by how close their meaning is to yours, even when no words match.</li>"
        "<li>Not sure where to start? Click one of the suggestions above the box.</li></ul></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# State, sidebar and history
# --------------------------------------------------------------------------- #
def init_state() -> None:
    """Initialise Streamlit session state (survives reruns within a browser tab)."""
    defaults = {
        "doc": None,  # processed document bundle
        "doc_signature": None,  # detects when file or chunk settings change
        "history": [],  # list of past queries (session analytics)
        "last_logged": None,  # prevents logging the same query on every rerun
        "query_input": "",  # bound to the search box
        "use_sample": False,  # True while the built-in sample document is loaded
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_query(query: str) -> None:
    """Callback for suggestion chips and history buttons: run a query."""
    st.session_state.query_input = query


def render_history(container) -> None:
    """Draw the query history and session analytics inside the sidebar container."""
    history: List[Dict] = st.session_state.history
    with container:
        st.markdown('<div class="dm-sidesec">Search history</div>', unsafe_allow_html=True)
        if not history:
            st.caption("Your searches will appear here. Click one to run it again.")
            return

        avg_latency = float(np.mean([h["latency_ms"] for h in history]))
        count = len(history)
        st.caption(
            f"{count} search{'es' if count != 1 else ''} this session, {avg_latency:.0f} ms on average."
        )
        for i, item in enumerate(list(reversed(history))[:MAX_HISTORY_ITEMS]):
            label = item["query"] if len(item["query"]) <= 34 else item["query"][:31] + "…"
            st.button(label, key=f"hist_{count}_{i}", on_click=set_query, args=(item["query"],),
                      help="Run this search again")
            st.caption(f"{item['time']}, {item['n_results']} passages, best {item['top_conf']:.0f}%")

        if st.button("Clear history", key="clear_history"):
            st.session_state.history = []
            st.session_state.last_logged = None
            st.rerun()


def render_sidebar():
    """Build the sidebar controls; returns the settings and a slot for the history."""
    with st.sidebar:
        st.markdown(
            f'<div class="dm-brand">{logo_svg(30)}<b>DocuMind</b></div>'
            '<div class="dm-sidesec" style="margin-top:0">1. Choose a document</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload a PDF or TXT file", type=["pdf", "txt"], label_visibility="collapsed",
            help="Your file is processed on this computer and never uploaded anywhere.",
        )
        if uploaded is None and not st.session_state.use_sample:
            st.button("Or try a sample document", key="sample_side", on_click=start_sample)
        elif uploaded is None:
            st.button("Close the sample document", key="sample_close", on_click=stop_sample)

        st.markdown('<div class="dm-sidesec">2. How many results?</div>', unsafe_allow_html=True)
        top_k = st.slider("Results to show", 1, 10, 5, label_visibility="collapsed")

        with st.expander("Advanced settings"):
            st.caption("The defaults work well for most documents.")
            chunk_size = st.slider(
                "Chunk size (words)", 50, 400, 150, 10,
                help="Larger chunks keep more context. Smaller chunks give more precise answers.",
            )
            overlap = st.slider(
                "Chunk overlap (words)", 0, 100, 30, 5,
                help="Words shared between neighbouring chunks so ideas aren't cut in half.",
            )
            overlap = min(overlap, chunk_size // 2)  # overlap must stay well below chunk size
            min_confidence = st.slider(
                "Hide matches below (%)", 0, 90, 0, 5,
                help="Raise this to see only the closest matches.",
            )
        history_container = st.container()  # filled at the end of the script run
    return uploaded, chunk_size, overlap, top_k, min_confidence, history_container


# --------------------------------------------------------------------------- #
# Workspace (everything in the main column)
# --------------------------------------------------------------------------- #
def render_workspace(
    uploaded, chunk_size: int, overlap: int, top_k: int, min_confidence: int
) -> None:
    # Decide which document to work on: an uploaded file wins over the sample.
    if uploaded is not None:
        st.session_state.use_sample = False
        file_bytes, name, preset = uploaded.getvalue(), uploaded.name, None
    elif st.session_state.use_sample:
        file_bytes, name, preset = SAMPLE_TEXT.encode("utf-8"), SAMPLE_NAME, SAMPLE_QUERIES
    else:
        st.session_state.doc = None
        st.session_state.doc_signature = None
        render_landing()
        return

    signature = (hashlib.md5(file_bytes).hexdigest(), chunk_size, overlap)

    if signature != st.session_state.doc_signature:
        status = st.status("Preparing your document…", expanded=True)
        try:
            st.session_state.doc = process_document(
                file_bytes, name, chunk_size, overlap, progress=status.write, suggestions=preset
            )
            st.session_state.doc_signature = signature
            st.session_state.last_logged = None
            status.update(label="Your document is ready", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            st.session_state.doc = None
            st.session_state.doc_signature = None
            status.update(label="Something went wrong", state="error", expanded=False)
            banner(
                "error",
                f"Couldn't read {html.escape(name)}",
                f"{html.escape(str(exc))} Try a different file, or use the sample document.",
            )
            return

    doc: Dict = st.session_state.doc
    banner(
        "success",
        f"{html.escape(doc['filename'])} is ready to search",
        f"{len(doc['chunks']):,} chunks were embedded and indexed in {doc['build_seconds']:.1f} seconds. "
        "Type a question below to begin.",
    )
    render_stats_strip(doc)

    tab_search, tab_insights = st.tabs(["Search", "Document insights"])

    with tab_insights:
        render_insights(doc)

    with tab_search:
        section("What would you like to know?")
        query = st.text_input(
            "Search query",
            key="query_input",
            label_visibility="collapsed",
            placeholder="Ask a question or describe a topic, for example “how can I save money?”",
        ).strip()
        st.markdown('<div class="dm-hint">Press Enter to search.</div>', unsafe_allow_html=True)

        suggestions = doc["suggestions"]
        if suggestions:
            st.caption("Not sure what to ask? Try one of these:")
            for col, text in zip(st.columns(len(suggestions)), suggestions):
                label = text if len(text) <= 48 else text[:45] + "…"
                col.button(label, key=f"sugg_{text}", on_click=set_query, args=(text,),
                           help="Click to search for this")

        hits: List[Dict] = []
        query_vector: Optional[np.ndarray] = None
        latency_ms = 0.0
        if query:
            model = load_model()
            started = time.perf_counter()
            hits, query_vector = semantic_search(query, model, doc["index"], doc["chunks"], top_k)
            latency_ms = (time.perf_counter() - started) * 1000.0
            hits = [h for h in hits if h["confidence"] >= min_confidence]

            # Log each distinct query once (Streamlit reruns the script on every widget change).
            log_key = (query, st.session_state.doc_signature)
            if log_key != st.session_state.last_logged:
                st.session_state.history.append(
                    {
                        "query": query,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "n_results": len(hits),
                        "top_conf": hits[0]["confidence"] if hits else 0.0,
                        "latency_ms": latency_ms,
                    }
                )
                st.session_state.last_logged = log_key

        st.write("")
        left, right = st.columns([7, 5], gap="large")

        with left:
            if not query:
                render_tips()
            elif not hits:
                banner(
                    "warn",
                    "No passages reached the minimum confidence",
                    "Lower “Hide matches below” in Advanced settings, or rephrase your question.",
                )
            else:
                average = float(np.mean([h["confidence"] for h in hits]))
                st.markdown(
                    f'<div class="dm-summary"><b>{len(hits)} passage{"s" if len(hits) != 1 else ""}</b> ranked in '
                    f"<b>{latency_ms:.0f} ms</b>. Best match <b>{hits[0]['confidence']:.0f}%</b>, "
                    f"average <b>{average:.0f}%</b>.</div>",
                    unsafe_allow_html=True,
                )
                st.caption("A higher percentage means the passage is closer in meaning to your question.")
                if hits[0]["confidence"] < MODERATE_THRESHOLD:
                    banner(
                        "warn",
                        "Nothing matches closely",
                        f"The best match is only {hits[0]['confidence']:.0f}%. "
                        "Try describing the idea differently, or ask about a topic this document covers.",
                    )
                st.markdown(results_html(hits, doc["is_pdf"]), unsafe_allow_html=True)

                st.write("")
                export_lines = ["rank,confidence_pct,page,text"]
                for h in hits:
                    text = h["chunk"].text.replace('"', '""')
                    export_lines.append(f'{h["rank"]},{h["confidence"]:.2f},{h["chunk"].page},"{text}"')
                st.download_button(
                    "Download these results (CSV)",
                    "\n".join(export_lines).encode("utf-8"),
                    file_name="documind_results.csv",
                    mime="text/csv",
                )

        with right:
            points, query_xy = normalize_for_map(doc, query_vector)
            map_hits = [
                {
                    "rank": h["rank"],
                    "xy": points[h["chunk_index"]],
                    "fill": confidence_style(h["confidence"])[0],
                    "on": confidence_style(h["confidence"])[1],
                    "title": f'#{h["rank"]}, {h["confidence"]:.0f}%: {h["chunk"].text[:90]}…',
                }
                for h in hits
            ]
            # Hit dots are drawn separately, so remove them from the grey background set.
            hit_ids = [h["chunk_index"] for h in hits]
            background = np.delete(points, hit_ids, axis=0) if hit_ids else points
            subtitle = (
                "Each grey dot is a passage from your document. Numbered dots are your best matches, "
                "placed by similarity to the question. The map flattens 384 dimensions into 2, so distances are approximate."
                if hits
                else "Each dot is one passage from your document. Passages with similar meaning sit close together."
            )
            svg = build_map_svg(background, query_xy if hits else None, map_hits)
            st.markdown(map_panel(svg, subtitle, show_legend=bool(hits)), unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        '<div class="dm-footer">DocuMind runs entirely on your computer. '
        "Embeddings come from all-MiniLM-L6-v2 and the vector index is FAISS.</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="DocuMind: semantic document search", page_icon="🧠", layout="wide"
    )
    inject_css()
    init_state()

    uploaded, chunk_size, overlap, top_k, min_confidence, history_container = render_sidebar()

    render_header()
    render_workspace(uploaded, chunk_size, overlap, top_k, min_confidence)
    render_history(history_container)  # after the workspace so new queries show immediately
    render_footer()


if __name__ == "__main__":
    main()
