import os
import time
import hashlib
from html import escape
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
import numpy as np

from transformers import T5ForConditionalGeneration, T5Tokenizer

try:
    import faiss  # faiss-cpu
except Exception:
    faiss = None


# -----------------------------
# Config
# -----------------------------
OPENFDA_BASE = "https://api.fda.gov"
DEFAULT_LIMIT = 25  # keep small for demo speed
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOGO_PATH = Path(__file__).parent / "assets" / "RECALL.png"


# -----------------------------
# Data model
# -----------------------------
@dataclass
class FindingCard:
    id: str
    category: str  # drug/food/device
    product: str
    classification: str
    reason: str
    recalling_firm: str
    recall_initiation_date: str
    status: str
    summary_text: str
    raw: Dict[str, Any]


def _safe_get(d: Dict[str, Any], key: str, default: str = "") -> str:
    v = d.get(key, default)
    if v is None:
        return default
    return str(v)


def _hash_id(*parts: str) -> str:
    s = "|".join([p or "" for p in parts])
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# -----------------------------
# openFDA client (enforcement)
# -----------------------------
def openfda_enforcement_search(
    category: str,
    query: str,
    limit: int = DEFAULT_LIMIT,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    category: "drug" | "food" | "device"
    Uses openFDA enforcement endpoint.
    """
    endpoint = f"{OPENFDA_BASE}/{category}/enforcement.json"

    # Very simple search: look for query in product_description OR recalling_firm.
    # openFDA uses Lucene-like search syntax.
    q = f'product_description:"{query}" OR recalling_firm:"{query}"'

    params = {
        "search": q,
        "limit": limit,
    }
    if api_key:
        params["api_key"] = api_key

    r = requests.get(endpoint, params=params, timeout=20)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    data = r.json()
    return data.get("results", []) or []


def normalize_to_cards(category: str, results: List[Dict[str, Any]]) -> List[FindingCard]:
    cards: List[FindingCard] = []
    for row in results:
        product = _safe_get(row, "product_description", "").strip()
        classification = _safe_get(row, "classification", "").strip()
        reason = _safe_get(row, "reason_for_recall", "").strip()
        firm = _safe_get(row, "recalling_firm", "").strip()
        init_date = _safe_get(row, "recall_initiation_date", "").strip()
        status = _safe_get(row, "status", "").strip()

        # Construct a human-readable summary that we embed + display
        summary = (
            f"[{category.upper()} RECALL] {product}\n"
            f"Classification: {classification}\n"
            f"Reason: {reason}\n"
            f"Firm: {firm}\n"
            f"Initiation Date: {init_date}\n"
            f"Status: {status}\n"
        ).strip()

        cid = _hash_id(category, product, firm, init_date, reason, classification)
        cards.append(
            FindingCard(
                id=cid,
                category=category,
                product=product,
                classification=classification,
                reason=reason,
                recalling_firm=firm,
                recall_initiation_date=init_date,
                status=status,
                summary_text=summary,
                raw=row,
            )
        )
    return cards


# -----------------------------
# Embeddings + FAISS (local)
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def build_faiss_index(vectors: np.ndarray):
    if faiss is None:
        raise RuntimeError("faiss-cpu is not available. Install faiss-cpu.")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity if vectors are normalized
    index.add(vectors)
    return index


def faiss_search(index, query_vec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    # query_vec should be shape (1, dim)
    scores, ids = index.search(query_vec, top_k)
    return scores[0], ids[0]


# -----------------------------
# Hugging Face LLM (local)
# -----------------------------
@st.cache_resource
def load_llm():
    # Flan-T5: use local path if set (avoids hub download); otherwise hub id
    model_name = os.environ.get("RECALL_MODEL_PATH", "google/flan-t5-base")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer


def answer_with_llm(question: str, context_cards: List[FindingCard]) -> str:
    """
    Simple RAG: retrieved cards -> concatenated context -> LLM answer.
    Uses ONLY provided context.
    """
    model, tokenizer = load_llm()

    context_blocks = []
    for c in context_cards:
        context_blocks.append(
            f"- Product: {c.product}\n"
            f"  Classification: {c.classification}\n"
            f"  Reason: {c.reason}\n"
            f"  Firm: {c.recalling_firm}\n"
            f"  Date: {c.recall_initiation_date}\n"
            f"  Status: {c.status}\n"
        )

    # Keep prompt bounded for CPU + model context constraints
    context = "\n".join(context_blocks)[:3500]

    prompt = f"""
You are RECALL, a safety assistant for FDA recall information.
Answer the user's question using ONLY the context below.
If the context is insufficient, say: "I don't have enough information in these results."

Question:
{question}

Context:
{context}

Answer (concise, bullet points if helpful):
""".strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# -----------------------------
# Simple insight helpers
# -----------------------------
def top_themes(cards: List[FindingCard], n: int = 5) -> List[Tuple[str, int]]:
    """
    Extremely simple "themes": top words in reasons (excluding tiny stopwords).
    Keeps it prototype-simple.
    """
    stop = set(["the", "and", "or", "for", "with", "due", "to", "of", "in", "on", "a", "an", "is", "are"])
    tokens = []
    for c in cards:
        words = (c.reason or "").lower().replace("/", " ").replace(",", " ").replace(".", " ").split()
        tokens.extend([w for w in words if len(w) >= 4 and w not in stop])
    if not tokens:
        return []
    s = pd.Series(tokens)
    counts = s.value_counts().head(n)
    return list(zip(counts.index.tolist(), counts.values.tolist()))


def _format_recall_date(yyyymmdd: str) -> str:
    """Turn YYYYMMDD into a readable date (e.g. 'Oct 3, 2025')."""
    if not yyyymmdd or len(yyyymmdd) != 8 or not yyyymmdd.isdigit():
        return yyyymmdd
    try:
        from datetime import datetime

        dt = datetime.strptime(yyyymmdd, "%Y%m%d")
        return dt.strftime("%b ") + str(dt.day) + dt.strftime(", %Y")
    except Exception:
        return yyyymmdd


def summarize_changes(cards: List[FindingCard]) -> Tuple[str, str, str]:
    """
    Prototype "what changed": latest recall date, most common classification, most common status.
    Returns (latest_date_readable, classification, status) for display.
    """
    dates = [c.recall_initiation_date for c in cards if c.recall_initiation_date]
    cls = [c.classification for c in cards if c.classification]
    status = [c.status for c in cards if c.status]

    def most_common(xs):
        if not xs:
            return ""
        return pd.Series(xs).value_counts().index[0]

    latest_raw = ""
    if dates:
        latest_raw = sorted(dates)[-1]
    latest_readable = _format_recall_date(latest_raw) if latest_raw else ""
    mc_cls = most_common(cls)
    mc_status = most_common(status)
    return (latest_readable, mc_cls, mc_status)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="RECALL", layout="wide", page_icon="📋")

# -----------------------------
# VISUAL UPGRADE (CSS ONLY)
# -----------------------------
st.markdown(
    """
<style>
/* ---------- Brand Tokens ---------- */
:root{
  --bg: #2F3545;            /* brand slate/navy */
  --bg2:#262B38;            /* deeper */
  --panel:#30384A;          /* panels */
  --panel2:#2A3040;         /* hover / deeper panels */
  --text:#FFFFFF;           /* primary text */
  --muted:#B8C0D0;          /* secondary text */
  --muted2:#9AA3B6;         /* tertiary text */
  --accent:#F28C6C;         /* coral accent */
  --accent2:#E07A5A;        /* hover coral */
  --border: rgba(255,255,255,.10);
  --border2: rgba(242,140,108,.25);
  --shadow: 0 12px 30px rgba(0,0,0,.35);
  --shadow2: 0 10px 24px rgba(0,0,0,.25);
  --radius: 16px;
}

/* ---------- App Base ---------- */
html, body, [class*="css"]  { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
.stApp {
  background: radial-gradient(1000px 600px at 30% 10%, rgba(242,140,108,.10), rgba(0,0,0,0) 60%),
              radial-gradient(900px 600px at 85% 25%, rgba(255,255,255,.06), rgba(0,0,0,0) 55%),
              linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
  color: var(--text);
}

h1,h2,h3,h4,h5,h6 { color: var(--text) !important; letter-spacing: -0.02em; }
p, li, label, span, div { color: var(--text); }
small, .stCaption, .stMarkdown p { color: var(--muted) !important; }

/* Make the main container feel centered and premium */
.block-container {
  padding-top: 2.0rem;
  padding-bottom: 2.5rem;
  max-width: 1200px;
}

/* ---------- Sidebar ---------- */
div[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(48,56,74,.95) 0%, rgba(38,43,56,.95) 100%);
  border-right: 1px solid var(--border);
}
div[data-testid="stSidebar"] .stMarkdown,
div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] span,
div[data-testid="stSidebar"] p{
  color: var(--text) !important;
}
div[data-testid="stSidebar"] [data-testid="stImage"] img{
  border-radius: 14px;
  box-shadow: var(--shadow2);
}

/* ---------- Inputs (text/select/slider) ---------- */
input, textarea {
  background: rgba(255,255,255,.06) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  color: var(--text) !important;
}
input:focus, textarea:focus{
  outline: none !important;
  border-color: var(--border2) !important;
  box-shadow: 0 0 0 4px rgba(242,140,108,.18) !important;
}

div[data-baseweb="select"] > div {
  background: rgba(255,255,255,.06) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
}
div[data-baseweb="select"] span{
  color: var(--text) !important;
}
div[data-baseweb="select"]:focus-within > div{
  border-color: var(--border2) !important;
  box-shadow: 0 0 0 4px rgba(242,140,108,.18) !important;
}

div[data-testid="stSlider"] > div {
  color: var(--text) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div{
  background: rgba(255,255,255,.10) !important;
}

/* ---------- Buttons ---------- */
.stButton > button{
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
  color: #fff !important;
  font-weight: 700 !important;
  border: 0 !important;
  border-radius: 14px
