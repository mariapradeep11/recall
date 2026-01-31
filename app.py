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
DEFAULT_LIMIT = 25  # keyword results per query
RECENT_DAYS_DEFAULT = 90
RECENT_LIMIT_DEFAULT = 75  # recent context per category
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOGO_PATH = Path(__file__).parent / "assets" / "RECALL.png"

ALL_CATEGORIES = ["drug", "food", "device"]


# -----------------------------
# Data model
# -----------------------------
@dataclass
class FindingCard:
    id: str
    category: str  # drug/food/device
    recall_number: str
    product: str
    classification: str
    reason: str
    recalling_firm: str
    recall_initiation_date: str
    status: str
    distribution_pattern: str
    code_info: str
    product_quantity: str
    voluntary_mandated: str
    state: str
    country: str
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
@st.cache_data(ttl=60 * 60, show_spinner=False)  # ✅ cache for 1 hour (good for testing many times/day)
def _openfda_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(endpoint, params=params, timeout=25)
    if r.status_code == 404:
        return {"results": []}
    r.raise_for_status()
    return r.json()


def _build_keyword_queries(keyword: str) -> List[str]:
    """
    Multi-query union strategy:
    - Each query targets different fields to maximize recall coverage.
    - Keep each query reasonably sized for openFDA's search parser.
    """
    k = keyword.replace('"', '\\"').strip()

    # Query 1: core identification fields
    q1 = (
        f'product_description:"{k}" OR recalling_firm:"{k}" OR brand_name:"{k}"'
    )

    # Query 2: safety / reason / codes (very valuable for “why” questions)
    q2 = (
        f'reason_for_recall:"{k}" OR code_info:"{k}" OR distribution_pattern:"{k}"'
    )

    # Query 3: nested openfda fields when present (brand/generic)
    # (Some records have openfda.* populated; not all.)
    q3 = (
        f'openfda.brand_name:"{k}" OR openfda.generic_name:"{k}" OR openfda.substance_name:"{k}"'
    )

    return [q1, q2, q3]


def openfda_enforcement_search(
    category: str,
    search_query: str,
    limit: int,
    api_key: Optional[str] = None,
    sort: Optional[str] = None,
) -> List[Dict[str, Any]]:
    endpoint = f"{OPENFDA_BASE}/{category}/enforcement.json"
    params: Dict[str, Any] = {"search": search_query, "limit": int(limit)}
    if sort:
        params["sort"] = sort
    if api_key:
        params["api_key"] = api_key

    data = _openfda_get(endpoint, params)
    return data.get("results", []) or []


def openfda_broadened_fetch(
    keyword: str,
    categories: List[str],
    keyword_limit: int,
    include_recent_context: bool,
    recent_days: int,
    recent_limit: int,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    All categories + multi-query union + recent context (by recall_initiation_date range).
    Returns raw openFDA rows (mixed categories) with an added "__category" field.
    """
    from datetime import datetime, timedelta

    all_rows: List[Dict[str, Any]] = []
    seen_keys: set = set()

    keyword = (keyword or "").strip()
    keyword_queries = _build_keyword_queries(keyword) if keyword else []

    # Build date range for recent context
    today = datetime.utcnow().date()
    start = today - timedelta(days=int(recent_days))
    start_s = start.strftime("%Y%m%d")
    end_s = today.strftime("%Y%m%d")

    for cat in categories:
        # A) Keyword expansion (3 queries)
        for q in keyword_queries:
            try:
                rows = openfda_enforcement_search(
                    category=cat,
                    search_query=q,
                    limit=keyword_limit,
                    api_key=api_key,
                )
            except Exception:
                rows = []

            for r in rows:
                recall_number = _safe_get(r, "recall_number", "").strip()
                product = _safe_get(r, "product_description", "").strip()
                firm = _safe_get(r, "recalling_firm", "").strip()
                init_date = _safe_get(r, "recall_initiation_date", "").strip()

                # Dedup key: prefer recall_number; else fallback on stable tuple
                dedupe_key = f"{cat}|{recall_number}" if recall_number else f"{cat}|{product}|{firm}|{init_date}"
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)

                rr = dict(r)
                rr["__category"] = cat
                all_rows.append(rr)

        # B) Recent context (pull last N days even if it doesn’t match keyword)
        if include_recent_context:
            recent_q = f"recall_initiation_date:[{start_s}+TO+{end_s}]"
            try:
                recent_rows = openfda_enforcement_search(
                    category=cat,
                    search_query=recent_q,
                    limit=recent_limit,
                    api_key=api_key,
                    sort="recall_initiation_date:desc",
                )
            except Exception:
                recent_rows = []

            for r in recent_rows:
                recall_number = _safe_get(r, "recall_number", "").strip()
                product = _safe_get(r, "product_description", "").strip()
                firm = _safe_get(r, "recalling_firm", "").strip()
                init_date = _safe_get(r, "recall_initiation_date", "").strip()

                dedupe_key = f"{cat}|{recall_number}" if recall_number else f"{cat}|{product}|{firm}|{init_date}"
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)

                rr = dict(r)
                rr["__category"] = cat
                all_rows.append(rr)

    return all_rows


def normalize_to_cards(results: List[Dict[str, Any]]) -> List[FindingCard]:
    cards: List[FindingCard] = []
    for row in results:
        category = _safe_get(row, "__category", "").strip() or _safe_get(row, "category", "").strip() or "unknown"

        recall_number = _safe_get(row, "recall_number", "").strip()
        product = _safe_get(row, "product_description", "").strip()
        classification = _safe_get(row, "classification", "").strip()
        reason = _safe_get(row, "reason_for_recall", "").strip()
        firm = _safe_get(row, "recalling_firm", "").strip()
        init_date = _safe_get(row, "recall_initiation_date", "").strip()
        status = _safe_get(row, "status", "").strip()

        distribution = _safe_get(row, "distribution_pattern", "").strip()
        code_info = _safe_get(row, "code_info", "").strip()
        qty = _safe_get(row, "product_quantity", "").strip()
        vol = _safe_get(row, "voluntary_mandated", "").strip()
        state = _safe_get(row, "state", "").strip()
        country = _safe_get(row, "country", "").strip()

        # Richer summary => better embeddings + better RAG answers
        summary = (
            f"[{category.upper()} RECALL] {product}\n"
            f"Recall #: {recall_number}\n"
            f"Classification: {classification}\n"
            f"Reason: {reason}\n"
            f"Firm: {firm}\n"
            f"Initiation Date: {init_date}\n"
            f"Status: {status}\n"
            f"Distribution: {distribution}\n"
            f"Codes/Lots: {code_info}\n"
            f"Quantity: {qty}\n"
            f"Voluntary/Mandated: {vol}\n"
            f"Location: {state}, {country}\n"
        ).strip()

        # Prefer a stable ID using recall_number if present
        if recall_number:
            cid = _hash_id(category, recall_number)
        else:
            cid = _hash_id(category, product, firm, init_date, reason, classification)

        cards.append(
            FindingCard(
                id=cid,
                category=category,
                recall_number=recall_number,
                product=product,
                classification=classification,
                reason=reason,
                recalling_firm=firm,
                recall_initiation_date=init_date,
                status=status,
                distribution_pattern=distribution,
                code_info=code_info,
                product_quantity=qty,
                voluntary_mandated=vol,
                state=state,
                country=country,
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
    scores, ids = index.search(query_vec, top_k)
    return scores[0], ids[0]


# -----------------------------
# Hugging Face LLM (local)
# -----------------------------
@st.cache_resource
def load_llm():
    model_name = os.environ.get("RECALL_MODEL_PATH", "google/flan-t5-base")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer


def answer_with_llm(question: str, context_cards: List[FindingCard]) -> str:
    model, tokenizer = load_llm()

    context_blocks = []
    for c in context_cards:
        context_blocks.append(
            f"- Category: {c.category}\n"
            f"  Product: {c.product}\n"
            f"  Recall #: {c.recall_number}\n"
            f"  Classification: {c.classification}\n"
            f"  Reason: {c.reason}\n"
            f"  Firm: {c.recalling_firm}\n"
            f"  Date: {c.recall_initiation_date}\n"
            f"  Status: {c.status}\n"
            f"  Distribution: {c.distribution_pattern}\n"
            f"  Codes/Lots: {c.code_info}\n"
        )

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

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# -----------------------------
# Simple insight helpers
# -----------------------------
def top_themes(cards: List[FindingCard], n: int = 6) -> List[Tuple[str, int]]:
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
    if not yyyymmdd or len(yyyymmdd) != 8 or not yyyymmdd.isdigit():
        return yyyymmdd
    try:
        from datetime import datetime

        dt = datetime.strptime(yyyymmdd, "%Y%m%d")
        return dt.strftime("%b ") + str(dt.day) + dt.strftime(", %Y")
    except Exception:
        return yyyymmdd


def summarize_changes(cards: List[FindingCard]) -> Tuple[str, str, str]:
    dates = [c.recall_initiation_date for c in cards if c.recall_initiation_date]
    cls = [c.classification for c in cards if c.classification]
    status = [c.status for c in cards if c.status]

    def most_common(xs):
        if not xs:
            return ""
        return pd.Series(xs).value_counts().index[0]

    latest_raw = sorted(dates)[-1] if dates else ""
    latest_readable = _format_recall_date(latest_raw) if latest_raw else ""
    mc_cls = most_common(cls)
    mc_status = most_common(status)
    return (latest_readable, mc_cls, mc_status)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(
    page_title="RECALL",
    layout="wide",
    page_icon="📋",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Canonical settings
# -----------------------------
if "category" not in st.session_state:
    st.session_state.category = "all"  # ✅ default to ALL for broader answers
if "limit" not in st.session_state:
    st.session_state.limit = DEFAULT_LIMIT
if "recent_days" not in st.session_state:
    st.session_state.recent_days = RECENT_DAYS_DEFAULT
if "recent_limit" not in st.session_state:
    st.session_state.recent_limit = RECENT_LIMIT_DEFAULT
if "include_recent" not in st.session_state:
    st.session_state.include_recent = True

# API key intentionally removed from UI. If you still set OPENFDA_API_KEY in env, it will be used.
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENFDA_API_KEY", "")


def _sync_from_sb():
    st.session_state.category = st.session_state.sb_category
    st.session_state.limit = st.session_state.sb_limit
    st.session_state.include_recent = st.session_state.sb_include_recent
    st.session_state.recent_days = st.session_state.sb_recent_days
    st.session_state.recent_limit = st.session_state.sb_recent_limit


def _sync_from_main():
    st.session_state.category = st.session_state.main_category
    st.session_state.limit = st.session_state.main_limit
    st.session_state.include_recent = st.session_state.main_include_recent
    st.session_state.recent_days = st.session_state.main_recent_days
    st.session_state.recent_limit = st.session_state.main_recent_limit


# -----------------------------
# VISUAL UPGRADE (CSS ONLY)
# -----------------------------
st.markdown(
    """
<style>
:root{
  --bg: #2F3545;
  --bg2:#262B38;
  --text:#FFFFFF;
  --muted:#B8C0D0;
  --muted2:#9AA3B6;
  --accent:#F28C6C;
  --accent2:#E07A5A;
  --border: rgba(255,255,255,.10);
  --border2: rgba(242,140,108,.25);
  --shadow: 0 12px 30px rgba(0,0,0,.35);
  --shadow2: 0 10px 24px rgba(0,0,0,.25);
  --radius: 16px;
}

html, body, [class*="css"]  { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
.stApp {
  background: radial-gradient(1000px 600px at 30% 10%, rgba(242,140,108,.10), rgba(0,0,0,0) 60%),
              radial-gradient(900px 600px at 85% 25%, rgba(255,255,255,.06), rgba(0,0,0,0) 55%),
              linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
  color: var(--text);
}

h1,h2,h3,h4,h5,h6 { color: var(--text) !important; letter-spacing: -0.02em; }
small, .stCaption, .stMarkdown p { color: var(--muted) !important; }

.block-container {
  padding-top: 2.0rem;
  padding-bottom: 2.5rem;
  max-width: 1200px;
}

/* ✅ FORCE ALL WIDGET LABELS WHITE (fixes Category + Pick a recall label) */
div[data-testid="stWidgetLabel"] *{
  color: #ffffff !important;
  font-weight: 800 !important;
}

/* Sidebar */
div[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(48,56,74,.95) 0%, rgba(38,43,56,.95) 100%);
  border-right: 1px solid var(--border);
}
div[data-testid="stSidebar"] *{
  color: var(--text) !important;
}
div[data-testid="stSidebar"] [data-testid="stImage"] img{
  border-radius: 14px;
  box-shadow: var(--shadow2);
}

/* Inputs */
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

/* Select boxes base */
div[data-baseweb="select"] > div {
  background: rgba(255,255,255,.06) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
}
div[data-baseweb="select"]:focus-within > div{
  border-color: var(--border2) !important;
  box-shadow: 0 0 0 4px rgba(242,140,108,.18) !important;
}

/* ✅ Keep select inputs WHITE with dark text (Category + Pick a recall dropdowns) */
div[data-testid="stSelectbox"] > div > div{
  background: #ffffff !important;
  border: 1px solid rgba(0,0,0,.18) !important;
  border-radius: 14px !important;
}
div[data-testid="stSelectbox"] span{
  color: #000000 !important;
}
div[data-testid="stSelectbox"] svg{
  fill: #000000 !important;
}

/* Buttons */
.stButton > button{
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
  color: #fff !important;
  font-weight: 800 !important;
  border: 0 !important;
  border-radius: 14px !important;
  padding: 0.65rem 1.0rem !important;
  box-shadow: 0 12px 22px rgba(242,140,108,.18), 0 10px 24px rgba(0,0,0,.25) !important;
  transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
}
.stButton > button:hover{
  filter: brightness(1.03);
  transform: translateY(-1px);
  box-shadow: 0 16px 26px rgba(242,140,108,.22), 0 12px 28px rgba(0,0,0,.28) !important;
}
button[kind="primary"]{
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
  border: 0 !important;
}

/* Metrics */
div[data-testid="stMetric"]{
  background: rgba(255,255,255,.05);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.9rem 1rem;
  box-shadow: 0 10px 22px rgba(0,0,0,.18);
}
.stMetric label { color: var(--muted) !important; font-weight: 700; }
div[data-testid="stMetricValue"]{ color: var(--text) !important; font-weight: 900; }

/* Expanders */
details{
  background: rgba(255,255,255,.045);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: 0 10px 22px rgba(0,0,0,.16);
}
details:hover{ border-color: rgba(242,140,108,.22); }
summary{ color: var(--text) !important; font-weight: 800; }

/* Alerts */
div[data-testid="stAlert"]{
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,.05) !important;
  box-shadow: 0 10px 22px rgba(0,0,0,.16);
}
div[data-testid="stAlert"] * { color: var(--text) !important; }

/* Dataframe */
div[data-testid="stDataFrame"]{
  background: rgba(255,255,255,.035);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: 0 12px 26px rgba(0,0,0,.20);
  overflow: hidden;
}
div[data-testid="stDataFrame"] *{ color: var(--text) !important; }

/* Code blocks */
pre, code, .stCodeBlock{
  background: rgba(0,0,0,.22) !important;
  border: 1px solid rgba(255,255,255,.08) !important;
  border-radius: 14px !important;
}

/* Dividers */
hr{
  border: none;
  height: 1px;
  background: linear-gradient(90deg, rgba(255,255,255,0), rgba(242,140,108,.45), rgba(255,255,255,0));
  margin: 1.2rem 0;
}

/* Summary box */
.recall-summary {
  font-family: inherit;
  background: linear-gradient(135deg, rgba(255,255,255,.06) 0%, rgba(255,255,255,.03) 100%);
  border: 1px solid rgba(242,140,108,.22);
  border-left: 5px solid var(--accent);
  padding: 1.05rem 1.2rem;
  border-radius: 14px;
  margin: 0.75rem 0 0.25rem 0;
  color: var(--text);
  box-shadow: 0 14px 30px rgba(0,0,0,.20);
}
.recall-summary-title{
  font-weight: 900;
  font-size: 0.95rem;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.6rem;
}
.recall-summary-label{ color: var(--muted) !important; }
.recall-summary-value{ color: var(--text) !important; font-weight: 900; }

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Focus ring */
.stTextInput input:focus,
.stTextInput textarea:focus {
  outline: none !important;
  box-shadow: 0 0 0 4px rgba(242,140,108,.35) !important;
  border-color: rgba(255,255,255,.22) !important;
}

/* Main Search term input white with black text */
div[data-testid="stTextInput"] input[aria-label="Search term"]{
  background: #ffffff !important;
  color: #000000 !important;
  border: 1px solid rgba(0,0,0,.18) !important;
}
div[data-testid="stTextInput"] input[aria-label="Search term"]::placeholder{
  color: rgba(0,0,0,.55) !important;
}

/* Ask RECALL input white with black text */
div[data-testid="stTextInput"] input[aria-label="Ask a question about these recall results"]{
  background: #ffffff !important;
  color: #000000 !important;
  border: 1px solid rgba(0,0,0,.18) !important;
}
div[data-testid="stTextInput"] input[aria-label="Ask a question about these recall results"]::placeholder{
  color: rgba(0,0,0,.55) !important;
}

/* Slider numeric value (keep readable) */
div[data-testid="stSlider"] span { color: #ffffff !important; }

/* Mobile */
@media (max-width: 768px){
  .block-container{
    padding-top: 1.0rem !important;
    padding-left: 0.85rem !important;
    padding-right: 0.85rem !important;
    padding-bottom: 1.4rem !important;
  }
  div[data-testid="stHorizontalBlock"]{
    flex-direction: column !important;
    gap: 0.75rem !important;
  }
  div[data-testid="stHorizontalBlock"] > div{
    width: 100% !important;
    min-width: 100% !important;
  }
  .stButton > button{
    width: 100% !important;
    padding: 0.85rem 1.0rem !important;
    border-radius: 16px !important;
  }
  input, textarea{
    font-size: 16px !important;
    padding-top: 0.75rem !important;
    padding-bottom: 0.75rem !important;
  }
  pre, code{
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-x: hidden !important;
    font-size: 12px !important;
  }
  div[data-testid="stDataFrame"]{
    overflow-x: auto !important;
  }
}
</style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar (synced)
# -----------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)

    st.header("Search Settings")

    st.selectbox(
        "Category",
        ["all", "drug", "food", "device"],
        index=["all", "drug", "food", "device"].index(st.session_state.category),
        key="sb_category",
        on_change=_sync_from_sb,
    )

    st.slider(
        "Keyword results per query",
        min_value=10,
        max_value=100,
        value=int(st.session_state.limit),
        step=5,
        key="sb_limit",
        on_change=_sync_from_sb,
    )

    st.toggle(
        "Include recent context (recommended)",
        value=bool(st.session_state.include_recent),
        key="sb_include_recent",
        on_change=_sync_from_sb,
    )

    st.slider(
        "Recent window (days)",
        min_value=7,
        max_value=365,
        value=int(st.session_state.recent_days),
        step=7,
        key="sb_recent_days",
        on_change=_sync_from_sb,
    )

    st.slider(
        "Recent records per category",
        min_value=10,
        max_value=200,
        value=int(st.session_state.recent_limit),
        step=5,
        key="sb_recent_limit",
        on_change=_sync_from_sb,
    )

    st.caption("Tip: Cached API calls make repeated testing fast.")


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div style="
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:16px;
  padding:14px 16px;
  border:1px solid rgba(255,255,255,.10);
  border-radius:16px;
  background: rgba(255,255,255,.04);
  box-shadow: 0 12px 26px rgba(0,0,0,.18);
  margin-bottom: 14px;
">
  <div>
    <div style="font-size:34px; font-weight:900; letter-spacing:-0.03em; line-height:1; color:#fff;">
      RECALL<span style="color:#F28C6C;">.</span>
    </div>
    <div style="color:#B8C0D0; font-weight:700; margin-top:6px;">
      FDA recall search • Similarity • Q&A
    </div>
  </div>
  <div style="color:#9AA3B6; font-weight:800; font-size:13px; text-transform:uppercase; letter-spacing:.12em;">
    Broadened retrieval
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Main-page settings (mobile)
# -----------------------------
sc1, sc2, sc3 = st.columns([1, 1, 1], gap="large")
with sc1:
    st.selectbox(
        "Category",
        ["all", "drug", "food", "device"],
        index=["all", "drug", "food", "device"].index(st.session_state.category),
        key="main_category",
        on_change=_sync_from_main,
    )
with sc2:
    st.slider(
        "Keyword results per query",
        min_value=10,
        max_value=100,
        value=int(st.session_state.limit),
        step=5,
        key="main_limit",
        on_change=_sync_from_main,
    )
with sc3:
    st.toggle(
        "Include recent context",
        value=bool(st.session_state.include_recent),
        key="main_include_recent",
        on_change=_sync_from_main,
    )

sc4, sc5 = st.columns([1, 1], gap="large")
with sc4:
    st.slider(
        "Recent window (days)",
        min_value=7,
        max_value=365,
        value=int(st.session_state.recent_days),
        step=7,
        key="main_recent_days",
        on_change=_sync_from_main,
    )
with sc5:
    st.slider(
        "Recent records per category",
        min_value=10,
        max_value=200,
        value=int(st.session_state.recent_limit),
        step=5,
        key="main_recent_limit",
        on_change=_sync_from_main,
    )

# Canonical values
category_choice = st.session_state.category
keyword_limit = int(st.session_state.limit)
include_recent = bool(st.session_state.include_recent)
recent_days = int(st.session_state.recent_days)
recent_limit = int(st.session_state.recent_limit)
api_key = st.session_state.api_key  # hidden from UI; env var only

categories_to_use = ALL_CATEGORIES if category_choice == "all" else [category_choice]


# -----------------------------
# Main UI
# -----------------------------
st.markdown("## RECALL")
st.write(
    "Type a **brand/product keyword** to see recalls. This version pulls:\n"
    "- **All categories** (drug + food + device)\n"
    "- **Multiple openFDA queries** (field expansion) + union & dedupe\n"
    "- **Recent context** (so the AI can answer broader trend questions)\n"
)

col1, col2 = st.columns([2, 1], gap="large")
with col1:
    query = st.text_input("Search term", placeholder="e.g., shrimp, listeria, Ozempic, Philips")
with col2:
    run_btn = st.button("Search openFDA", use_container_width=True)

if "cards" not in st.session_state:
    st.session_state.cards = []
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "index" not in st.session_state:
    st.session_state.index = None

if run_btn and query.strip():
    with st.spinner("Fetching broadened recall data from openFDA (cached for fast repeat testing)..."):
        try:
            raw_rows = openfda_broadened_fetch(
                keyword=query.strip(),
                categories=categories_to_use,
                keyword_limit=keyword_limit,
                include_recent_context=include_recent,
                recent_days=recent_days,
                recent_limit=recent_limit,
                api_key=api_key or None,
            )
        except requests.HTTPError as e:
            st.error(f"openFDA request failed: {e}")
            raw_rows = []
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            raw_rows = []

    cards = normalize_to_cards(raw_rows)
    st.session_state.cards = cards

    if not cards:
        st.warning("No recalls found. Try a broader keyword (e.g., 'salad', 'metal', 'contamination').")
    else:
        with st.spinner("Building local embeddings + FAISS index..."):
            model = load_embedder()
            texts = [c.summary_text for c in cards]
            vecs = embed_texts(model, texts)
            st.session_state.vectors = vecs
            st.session_state.index = build_faiss_index(vecs)

        st.success(f"Loaded {len(cards)} recall records across {', '.join(categories_to_use)} for similarity + AI Q&A.")


# --- Display results + insights ---
cards: List[FindingCard] = st.session_state.cards

if cards:
    st.subheader("Quick insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Results", len(cards))
    with c2:
        st.metric("Categories", "ALL" if category_choice == "all" else category_choice.upper())
    with c3:
        classifications = [c.classification for c in cards if c.classification]
        top_cls = pd.Series(classifications).value_counts().index[0] if classifications else "N/A"
        st.metric("Top classification", top_cls)

    latest_date, top_class, top_status = summarize_changes(cards)
    if latest_date or top_class or top_status:
        a, b, c = escape(latest_date), escape(top_class), escape(top_status)
        st.markdown(
            '<div class="recall-summary">'
            '<p class="recall-summary-title">Quick summary</p>'
            + (f'<p><span class="recall-summary-label">Latest recall date</span> <span class="recall-summary-value">{a}</span></p>' if latest_date else "")
            + (f'<p><span class="recall-summary-label">Most common classification</span> <span class="recall-summary-value">{b}</span></p>' if top_class else "")
            + (f'<p><span class="recall-summary-label">Most common status</span> <span class="recall-summary-value">{c}</span></p>' if top_status else "")
            + "</div>",
            unsafe_allow_html=True,
        )

    themes = top_themes(cards, n=6)
    if themes:
        st.write("**Common complaint themes (from recall reasons):**")
        st.write(", ".join([f"`{t}` ({n})" for t, n in themes]))

    st.divider()
    st.subheader("Results")

    df = pd.DataFrame(
        [
            {
                "category": c.category,
                "recall_number": c.recall_number,
                "product": c.product[:140],
                "firm": c.recalling_firm[:80],
                "classification": c.classification,
                "status": c.status,
                "init_date": c.recall_initiation_date,
                "reason": c.reason[:160],
            }
            for c in cards
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True, height=420)

    st.divider()
    st.subheader("Find similar recalls (semantic)")

    selected_id = st.selectbox(
        "Pick a recall to find similar ones",
        options=[c.id for c in cards],
        format_func=lambda cid: next(
            (
                f"[{c.category}] {c.product[:70]} – {c.classification} – {c.recall_initiation_date}"
                for c in cards
                if c.id == cid
            ),
            cid,
        ),
    )

    top_k = st.slider("Similar results", 3, 10, 5)
    find_btn = st.button("Find similar", type="primary")

    if find_btn and st.session_state.index is not None:
        model = load_embedder()
        selected_card = next(c for c in cards if c.id == selected_id)

        qvec = embed_texts(model, [selected_card.summary_text])  # (1, dim)
        scores, ids = faiss_search(st.session_state.index, qvec, top_k=top_k + 1)

        similar = []
        for score, idx in zip(scores, ids):
            if idx < 0:
                continue
            cand = cards[int(idx)]
            if cand.id == selected_card.id:
                continue
            similar.append((float(score), cand))
            if len(similar) >= top_k:
                break

        st.write("### Selected")
        st.code(selected_card.summary_text)

        st.write("### Similar recalls")
        if not similar:
            st.warning("No similar items found (try increasing recent records per category or keyword results per query).")
        else:
            for score, c in similar:
                with st.expander(
                    f"[{c.category}] {c.product[:80]}  |  {c.classification}  |  {c.recall_initiation_date}  (score: {score:.3f})"
                ):
                    st.code(c.summary_text)
                    st.caption(f"Firm: {c.recalling_firm} | Status: {c.status} | Recall #: {c.recall_number}")

    st.divider()
    st.subheader("Ask RECALL (AI Q&A)")

    user_question = st.text_input(
        "Ask a question about these recall results",
        placeholder="e.g., What are the main reasons? Any contamination patterns? Which products are most impacted?",
    )

    rag_k = st.slider("Number of recall records to use as context", 3, 12, 6)
    ask_btn = st.button("Answer with AI")

    if ask_btn and user_question.strip() and st.session_state.index is not None:
        retrieved_cards = []
        try:
            with st.spinner("Retrieving relevant recalls + generating answer..."):
                model = load_embedder()
                qvec = embed_texts(model, [user_question.strip()])  # (1, dim)
                scores, ids = faiss_search(st.session_state.index, qvec, top_k=rag_k)
                for idx in ids:
                    if idx >= 0:
                        retrieved_cards.append(cards[int(idx)])

                answer = answer_with_llm(user_question.strip(), retrieved_cards)

            st.markdown("### AI Answer")
            st.write(answer)

            with st.expander("Show context used by AI"):
                for c in retrieved_cards:
                    st.code(c.summary_text)

        except OSError as e:
            st.error(
                "**Could not load the AI model.** "
                "The model is downloaded from the internet on first use. "
                "Please check that you have internet access and try again. "
                "If you're behind a firewall or offline, download the model once with: "
                "`huggingface-cli download google/flan-t5-base` then set environment variable "
                "`RECALL_MODEL_PATH` to the download folder."
            )
            st.caption(f"Details: {e}")

else:
    st.caption("Run a search to load recall records and enable semantic similarity + AI Q&A.")
