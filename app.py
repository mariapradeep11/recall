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
import faiss # faiss-cpu
except Exception:
faiss = None


# -----------------------------
# Config
# -----------------------------
OPENFDA_BASE = "https://api.fda.gov"
DEFAULT_LIMIT = 25 # keep small for demo speed
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOGO_PATH = Path(__file__).parent / "assets" / "RECALL.png"


# -----------------------------
# Data model
# -----------------------------
@dataclass
class FindingCard:
id: str
category: str # drug/food/device
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
index = faiss.IndexFlatIP(dim) # cosine similarity if vectors are normalized
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
f" Classification: {c.classification}\n"
f" Reason: {c.reason}\n"
f" Firm: {c.recalling_firm}\n"
f" Date: {c.recall_initiation_date}\n"
f" Status: {c.status}\n"
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

# RECALL theme: accent #F28C6C, background #313645, text #FFFFFF
st.markdown(
"""
<style>
/* RECALL brand colors */
.stApp { background-color: #1a1f28; }
h1, h2, h3 { color: #ffffff !important; font-weight: 700; }
.stMetric label { color: #b0b8c4 !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; }
div[data-testid="stSidebar"] { background: linear-gradient(180deg, #313645 0%, #252a33 100%); }
div[data-testid="stSidebar"] .stMarkdown { color: #ffffff; }
.stButton > button { background-color: #F28C6C !important; color: #ffffff !important; font-weight: 600; border: none; }
.stButton > button:hover { background-color: #e07a5a !important; color: #ffffff !important; }
hr { border-color: #F28C6C33 !important; }
.recall-summary {
font-family: sans-serif;
background: linear-gradient(135deg, #252a33 0%, #313645 100%);
border-left: 4px solid #F28C6C;
padding: 1rem 1.25rem;
border-radius: 0 8px 8px 0;
margin: 0.5rem 0;
color: #FFFFFF;
}
.recall-summary-title {
font-weight: 700;
font-size: 0.95rem;
color: #F28C6C;
margin-bottom: 0.6rem;
}
.recall-summary p {
margin: 0.35rem 0;
font-size: 0.95rem;
}
.recall-summary-label {
color: #b0b8c4;
margin-right: 0.35rem;
}
.recall-summary-value {
color: #FFFFFF;
font-weight: 600;
}
</style>
""",
unsafe_allow_html=True,
)

with st.sidebar:
if LOGO_PATH.exists():
st.image(str(LOGO_PATH), use_container_width=True)
st.header("Search Settings")
category = st.selectbox("Category", ["drug", "food", "device"], index=0)
limit = st.slider("Max results", min_value=5, max_value=50, value=DEFAULT_LIMIT, step=5)
api_key = st.text_input("openFDA API key (optional)", type="password", value=os.getenv("OPENFDA_API_KEY", ""))
st.caption("Tip: API key helps rate limits, but this prototype works without it.")

st.markdown("## RECALL")
st.write(
"Type a **brand/product keyword** to see recalls. Then use:\n"
"- **Find similar recalls** (semantic search)\n"
"- **Ask RECALL** (LLM Q&A over retrieved results)\n"
)

col1, col2 = st.columns([2, 1], gap="large")
with col1:
query = st.text_input("Search term", placeholder="e.g., Tylenol, Ozempic, lettuce, Philips")
with col2:
run_btn = st.button("Search openFDA", use_container_width=True)

if "cards" not in st.session_state:
st.session_state.cards = []
if "vectors" not in st.session_state:
st.session_state.vectors = None
if "index" not in st.session_state:
st.session_state.index = None

if run_btn and query.strip():
with st.spinner("Fetching recalls from openFDA..."):
try:
results = openfda_enforcement_search(category, query.strip(), limit=limit, api_key=api_key or None)
except requests.HTTPError as e:
st.error(f"openFDA request failed: {e}")
results = []
except Exception as e:
st.error(f"Unexpected error: {e}")
results = []

cards = normalize_to_cards(category, results)
st.session_state.cards = cards

if not cards:
st.warning("No recalls found for that query. Try a broader keyword (e.g., 'salad', 'insulin', 'cheese').")
else:
with st.spinner("Building local embeddings + FAISS index..."):
model = load_embedder()
texts = [c.summary_text for c in cards]
vecs = embed_texts(model, texts)
st.session_state.vectors = vecs
st.session_state.index = build_faiss_index(vecs)

st.success(f"Loaded {len(cards)} recall records for semantic search and Q&A.")

# --- Display results + insights ---
cards: List[FindingCard] = st.session_state.cards

if cards:
st.subheader("Quick insights")
c1, c2, c3 = st.columns(3)
with c1:
st.metric("Results", len(cards))
with c2:
st.metric("Category", category.upper())
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
+ (f'<p><span class="recall-summary-label">Latest recall date</span> <span class="recall-summary-value">{a}</span></p>' if latest_date else '')
+ (f'<p><span class="recall-summary-label">Most common classification</span> <span class="recall-summary-value">{b}</span></p>' if top_class else '')
+ (f'<p><span class="recall-summary-label">Most common status</span> <span class="recall-summary-value">{c}</span></p>' if top_status else '')
+ '</div>',
unsafe_allow_html=True,
)
else:
st.caption("No high-level summary available for this query.")

themes = top_themes(cards, n=6)
if themes:
st.write("**Common complaint themes (from recall reasons):**")
st.write(", ".join([f"`{t}` ({n})" for t, n in themes]))

st.divider()
st.subheader("Results")
df = pd.DataFrame(
[
{
"id": c.id,
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
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Find similar recalls (semantic)")

selected_id = st.selectbox(
"Pick a recall to find similar ones",
options=[c.id for c in cards],
format_func=lambda cid: next(
(f"{c.product[:80]} – {c.classification} – {c.recall_initiation_date}" for c in cards if c.id == cid),
cid,
),
)

top_k = st.slider("Similar results", 3, 10, 5)
find_btn = st.button("Find similar", type="primary")

if find_btn and st.session_state.index is not None:
model = load_embedder()
selected_card = next(c for c in cards if c.id == selected_id)

qvec = embed_texts(model, [selected_card.summary_text]) # (1, dim)
scores, ids = faiss_search(st.session_state.index, qvec, top_k=top_k + 1)

# Filter out itself
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
st.warning("No similar items found (try increasing max results or broaden the search term).")
else:
for score, c in similar:
with st.expander(
f"{c.product[:90]} | {c.classification} | {c.recall_initiation_date} (score: {score:.3f})"
):
st.code(c.summary_text)
st.caption(f"Firm: {c.recalling_firm} | Status: {c.status}")

st.divider()
st.subheader("Ask RECALL (AI Q&A)")

user_question = st.text_input(
"Ask a question about these recall results",
placeholder="e.g., What are the main reasons for these recalls? Any patterns?",
)

rag_k = st.slider("Number of recall records to use as context", 3, 12, 6)
ask_btn = st.button("Answer with AI")

if ask_btn and user_question.strip() and st.session_state.index is not None:
retrieved_cards = []
try:
with st.spinner("Retrieving relevant recalls + generating answer..."):
model = load_embedder()
qvec = embed_texts(model, [user_question.strip()]) # (1, dim)
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
"`RECALL_MODEL_PATH` to the download folder (e.g. `~/cache/hub/models--google--flan-t5-base/snapshots/...`)."
)
st.caption(f"Details: {e}")

else:
st.caption("Run a search to load recall records and enable semantic similarity + AI Q&A.")