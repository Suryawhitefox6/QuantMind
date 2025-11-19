#!/usr/bin/env python3
"""
Merged script: FAISS + MiniLM + T5 summarizer + Ollama reasoning
with BDH-inspired Hebbian chunk-strength updates (edge reweighting simulation).

Requirements:
 - sentence-transformers
 - transformers
 - faiss (faiss-cpu or faiss-gpu)
 - gradio
 - networkx (optional, only used for future graph extension; not required for this script)
 - requests, tqdm
"""

import os
import faiss
import pickle
import numpy as np
import gradio as gr
import requests
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import time

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "./data"
EMBED_MODEL_PATH = r"C:\Users\Shobith\Downloads\all-MiniLM-L6-v2"  # 384-dim
DB_FAISS_PATH = "./finance_index.faiss"
DB_META_PATH = "./finance_meta.pkl"       # will store: {"chunks": [...], "summaries": {...}, "embeddings": np.array([...])}
MEMORY_PATH = "memory.pkl"                # stores memory {"queries":[],"answers":[],"embeddings":[...]}
HEBB_PATH = "hebbian.pkl"                 # stores chunk_strength dict + metadata
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3

# Hebbian hyperparams
HEBB_LR = 0.05           # learning rate for hebbian updates
HEBB_DECAY = 0.985       # multiplicative decay applied each hebbian update pass (close to 1)
ALPHA = 0.8              # weight for semantic similarity in retrieval score
BETA = 1.2               # weight for hebbian strength in retrieval score
MAX_HEBB = 5.0           # clamp on max hebbian strength to avoid runaway

# embedding model dims (miniLM small)
EMBED_DIM = 384

# -----------------------------
# UTILS: load & chunk text
# -----------------------------
def load_texts(data_dir):
    docs = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8", errors="ignore") as f:
                docs.append(f.read())
    return docs

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -----------------------------
# BUILD FAISS INDEX (and save chunk embeddings)
# -----------------------------
def build_faiss_index():
    print("📚 Loading documents...")
    docs = load_texts(DATA_DIR)
    print(f"Loaded {len(docs)} documents")

    print("✂ Chunking text...")
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))
    print(f"Total chunks: {len(chunks)}")

    print("🔢 Generating embeddings...")
    model = SentenceTransformer(EMBED_MODEL_PATH)
    embeddings = []
    for chunk in tqdm(chunks, desc="Embedding chunks"):
        emb = model.encode(chunk)
        embeddings.append(emb)
    embeddings = np.array(embeddings, dtype="float32")

    print("💾 Saving FAISS index...")
    dim = embeddings.shape[1]
    # We'll use L2 index but normalize embeddings so similarity ~ inner product if desired.
    # For simplicity we'll store as flat L2 index but also save embeddings for cosine/dot usage.
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, DB_FAISS_PATH)

    # Save metadata (chunks, summaries, embeddings)
    meta = {"chunks": chunks, "summaries": {}, "embeddings": embeddings}
    with open(DB_META_PATH, "wb") as f:
        pickle.dump(meta, f)

    # Initialize memory and hebbian store if not present
    if not os.path.exists(MEMORY_PATH):
        pickle.dump({"queries": [], "answers": [], "embeddings": []}, open(MEMORY_PATH, "wb"))

    if not os.path.exists(HEBB_PATH):
        # chunk_strength: dict chunk_idx -> float
        hebb = {"chunk_strength": {}, "last_update": time.time()}
        pickle.dump(hebb, open(HEBB_PATH, "wb"))

    print("✅ FAISS index built successfully.")

# -----------------------------
# SUMMARIZER (T5-small)
# -----------------------------
print("🧠 Loading summarizer (T5-small)...")
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

def summarize_text(text):
    text = text.strip()
    if len(text) == 0:
        return ""
    if len(text) < 120:
        return text
    try:
        # keep conservative lengths
        out = summarizer(text, max_length=80, min_length=20, do_sample=False)
        return out[0]["summary_text"]
    except Exception as e:
        return f"[Summary Error] {e}"

# -----------------------------
# HEBBIAN (chunk-strength) helpers
# -----------------------------
def load_hebbian():
    if os.path.exists(HEBB_PATH):
        return pickle.load(open(HEBB_PATH, "rb"))
    else:
        hebb = {"chunk_strength": {}, "last_update": time.time()}
        pickle.dump(hebb, open(HEBB_PATH, "wb"))
        return hebb

def save_hebbian(hebb):
    hebb["last_update"] = time.time()
    pickle.dump(hebb, open(HEBB_PATH, "wb"))

def hebbian_decay_step(hebb, decay_factor=HEBB_DECAY):
    # apply multiplicative decay to all strengths
    for k in list(hebb["chunk_strength"].keys()):
        hebb["chunk_strength"][k] *= decay_factor

# -----------------------------
# RETRIEVAL: combine sim + hebb
# -----------------------------
def retrieve_context(query, top_k=TOP_K, alpha=ALPHA, beta=BETA):
    model = SentenceTransformer(EMBED_MODEL_PATH)
    q_emb = model.encode(query).astype("float32")

    # load index and metadata
    if not os.path.exists(DB_FAISS_PATH) or not os.path.exists(DB_META_PATH):
        raise RuntimeError("FAISS index or metadata missing. Run build_faiss_index() first.")

    index = faiss.read_index(DB_FAISS_PATH)
    with open(DB_META_PATH, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    summaries = data.get("summaries", {})
    chunk_embeddings = data["embeddings"]  # numpy array of shape (n_chunks, dim)

    # ensure q_emb shape
    q_emb_arr = np.array([q_emb], dtype="float32")

    # L2 distances (lower is better). We'll convert to a similarity score in [0,1]
    D, I = index.search(q_emb_arr, min(len(chunks), max(top_k*5, top_k*3)))  # retrieve more, then re-rank with hebb
    retrieved_idxs = I[0].tolist()

    # compute similarity score (cosine) between q and chunk embeddings for better semantics
    # normalize embeddings to compute cosine similarity
    # avoid division by zero
    def cosine(a, b):
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an == 0 or bn == 0:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    # load hebbian map
    hebb = load_hebbian()
    chunk_strength = hebb.get("chunk_strength", {})

    scores = []
    for idx in retrieved_idxs:
        emb = chunk_embeddings[idx]
        sim = cosine(q_emb, emb)  # in [-1,1]
        sim_scaled = max(0.0, (sim + 1) / 2.0)  # map to [0,1]
        hebb_score = float(chunk_strength.get(int(idx), 0.0))
        combined = alpha * sim_scaled + beta * hebb_score
        scores.append((idx, combined, sim_scaled, hebb_score))

    # sort by combined score, pick top_k
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    selected_idxs = [s[0] for s in scores_sorted]

    # build doc_context and summary_context
    doc_context = "\n\n".join([chunks[i] for i in selected_idxs])
    summary_context = ""
    changed = False
    for i in selected_idxs:
        if i not in summaries:
            summaries[i] = summarize_text(chunks[i])
            changed = True
        summary_context += summaries[i] + "\n"

    if changed:
        # persist updated summaries
        with open(DB_META_PATH, "wb") as f:
            pickle.dump({"chunks": chunks, "summaries": summaries, "embeddings": chunk_embeddings}, f)

    # memory recall: include past memory QA pairs if present (simple concatenation)
    mem_context = ""
    if os.path.exists(MEMORY_PATH):
        memory = pickle.load(open(MEMORY_PATH, "rb"))
        mem_queries = memory.get("queries", [])
        mem_answers = memory.get("answers", [])
        # include last N memory entries (N = TOP_K)
        for q, a in zip(mem_queries[-top_k:], mem_answers[-top_k:]):
            mem_context += f"Previous Q: {q}\nA: {a}\n\n"

    combined_context = f"{doc_context}\n\nLearned Insights:\n{mem_context}"
    return combined_context, summary_context, selected_idxs, q_emb

# -----------------------------
# OLLAMA REASONING (streaming)
# -----------------------------
def query_ollama(prompt):
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=600) as response:
            response.raise_for_status()
            output = ""
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        output += data["response"]
                except json.JSONDecodeError:
                    continue
            return output.strip()
    except requests.exceptions.Timeout:
        return "[Ollama Error] Request timed out. Try reducing context or check model size."
    except Exception as e:
        return f"[Ollama Error] {e}"

# -----------------------------
# ANSWERING FUNCTIONS (with hebbian update)
# -----------------------------
def save_to_memory_and_hebbian(query, answer, used_chunk_idxs, q_emb):
    """
    Saves QA pair to memory and updates Hebbian chunk strengths for used chunks.
    We strengthen each used chunk proportional to dot(q_emb, chunk_emb) and clamp/decay globally.
    """
    # load sentence transformer for embeddings
    model = SentenceTransformer(EMBED_MODEL_PATH)
    a_emb = model.encode(answer).astype("float32")

    # save to memory file
    if os.path.exists(MEMORY_PATH):
        memory = pickle.load(open(MEMORY_PATH, "rb"))
    else:
        memory = {"queries": [], "answers": [], "embeddings": []}

    memory["queries"].append(query)
    memory["answers"].append(answer)
    memory["embeddings"].append(a_emb.tolist())
    pickle.dump(memory, open(MEMORY_PATH, "wb"))

    # update hebbian strengths
    if os.path.exists(DB_META_PATH):
        meta = pickle.load(open(DB_META_PATH, "rb"))
        chunk_embeddings = meta["embeddings"]
    else:
        chunk_embeddings = None

    hebb = load_hebbian()

    # apply small decay step first (global)
    hebbian_decay_step(hebb, decay_factor=HEBB_DECAY)

    if chunk_embeddings is not None and used_chunk_idxs:
        qv = np.array(q_emb, dtype="float32")
        for idx in used_chunk_idxs:
            # compute dot as simple co-activation metric (can also use cosine)
            emb = chunk_embeddings[int(idx)]
            dot = float(np.dot(qv, emb))
            # scale to moderate magnitude
            delta = HEBB_LR * (dot / (np.linalg.norm(qv) * (np.linalg.norm(emb) + 1e-8) + 1e-12))
            # ensure non-negative increment
            delta = max(0.0, delta)
            prev = float(hebb["chunk_strength"].get(int(idx), 0.0))
            new = min(MAX_HEBB, prev + delta)
            hebb["chunk_strength"][int(idx)] = new

    save_hebbian(hebb)
    print("🧩 Memory and Hebbian strengths updated!")

def answer_query(user_query):
    context, summary_context, used_chunks, q_emb = retrieve_context(user_query)
    prompt = f"""
You are an expert financial AI assistant.
Use the context below to answer the question concisely and accurately.

Context:
{context}

Summaries:
{summary_context}

Question: {user_query}

Answer precisely:
"""
    answer = query_ollama(prompt)
    # save to memory and update hebbian based on used chunks
    save_to_memory_and_hebbian(user_query, answer, used_chunks, q_emb)
    return answer

def analyze_and_reason(user_query):
    context, summary_context, used_chunks, q_emb = retrieve_context(user_query, top_k=TOP_K*2)
    prompt = f"""
You are a highly analytical financial expert.
Analyze the data and provide deep insights, not just retrieval-based answers.

Raw Context:
{context}

Summaries:
{summary_context}

Question: {user_query}

Provide a structured, reasoned, and insightful response.
"""
    answer = query_ollama(prompt)
    save_to_memory_and_hebbian(user_query, answer, used_chunks, q_emb)
    return answer

# -----------------------------
# GRADIO UI
# -----------------------------
def launch_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 💹 QuantMind (BDH-style Hebbian retrieval)")
        gr.Markdown("MiniLM for retrieval embeddings, T5-small for summarization, Mistral (via Ollama) for reasoning. "
                    "BDH-inspired fast-weight (hebbian) updates boost chunks that helped answers.")

        with gr.Tab("Ask (Normal Mode)"):
            query_input = gr.Textbox(label="Enter your question", placeholder="e.g. What was the market trend in 2021?")
            query_btn = gr.Button("Ask")
            query_output = gr.Textbox(label="Answer", lines=12)
            query_btn.click(answer_query, inputs=query_input, outputs=query_output)

        with gr.Tab("Analytical Mode"):
            analyze_input = gr.Textbox(label="Enter analytical query", placeholder="e.g. Which sector shows consistent growth and why?")
            analyze_btn = gr.Button("Analyze")
            analyze_output = gr.Textbox(label="Analytical Answer", lines=16)
            analyze_btn.click(analyze_and_reason, inputs=analyze_input, outputs=analyze_output)

        with gr.Tab("Admin"):
            build_btn = gr.Button("(Re)Build FAISS Index from ./data")
            build_txt = gr.Textbox(label="Status", lines=4)
            def _build():
                try:
                    build_faiss_index()
                    return "Index built successfully."
                except Exception as e:
                    return f"Build error: {e}"
            build_btn.click(_build, inputs=None, outputs=build_txt)

            hebb_btn = gr.Button("View Hebbian Top Chunks")
            hebb_out = gr.Textbox(label="Hebbian info", lines=12)
            def _view_hebb():
                if not os.path.exists(HEBB_PATH) or not os.path.exists(DB_META_PATH):
                    return "No hebbian or meta data found."
                hebb = load_hebbian()
                meta = pickle.load(open(DB_META_PATH, "rb"))
                chunks = meta["chunks"]
                cs = hebb.get("chunk_strength", {})
                top = sorted(cs.items(), key=lambda x: x[1], reverse=True)[:10]
                lines = []
                for idx, val in top:
                    lines.append(f"idx:{idx} score:{val:.4f} chunk_preview: {chunks[int(idx)][:120].replace('/n',' ')}")
                return "\n".join(lines) if lines else "Hebbian store empty."
            hebb_btn.click(_view_hebb, inputs=None, outputs=hebb_out)

    demo.launch(server_name="localhost", server_port=7860)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # build index automatically if missing
    if not os.path.exists(DB_FAISS_PATH) or not os.path.exists(DB_META_PATH):
        build_faiss_index()
    launch_ui()
