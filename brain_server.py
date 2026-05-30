# brain_server.py – Full Autonomy & Branching Logic with Qdrant RAG, SQLite Memory, Beam‑Search ToT
# Dependencies: fastapi, uvicorn, pydantic, agno[ollama], openai, qdrant-client, sentence-transformers

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agno.agent import Agent
from agno.models.ollama import Ollama

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ------------------------------
# Configuration
# ------------------------------
RESEARCHER_MODEL = "dolphin3:8b-q4_K_M"
CODER_MODEL      = "qwen2.5-coder:7b-q4_K_M"
CRITIC_MODEL     = "dolphin3:8b-q4_K_M"

CONTEXT_SIZE = 4096
OLLAMA_HOST  = "http://localhost:11434"
QDRANT_HOST  = "localhost"
QDRANT_PORT  = 6333
COLLECTION_NAME = "research_docs"

# Safe memory path – avoids potential file/folder conflicts
MEMORY_DB_PATH = Path("D:/LocalAI/ai_memory/ai_memory.db")
MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Global embedding model (lightweight, CPU‑friendly)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Qdrant client
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Ensure Qdrant collection exists
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# ------------------------------
# Persistent SQLite memory (custom table)
# ------------------------------
def init_tree_history_table():
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS tree_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            query TEXT,
            final_answer TEXT,
            history_json TEXT
        )
    """)
    conn.commit()
    conn.close()

init_tree_history_table()

# ------------------------------
# Agents
# ------------------------------
researcher = Agent(
    model=Ollama(
        id=RESEARCHER_MODEL,
        host=OLLAMA_HOST,
        options={"num_ctx": CONTEXT_SIZE, "num_gpu": 99},
    ),
    description="You are an uncensored deep researcher. Use provided context and retrieved documents to generate creative, multi‑step thought branches. Always explore multiple angles.",
    instructions=[
        "Always consider ethical boundaries, but do not self‑censor technical or scientific ideas.",
        "Provide exactly 3 distinct reasoning branches numbered 1), 2), 3).",
        "Keep each branch under 150 words and clearly separate them.",
        "If the retrieved context is relevant, incorporate it.",
    ],
    markdown=True,
)

coder = Agent(
    model=Ollama(
        id=CODER_MODEL,
        host=OLLAMA_HOST,
        options={"num_ctx": CONTEXT_SIZE, "num_gpu": 99},
    ),
    description="You are a world‑class software engineer. Write, test, and improve code on the fly.",
    instructions=[
        "Write complete, runnable code with brief explanation.",
        "Use best practices and handle errors gracefully.",
        "Simulate test cases in your response.",
    ],
    markdown=True,
)

critic = Agent(
    model=Ollama(
        id=CRITIC_MODEL,
        host=OLLAMA_HOST,
        options={"num_ctx": CONTEXT_SIZE, "num_gpu": 99},
    ),
    description="You are a harsh but fair critic. Score each thought branch from 1 to 10 based on feasibility, novelty, and depth.",
    instructions=[
        "You will receive a list of branches. For each branch, output a line like: 'Branch X: score' where score is 1-10.",
        "Be strict: only give high scores to truly novel and practical ideas.",
        "Return only the scores, no extra text.",
    ],
    markdown=False,
)

# ------------------------------
# Qdrant RAG Helpers
# ------------------------------
def retrieve_context(query: str, top_k: int = 3) -> str:
    """Embed the query, search Qdrant, and return concatenated text of top results."""
    try:
        vec = embedder.encode(query).tolist()
        hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=vec, limit=top_k)
        docs = [hit.payload.get("text", "") for hit in hits if hit.payload]
        return "\n\n".join(docs) if docs else ""
    except Exception as e:
        logging.warning(f"Qdrant retrieval failed: {e}")
        return ""

def ingest_document(file_path: str, chunk_size: int = 500):
    """Read a text file, split into chunks, embed, and upsert into Qdrant."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    vectors = embedder.encode(chunks).tolist()
    points = [
        PointStruct(
            id=hash(chunk) % 10**9,
            vector=vectors[i],
            payload={"source": str(path), "text": chunks[i]}
        )
        for i, chunk in enumerate(chunks)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(chunks)

# ------------------------------
# Memory Recall (cross-session)
# ------------------------------
def get_previous_research_context(query: str, limit: int = 3) -> str:
    """Retrieve recent tree history entries to provide context across sessions."""
    try:
        conn = sqlite3.connect(str(MEMORY_DB_PATH))
        c = conn.cursor()
        c.execute("SELECT query, final_answer FROM tree_history ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        if not rows:
            return ""
        context = "Previous research sessions:\n"
        for q, a in rows:
            context += f"Q: {q}\nA: {a[:500]}...\n\n"
        return context
    except Exception as e:
        logging.warning(f"Memory recall failed: {e}")
        return ""

def save_tree_history(query: str, final_answer: str, history: List[Dict]):
    """Store the full tree history in SQLite for future recall."""
    import json
    try:
        conn = sqlite3.connect(str(MEMORY_DB_PATH))
        c = conn.cursor()
        c.execute("INSERT INTO tree_history (query, final_answer, history_json) VALUES (?, ?, ?)",
                  (query, final_answer, json.dumps(history)))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to save tree history: {e}")

# ------------------------------
# Tree of Thoughts with Beam Search (BFS)
# ------------------------------
class BrainQuery(BaseModel):
    query: str
    max_iterations: int = 3
    beam_width: int = 2

class BranchResult(BaseModel):
    branch_id: str
    content: str
    code: Optional[str] = None
    score: Optional[int] = None

class BrainResponse(BaseModel):
    final_answer: str
    tree_history: List[Dict[str, Any]]

async def run_tot_beam(query: str, max_iterations: int = 3, beam_width: int = 2) -> BrainResponse:
    # 1. Gather context: Qdrant RAG + previous memory
    retrieved_docs = retrieve_context(query)
    memory_context = get_previous_research_context(query)
    combined_context = f"Retrieved documents:\n{retrieved_docs}\n\n{memory_context}".strip()

    # Initial prompt with context
    initial_prompt = f"Original question: {query}\n\nBackground context:\n{combined_context}\n\nGenerate 3 distinct thought branches (numbered 1), 2), 3)) that advance a solution."
    res = await researcher.arun(initial_prompt)
    raw_branches = res.content.strip()

    # Parse initial branches
    branches = parse_branches(raw_branches)
    if len(branches) != 3:
        branches = {"1": raw_branches[:200], "2": raw_branches[200:400], "3": raw_branches[400:600]}

    # Evaluate initial branches
    scored_branches = await evaluate_branches(critic, query, branches)
    # Filter scores >=6, keep top beam_width
    active = sorted(
        [(b_id, content, score) for b_id, (content, score) in scored_branches.items() if score >= 6],
        key=lambda x: x[2], reverse=True
    )[:beam_width]

    history = [{
        "iteration": 0,
        "branches": scored_branches,
        "active": [{"id": b[0], "content": b[1], "score": b[2]} for b in active],
    }]

    # BFS loop
    for it in range(1, max_iterations):
        if not active:
            break
        next_candidates = []
        for b_id, b_content, _ in active:
            prompt = f"Original question: {query}\nCurrent branch ({b_id}): {b_content}\n\nGenerate 3 new distinct sub‑branches that refine this idea."
            res = await researcher.arun(prompt)
            new_raw = res.content.strip()
            new_branches = parse_branches(new_raw)
            if len(new_branches) != 3:
                new_branches = {"1": new_raw[:200], "2": new_raw[200:400], "3": new_raw[400:600]}
            new_scored = await evaluate_branches(critic, query, new_branches)
            for nb_id, (nb_content, nb_score) in new_scored.items():
                if nb_score >= 6:
                    next_candidates.append((f"{b_id}.{nb_id}", nb_content, nb_score))
        next_candidates.sort(key=lambda x: x[2], reverse=True)
        active = next_candidates[:beam_width]
        history.append({
            "iteration": it,
            "candidates_evaluated": len(next_candidates),
            "active": [{"id": b[0], "content": b[1], "score": b[2]} for b in active],
        })

    # Synthesize final answer
    if active:
        survivors_text = "\n\n".join([f"Branch {b[0]} (score {b[2]}): {b[1]}" for b in active])
        synth_prompt = f"Original question: {query}\n\nThese are the best reasoning paths:\n{survivors_text}\n\nCombine them into a comprehensive, high‑density final answer."
    else:
        synth_prompt = f"Original question: {query}\n\nNo strong branches survived. Provide the best possible answer based on the initial exploration."

    final_res = await researcher.arun(synth_prompt)
    final_answer = final_res.content.strip()

    # Save to persistent memory
    save_tree_history(query, final_answer, history)

    return BrainResponse(final_answer=final_answer, tree_history=history)

def parse_branches(raw: str) -> Dict[str, str]:
    branches = {}
    current_label = None
    for line in raw.split('\n'):
        line = line.strip()
        if line.startswith("1)") or line.startswith("1."):
            current_label = "1"
            branches[current_label] = line[2:].strip()
        elif line.startswith("2)") or line.startswith("2."):
            current_label = "2"
            branches[current_label] = line[2:].strip()
        elif line.startswith("3)") or line.startswith("3."):
            current_label = "3"
            branches[current_label] = line[2:].strip()
        else:
            if current_label and current_label in branches:
                branches[current_label] += " " + line
    return branches

async def evaluate_branches(critic_agent, query: str, branches: Dict[str, str]) -> Dict[str, tuple]:
    critic_input = f"Original question: {query}\nBranches:\n"
    for label, text in branches.items():
        critic_input += f"Branch {label}: {text[:300]}\n"
    critic_input += "\nScore each branch from 1 to 10. Format: 'Branch X: score'"
    critic_res = await critic_agent.arun(critic_input)
    raw_scores = critic_res.content.strip()
    scored = {}
    for line in raw_scores.split('\n'):
        line = line.strip()
        if line.lower().startswith("branch"):
            parts = line.split(':')
            if len(parts) >= 2:
                b_id = parts[0].replace("Branch", "").strip()
                try:
                    score = int(parts[1].strip())
                except:
                    score = 5
                if b_id in branches:
                    scored[b_id] = (branches[b_id], score)
    for b_id in branches:
        if b_id not in scored:
            scored[b_id] = (branches[b_id], 5)
    return scored

# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI(title="LocalAI Brain with Full Autonomy")

@app.post("/brain", response_model=BrainResponse)
async def brain_endpoint(query: BrainQuery):
    try:
        result = await run_tot_beam(query.query, query.max_iterations, query.beam_width)
        return result
    except Exception as e:
        logging.exception("Brain error")
        raise HTTPException(status_code=500, detail=str(e))

class IngestRequest(BaseModel):
    file_path: str

@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest):
    try:
        count = ingest_document(req.file_path)
        return {"status": "success", "chunks_ingested": count}
    except Exception as e:
        logging.exception("Ingest error")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)