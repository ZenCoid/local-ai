# brain_server.py — Autonomous Agent System with Manual Web Search, File/Shell Control
# Dependencies: fastapi, uvicorn, pydantic, agno[ollama], qdrant-client, sentence-transformers, duckduckgo_search

import asyncio
import logging
import sqlite3
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agno.agent import Agent
from agno.models.ollama import Ollama

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS

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

WORK_DIR = Path("D:/LocalAI/PythonProject")
WORK_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_DB_PATH = Path("D:/LocalAI/ai_memory/ai_memory.db")
MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

file_lock = asyncio.Lock()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# ------------------------------
# SQLite Memory
# ------------------------------
def init_tree_history_table():
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.execute("""
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

def get_previous_research_context(query: str, limit: int = 3) -> str:
    try:
        conn = sqlite3.connect(str(MEMORY_DB_PATH))
        rows = conn.execute(
            "SELECT query, final_answer FROM tree_history ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        if not rows:
            return ""
        ctx = "Previous research sessions:\n"
        for q, a in rows:
            ctx += f"Q: {q}\nA: {a[:500]}...\n\n"
        return ctx
    except Exception as e:
        logging.warning(f"Memory recall failed: {e}")
        return ""

def save_tree_history(query, final_answer, history):
    import json
    try:
        conn = sqlite3.connect(str(MEMORY_DB_PATH))
        conn.execute(
            "INSERT INTO tree_history (query, final_answer, history_json) VALUES (?, ?, ?)",
            (query, final_answer, json.dumps(history))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to save history: {e}")

# ------------------------------
# Qdrant RAG & Ingestion
# ------------------------------
def retrieve_context(query: str, top_k: int = 3) -> str:
    try:
        vec = embedder.encode(query).tolist()
        hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=vec, limit=top_k)
        docs = [hit.payload.get("text", "") for hit in hits if hit.payload]
        return "\n\n".join(docs) if docs else ""
    except Exception as e:
        logging.warning(f"Qdrant retrieval failed: {e}")
        return ""

def ingest_document(file_path: str, chunk_size: int = 500):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    ingest_raw_text(text, source=str(path), chunk_size=chunk_size)

def ingest_raw_text(text: str, source: str = "direct_input", chunk_size: int = 500):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    vectors = embedder.encode(chunks).tolist()
    points = [
        PointStruct(
            id=hash(chunk) % 10**9,
            vector=vectors[i],
            payload={"source": source, "text": chunks[i]}
        )
        for i, chunk in enumerate(chunks)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(chunks)

# ------------------------------
# Manual tool helpers (no model tool support needed)
# ------------------------------
def web_search(query: str, max_results: int = 3) -> str:
    """Perform a DuckDuckGo search and return concatenated snippets."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No search results found."
        return "\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        logging.error(f"Web search failed: {e}")
        return f"Web search error: {e}"

def write_file(filename: str, content: str):
    """Write content to a file inside WORK_DIR."""
    path = WORK_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return str(path)

def run_shell(command: str) -> str:
    """Run a shell command and return stdout+stderr."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)

# ------------------------------
# Agents (no tools – purely text‑based)
# ------------------------------
researcher = Agent(
    model=Ollama(
        id=RESEARCHER_MODEL,
        host=OLLAMA_HOST,
        options={"num_ctx": CONTEXT_SIZE, "num_gpu": 99},
    ),
    description="You are a deep researcher. You will be given search results and must synthesise an answer.",
    instructions=[
        "Read the provided web search results carefully.",
        "Provide a concise, accurate answer based ONLY on the provided information.",
        "Always cite the source title when using a result.",
    ],
    markdown=True,
)

coder = Agent(
    model=Ollama(
        id=CODER_MODEL,
        host=OLLAMA_HOST,
        options={"num_ctx": CONTEXT_SIZE, "num_gpu": 99},
    ),
    description="You are a senior software engineer. You will be asked to write code; return the full script and a brief explanation.",
    instructions=[
        "Always write complete, runnable Python code.",
        "Output ONLY the code and a short explanation, nothing else.",
        "Use best practices and handle errors gracefully.",
    ],
    markdown=True,
)

critic = Agent(
    model=Ollama(
        id=CRITIC_MODEL,
        host=OLLAMA_HOST,
        options={"num_ctx": CONTEXT_SIZE, "num_gpu": 99},
    ),
    description="You are a critic. Score the given progress from 1 to 10.",
    instructions=[
        "Read the project progress and provide a single score (1-10) and a brief justification.",
        "Format: 'Score: X/10. Justification: ...'",
    ],
    markdown=False,
)

planner = Agent(
    model=Ollama(
        id=RESEARCHER_MODEL,
        host=OLLAMA_HOST,
        options={"num_ctx": CONTEXT_SIZE, "num_gpu": 99},
    ),
    description="You are a planner. Break a high‑level goal into a JSON list of tasks.",
    instructions=[
        "Output ONLY a valid JSON array of task objects. Each task has:",
        "  - 'id': integer",
        "  - 'type': one of 'research', 'code', 'critique'",
        "  - 'description': string",
        "  - 'depends_on': list of task ids (or empty)",
        "Example: [{'id':1, 'type':'research', 'description':'Find latest...', 'depends_on':[]}]",
    ],
    markdown=False,
)

# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI(title="LocalAI Autonomous Brain")

class ProjectRequest(BaseModel):
    goal: str
    max_tasks: int = 5

class ProjectResponse(BaseModel):
    tasks_completed: List[Dict]
    final_summary: str

class IngestRequest(BaseModel):
    file_path: str

@app.post("/run_project", response_model=ProjectResponse)
async def run_project(req: ProjectRequest):
    # 1. Planner
    plan_raw = await planner.arun(f"Goal: {req.goal}")
    import json
    try:
        tasks = json.loads(plan_raw.content.strip().replace("'", '"'))
    except Exception:
        tasks = [
            {"id": 1, "type": "research", "description": f"Research about {req.goal}", "depends_on": []}
        ]
    if len(tasks) > req.max_tasks:
        tasks = tasks[:req.max_tasks]

    completed = []
    final_context = ""
    # 2. Execute tasks with manual tools
    for task in tasks:
        if task["type"] == "research":
            # Manual web search
            query = task['description']
            search_results = web_search(query, max_results=3)
            # Feed search results to researcher agent
            prompt = f"Task: {query}\n\nSearch Results:\n{search_results}\n\nSynthesise a concise answer."
            res = await researcher.arun(prompt)
            answer = res.content.strip()
            # Auto‑ingest the research result
            try:
                ingest_raw_text(f"Research result for '{query}':\n{answer}", source="web_research")
                logging.info("Auto‑ingested research result")
            except Exception as e:
                logging.warning(f"Auto‑ingestion failed: {e}")
            completed.append({"task_id": task["id"], "result": answer})
            final_context += f"\n{answer}"

        elif task["type"] == "code":
            async with file_lock:
                prompt = f"Task: {task['description']}\nWrite the Python code. Output ONLY the code and a short explanation."
                res = await coder.arun(prompt)
                answer = res.content.strip()
                # Attempt to extract code block and write to file
                code = answer
                if "```" in answer:
                    parts = answer.split("```")
                    if len(parts) >= 2:
                        code = parts[1].replace("python", "").strip()
                # Generate filename from description
                import re
                filename = re.sub(r'[^\w\s]', '', task['description']).replace(' ', '_').lower() + ".py"
                file_path = write_file(filename, code)
                completed.append({"task_id": task["id"], "result": f"File written to {file_path}\nCode:\n{code}"})
                final_context += f"\n{code}"

        elif task["type"] == "critique":
            prompt = f"Evaluate the following progress:\n{final_context}\nProvide a score (1-10) and justification."
            res = await critic.arun(prompt)
            completed.append({"task_id": task["id"], "result": res.content.strip()})

    # 3. Final synthesis
    synth = await researcher.arun(f"Summarize the whole project: {req.goal}\nContext:\n{final_context}")
    save_tree_history(req.goal, synth.content.strip(), completed)
    return ProjectResponse(tasks_completed=completed, final_summary=synth.content.strip())

@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest):
    count = ingest_document(req.file_path)
    return {"status": "success", "chunks_ingested": count}

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)