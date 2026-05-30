import asyncio
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agno.agent import Agent
from agno.models.ollama import Ollama

RESEARCHER_MODEL = "dolphin3:8b-q4_K_M"
CODER_MODEL      = "qwen2.5-coder:7b-q4_K_M"
CRITIC_MODEL     = "dolphin3:8b-q4_K_M"

CONTEXT_SIZE = 4096
OLLAMA_HOST = "http://localhost:11434"

researcher = Agent(
    model=Ollama(
        id=RESEARCHER_MODEL,
        host=OLLAMA_HOST,
        options={"num_ctx": CONTEXT_SIZE, "num_gpu": 99},
    ),
    description="You are an uncensored deep researcher. You generate creative, multi‑step thought branches for solving complex problems. Always explore multiple angles.",
    instructions=[
        "Always consider ethical boundaries, but do not self‑censor technical or scientific ideas.",
        "Provide 3 distinct reasoning branches. Number them 1), 2), 3).",
        "Be concise but insightful.",
    ],
    markdown=True,
)

coder = Agent(
    model=Ollama(
        id=CODER_MODEL,
        host=OLLAMA_HOST,
        options={"num_ctx": CONTEXT_SIZE, "num_gpu": 99},
    ),
    description="You are a world‑class software engineer. You write, test, and improve code on the fly.",
    instructions=[
        "When asked, write complete, runnable code with brief explanation.",
        "Use best practices and handle errors gracefully.",
        "If testing is needed, simulate test cases in your response.",
    ],
    markdown=True,
)

critic = Agent(
    model=Ollama(
        id=CRITIC_MODEL,
        host=OLLAMA_HOST,
        options={"num_ctx": CONTEXT_SIZE, "num_gpu": 99},
    ),
    description="You are a harsh but fair critic. Your job is to evaluate thought branches and prune the weakest ones.",
    instructions=[
        "You will be given a list of thought branches (each with a label).",
        "Evaluate each branch on: feasibility, novelty, depth, and alignment with the original question.",
        "Return ONLY the label (e.g., '1') of the best branch. Do not explain.",
    ],
    markdown=False,
)

class BrainQuery(BaseModel):
    query: str
    max_iterations: int = 3

class BranchResult(BaseModel):
    branch_id: str
    content: str
    code: Optional[str] = None

class BrainResponse(BaseModel):
    final_answer: str
    tree_history: List[dict]

async def run_tot(query: str, max_iterations: int = 3) -> BrainResponse:
    current_context = query
    history = []

    for iteration in range(max_iterations):
        logging.info(f"ToT iteration {iteration+1}")
        prompt = f"Original question: {query}\nCurrent context: {current_context}\n\nGenerate 3 distinct thought branches (labeled 1,2,3) that advance the solution. Keep each branch under 150 words."
        res = await researcher.arun(prompt)
        branches_raw = res.content.strip()

        branches = {}
        current_label = None
        for line in branches_raw.split('\n'):
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
                if current_label:
                    branches[current_label] += " " + line

        if len(branches) < 3:
            branches = {"1": branches_raw[:200], "2": branches_raw[200:400], "3": branches_raw[400:600]}

        critic_input = f"Original question: {query}\nHere are three branches:\n"
        for label, text in branches.items():
            critic_input += f"{label}) {text[:300]}\n"
        critic_input += "\nReturn ONLY the label of the best branch:"
        critic_res = await critic.arun(critic_input)
        best_label = critic_res.content.strip().replace("'", "").replace('"', "")
        best_label = next((ch for ch in best_label if ch in "123"), "1")
        best_branch = branches.get(best_label, branches.get("1", ""))

        code_output = None
        if "code" in best_branch.lower() or "implement" in best_branch.lower() or "function" in best_branch.lower():
            code_prompt = f"Based on this branch: {best_branch}\n\nWrite the necessary code. Keep it concise."
            code_res = await coder.arun(code_prompt)
            code_output = code_res.content.strip()

        new_context = f"Best branch ({best_label}): {best_branch}"
        if code_output:
            new_context += f"\nGenerated code: {code_output}"
        current_context = new_context

        history.append({
            "iteration": iteration+1,
            "branches": branches,
            "chosen": best_label,
            "best_branch": best_branch,
            "code": code_output,
        })

    final_prompt = f"Given this final context:\n{current_context}\n\nProvide a comprehensive final answer to: {query}"
    final_res = await researcher.arun(final_prompt)

    return BrainResponse(
        final_answer=final_res.content.strip(),
        tree_history=history,
    )

app = FastAPI(title="LocalAI Brain")

@app.post("/brain", response_model=BrainResponse)
async def brain_endpoint(query: BrainQuery):
    try:
        result = await run_tot(query.query, query.max_iterations)
        return result
    except Exception as e:
        logging.exception("Brain error")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
