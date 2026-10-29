"""
Minimal long-lived serving process for inference.py.

Loads every model exactly once at startup and answers many requests against
that same in-memory registry - this is what actually fixes the "10 seconds
per message" problem, since `python inference.py` pays the full load cost
on every single invocation (a fresh process every time).

Run:
    uvicorn serve:app --host 0.0.0.0 --port 8000

Then, e.g. from Java (or curl):
    POST http://localhost:8000/score
    {"text": "you are an idiot"}
    -> {"toxic": 0.999, "inappropriate": 0.41, "cyberbullying": 0.87}
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

import inference

MODELS_DIR = "models"


def ensure_models() -> None:
    """On Railway, fetch models/ from the Hugging Face repo named by HF_MODEL_REPO.

    Locally HF_MODEL_REPO is unset, so the existing models/ folder is used as-is.
    HF_TOKEN is only needed when the repo is private.
    """
    repo_id = os.environ.get("HF_MODEL_REPO")
    if not repo_id:
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=MODELS_DIR, token=os.environ.get("HF_TOKEN"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_models()
    app.state.registry = inference.load_registry(MODELS_DIR)
    yield


app = FastAPI(lifespan=lifespan)


class ScoreRequest(BaseModel):
    text: str


@app.post("/score")
def score(req: ScoreRequest):
    return inference.score_text(app.state.registry, req.text)


@app.get("/health")
def health():
    return {"status": "ok"}
