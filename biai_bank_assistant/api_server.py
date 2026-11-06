"""Serve the GPU model stack over HTTP so the PySide6 desktop app can use it.

Run this where the models live (a Kaggle/Colab GPU notebook, or any host with
torch installed). The desktop app then sets BIAI_API_URL and gets the same
answer the notebook's evaluate.py produced, over one POST per question.

    uvicorn api_server:app --host 0.0.0.0 --port 8000
    BIAI_API_TOKEN=secret python api_server.py

Kaggle gives no inbound port, so publish the local port with a quick tunnel
(cloudflared or ngrok) and put the printed https URL in the desktop app.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path

from src.assistant import BankAssistant
from src.data import load_faq
from src.remote import to_payload

ROOT = Path(__file__).resolve().parent
MAX_QUESTION_CHARS = 300

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel
except ModuleNotFoundError as exc:  # pragma: no cover - the dependency is optional
    raise SystemExit("This endpoint needs FastAPI: pip install fastapi uvicorn") from exc


api_token = os.getenv("BIAI_API_TOKEN", "")
assistant = BankAssistant(load_faq(ROOT / "data" / "bank-faq.csv"))
assistant.generator.load()

app = FastAPI(title="NB Assist model endpoint", version="1.0")


class Question(BaseModel):
    question: str


def check_authorization(authorization: str | None = Header(default=None)) -> None:
    """Refuse everyone but the desktop app when a token is configured."""
    if not api_token:
        return
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, api_token):
        raise HTTPException(status_code=401, detail="Missing or wrong BIAI_API_TOKEN")


@app.get("/health")
def health() -> dict:
    return {
        "online": True,
        "retrieval_mode": assistant.retriever.mode,
        "generator_loaded": assistant.generator.available,
        "generator_status": assistant.generator.load_error or "loaded",
        "generation_model": assistant.generator.model_name,
        "token_required": bool(api_token),
    }


@app.post("/answer", dependencies=[Depends(check_authorization)])
def answer(body: Question) -> dict:
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=413, detail=f"question longer than {MAX_QUESTION_CHARS} characters")
    return to_payload(assistant.answer(question))


if __name__ == "__main__":  # pragma: no cover - manual launch
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
