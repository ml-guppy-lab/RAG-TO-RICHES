import sys
import os

# ensure ragDialogues is importable from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ragDialogues import get_best_dialogue, FALLBACKS

_fallback_dialogues = {f["dialogue"] for f in FALLBACKS}

app = FastAPI(title="RAG to Riches API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class SituationRequest(BaseModel):
    situation: str


@app.post("/dialogue")
def dialogue(req: SituationRequest):
    try:
        result = get_best_dialogue(req.situation)
        result["is_fallback"] = result.get("dialogue") in _fallback_dialogues
        return result
    except Exception:
        return {"error": "Something went wrong. Please try again."}
