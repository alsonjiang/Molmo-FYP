import os
import io
import base64
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

import torch
from transformers import AutoModelForCausalLM

# ───────── Config ─────────

MODEL_ID = os.getenv("MOONDREAM_MODEL_ID", "vikhyatk/moondream2")
MODEL_REVISION = os.getenv("MOONDREAM_REVISION", "2025-06-21")

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = os.getenv("DEVICE", DEFAULT_DEVICE)

print(f"[moondream-service] Loading {MODEL_ID}@{MODEL_REVISION} on {DEVICE}...")


# ───────── Model load ─────────

# This uses the custom moondream helper methods (caption/query) from hf_moondream.py
# via trust_remote_code.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True,
    device_map={"": DEVICE},
)


# ───────── FastAPI setup ─────────

app = FastAPI(
    title="Moondream Service (Molmo-compatible)",
    version="0.1.0",
)


# ───────── Schemas (match orchestrator) ─────────

class CaptionRequest(BaseModel):
    image_b64: str
    prompt: Optional[str] = None


class CaptionResponse(BaseModel):
    caption: str


# ───────── Helpers ─────────

def load_image_from_b64(data: str) -> Image.Image:
    # Accept both raw base64 and data: URLs
    if "," in data:
        _, _, data = data.partition(",")

    try:
        img_bytes = base64.b64decode(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    return img


# ───────── Routes ─────────

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": DEVICE,
    }


@app.post("/caption", response_model=CaptionResponse)
def caption(req: CaptionRequest):
    """
    Molmo-compatible endpoint.

    Orchestrator sends:
        { "image_b64": "<base64>", "prompt": "<question>" }

    We call model.query(image, question) and return:
        { "caption": "<answer>" }
    """
    if not req.image_b64:
        raise HTTPException(status_code=400, detail="image_b64 is required")

    question = req.prompt or (
        "Are the two people in the image the same person? "
        "Answer with exactly 'same person' or 'different person'."
    )

    image = load_image_from_b64(req.image_b64)

    try:
        # Moondream HF impl: model.query(image, question)["answer"]
        result = model.query(image, question)
        answer = (result.get("answer") or "").strip()
    except Exception as e:
        # This is what was previously surfacing as a 500
        print("[moondream error]", repr(e), flush=True)
        raise HTTPException(status_code=500, detail=f"Moondream error: {e}")

    return CaptionResponse(caption=answer)
