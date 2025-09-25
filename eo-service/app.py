import os, io, base64, torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# Hugging Face
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM

"""
EO-1 notes:
- Public HF id commonly seen: "IPEC-COMMUNITY/EO-1-3B" (adjust if you use another ckpt)
- EO-1 repos often ship a custom processor with a .generate(...) helper.
- We keep the /caption schema identical to molmo-service for drop-in replacement.
"""

MODEL_ID = os.getenv("EO_MODEL_ID", "IPEC-COMMUNITY/EO-1-3B")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# Silence TF etc, optional
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

print(f"[eo-service] Loading model: {MODEL_ID} on {DEVICE}")

# Load processor (trust_remote_code because EO-1 uses custom code)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# EO-1 may register as AutoModel or AutoModelForCausalLM depending on repo;
# try causal LM first, then generic AutoModel.
_model = None
_errs = []
for cls in (AutoModelForCausalLM, AutoModel):
    try:
        _model = cls.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto" if DEVICE == "cuda" else None
        )
        break
    except Exception as e:
        _errs.append(str(e))

if _model is None:
    raise RuntimeError(f"Failed to load EO-1 model: {' | '.join(_errs)}")

model = _model.eval()

app = FastAPI(title="EO-1 Service (Molmo-compatible)")

class CaptionIn(BaseModel):
    image_b64: str
    prompt: str = "Describe the image."

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_id": MODEL_ID,
        "device": DEVICE,
    }

def _open_image(b64: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

@app.post("/caption")
def caption(inp: CaptionIn):
    """
    Keep the endpoint identical to Molmo:
      IN : { image_b64, prompt }
      OUT: { caption: str, actions?: list }
    """
    img = _open_image(inp.image_b64)

    # EO-1 repos typically expect a dict with keys like:
    #  "observation.images.image" and "task" (or similar)
    # We try the most common schema first.
    batch = {
        "observation.images.image": [img],
        "task": [inp.prompt],
    }

    with torch.inference_mode():
        # Prefer processor.generate(model, batch) if implemented
        caption_text = None
        actions_out = None

        # Some repos implement a Processor.generate(model, inputs) helper:
        gen = getattr(processor, "generate", None)
        if callable(gen):
            out = processor.generate(model, batch)
            # Common attributes (defensive):
            caption_text = getattr(out, "text", None) or getattr(out, "caption", None)
            actions_out  = getattr(out, "action", None) or getattr(out, "actions", None)
        else:
            # Fallback path: preprocess then model.generate if available
            proc_inputs = processor(**batch, return_tensors="pt")
            # Move tensors to model device
            for k, v in list(proc_inputs.items()):
                if isinstance(v, torch.Tensor):
                    proc_inputs[k] = v.to(next(model.parameters()).device)

            # Try model.generate(...) (for causal models)
            mg = getattr(model, "generate", None)
            if callable(mg):
                gen_ids = model.generate(**proc_inputs, max_new_tokens=64, do_sample=False)
                # Use processor (tokenizer) to decode if available
                decode = getattr(processor, "batch_decode", None)
                if callable(decode):
                    caption_text = processor.batch_decode(gen_ids)[0].strip()
                else:
                    # Very last resort
                    caption_text = "EO-1 generated output."
            else:
                # As a final fallback, call the model forward and hope repo returns text
                out = model(**proc_inputs)
                caption_text = getattr(out, "text", None) or "EO-1 forward() run. No decode path."

    result = {"caption": str(caption_text or "").strip()}
    if actions_out is not None:
        try:
            # Numpy/torch to list if needed
            result["actions"] = actions_out.tolist() if hasattr(actions_out, "tolist") else actions_out
        except Exception:
            pass
    return result
