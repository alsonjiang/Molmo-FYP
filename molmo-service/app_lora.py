import os, io, base64, time, traceback
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
import peft
import uvicorn

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = (ROOT / "MolmoE-1B-0924-NF4").resolve()
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")
LORA_DIR = (ROOT / "MolmoE_Human_Pointing_LoRA").resolve()
if not LORA_DIR.exists():
    raise FileNotFoundError(f"LoRA not found: {LORA_DIR}, loading base model without LoRA.")

# ── Quiet logs ───────────────────────────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ── Settings ─────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = int(os.getenv("MOLMO_MAX_NEW_TOKENS", "64"))
MAX_SIDE = int(os.getenv("MOLMO_MAX_SIDE", "768"))
FORCE_FP32 = os.getenv("MOLMO_FORCE_FP32", "0") == "1"
FORCE_CPU  = os.getenv("MOLMO_FORCE_CPU", "0") == "1"   # optional override

# ── Load processor/model with safe device logic ──────────────────────────────
processor = AutoProcessor.from_pretrained(
    MODEL_DIR.as_posix(), trust_remote_code=True, local_files_only=True
)

USE_CUDA = torch.cuda.is_available() and not FORCE_CPU
# dtype: fp16 on CUDA (unless FORCE_FP32), fp32 on CPU for safety
dtype = torch.float16 if (USE_CUDA and not FORCE_FP32) else torch.float32

# Device map: CUDA auto if available, otherwise stick to CPU
device_map = "auto" if USE_CUDA else {"": "cpu"}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR.as_posix(),
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=dtype,
    device_map=device_map,
    offload_folder=None,
)
model.eval()
if LORA_DIR.exists():
    model = peft.PeftModel.from_pretrained(
        model,
        LORA_DIR.as_posix(),
        torch_dtype=dtype,
        device_map=device_map,
        offload_folder=None,
    )
    print(f"[molmo-service] Loaded LoRA from {LORA_DIR.name}")

try:
    EMBED_DEVICE = model.get_input_embeddings().weight.device
except Exception:
    EMBED_DEVICE = next(model.parameters()).device
MODEL_DTYPE = next(model.parameters()).dtype

tok = getattr(processor, "tokenizer", None)
eos_id = getattr(model.config, "eos_token_id", None) or (getattr(tok, "eos_token_id", None) if tok else None)
pad_id = (getattr(tok, "pad_token_id", None) if tok else None) or eos_id

GENCFG = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    num_beams=1,
    eos_token_id=eos_id,
    pad_token_id=pad_id,
    use_cache=True,
)
if tok is not None:
    setattr(GENCFG, "stop_strings", "<|endoftext|>")

# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI()

class CaptionIn(BaseModel):
    image_b64: str
    prompt: Optional[str] = "Describe the image."


# ── Helpers ──────────────────────────────────────────────────────────────────

def _b64_to_pil(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64)
        if not raw:
            raise ValueError("empty payload")
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")

def _resize_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= MAX_SIDE:
        return img
    scale = MAX_SIDE / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)

def _move_cast(t: torch.Tensor) -> torch.Tensor:
    # move to model device; cast floats to model dtype (fp16/fp32)
    if t.is_floating_point():
        return t.to(EMBED_DEVICE, dtype=MODEL_DTYPE, non_blocking=True)
    return t.to(EMBED_DEVICE, non_blocking=True)

def _build_batch(img: Image.Image, prompt: str, timings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    t0 = time.time()
    batch = processor.process(images=[img], text=prompt)
    batch = {k: v for k, v in batch.items() if v is not None}

    # Ensure explicit batch dim
    batch["images"] = torch.unsqueeze(batch["images"], 0)
    batch["image_input_idx"] = torch.unsqueeze(batch["image_input_idx"], 0)
    batch["image_masks"] = torch.unsqueeze(batch["image_masks"], 0)

    for k, v in list(batch.items()):
        # Debug print (you can comment this out if too noisy)
        print(f"{k} dimensions: {getattr(v, 'shape', None)}")
        if isinstance(v, torch.Tensor):
            if v.dim() == 1:
                v = torch.unsqueeze(v, 0)
            batch[k] = v.to(EMBED_DEVICE, non_blocking=True)

    if "past_key_values" in batch:
        pkv = batch["past_key_values"]
        if pkv is None or (isinstance(pkv, (list, tuple)) and any(x is None for x in pkv)):
            batch.pop("past_key_values", None)

    if timings is not None:
        timings["prep_s"] = time.time() - t0
    return batch

def _generate(batch: Dict[str, Any], timings: Optional[Dict[str, float]] = None) -> str:
    t0 = time.time()
    use_cuda_autocast = (EMBED_DEVICE.type == "cuda") and (MODEL_DTYPE == torch.float16)
    with torch.inference_mode():
        if hasattr(model, "generate_from_batch"):
            with torch.autocast(device_type="cuda", enabled=use_cuda_autocast, dtype=torch.float16):
                out = model.generate_from_batch(batch, GENCFG, tokenizer=tok, use_cache=False)
        else:
            with torch.autocast(device_type="cuda", enabled=use_cuda_autocast, dtype=torch.float16):
                out = model.generate(**batch, generation_config=GENCFG, use_cache=False)

    if timings is not None:
        timings["generate_s"] = time.time() - t0

    # decode
    if isinstance(out, dict) and "sequences" in out and isinstance(out["sequences"], torch.Tensor):
        seq = out["sequences"]
        if tok and hasattr(tok, "batch_decode"):
            return tok.batch_decode(seq, skip_special_tokens=True)[0].strip()
        return str(seq.tolist())
    if hasattr(out, "sequences"):
        seq = out.sequences
        if tok and hasattr(tok, "batch_decode") and isinstance(seq, torch.Tensor):
            return tok.batch_decode(seq, skip_special_tokens=True)[0].strip()
        return str(seq)
    if isinstance(out, torch.Tensor):
        if tok and hasattr(tok, "batch_decode"):
            return tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        return str(out.tolist())
    if isinstance(out, str):
        return out.strip()
    return str(out)

def _strip_to_answer(text: str) -> str:
    """
    Given a Molmo-style chat-like transcript, keep only the assistant answer.

    Example raw:
      'User: ... same person\\n different person Assistant: same person'

    We want:
      'same person'
    """
    if not text:
        return ""

    t = text.strip()
    lower = t.lower()

    # If we see 'assistant:', keep only the part after the LAST occurrence
    if "assistant:" in lower:
        last = lower.rfind("assistant:")
        # Slice original string to keep original casing
        t = t[last + len("assistant:"):].strip()

    # Only keep the first line after that
    t = t.splitlines()[0].strip()

    return t


# ── Lifespan / health / profile ──────────────────────────────────────────────

async def lifespan(app: FastAPI):
    try:
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        img = _resize_if_needed(img)
        batch = _build_batch(img, "ok")
        _ = _generate(batch)
        print("[warmup] Molmo ready on", EMBED_DEVICE, "dtype", MODEL_DTYPE)
    except Exception as e:
        print("[warmup] skipped:", e)
    yield
    print("[shutdown] Molmo service exiting.")

@app.get("/health")
def health():
    try:
        return {
            "ok": True,
            "model_dir": str(MODEL_DIR),
            "device": str(EMBED_DEVICE),
            "dtype": str(MODEL_DTYPE),
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_side": MAX_SIDE,
            "force_fp32": FORCE_FP32,
            "force_cpu": FORCE_CPU,
            "has_generate_from_batch": hasattr(model, "generate_from_batch"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

@app.get("/profile")
def profile():
    """Run a tiny in-memory image and return stage timings."""
    times: Dict[str, float] = {}
    try:
        img = Image.new("RGB", (320, 240), (180, 180, 180))
        img = _resize_if_needed(img)
        batch = _build_batch(img, "ok", timings=times)
        _ = _generate(batch, timings=times)
        times["ok"] = True
    except Exception as e:
        times["ok"] = False
        times["error"] = f"{type(e).__name__}: {e}"
    return times


# ── Caption endpoint ─────────────────────────────────────────────────────────

@app.post("/caption")
def caption(inp: CaptionIn):
    img = _b64_to_pil(inp.image_b64)
    img = _resize_if_needed(img)
    batch = _build_batch(img, inp.prompt or "Describe the image.")
    raw = _generate(batch)

    # Strip to just the assistant's answer for the HTTP response
    answer = _strip_to_answer(raw)

    # Optional debug logs to see what Molmo actually produced
    print(f"[molmo raw] {raw!r}")
    print(f"[molmo answer] {answer!r}")

    return {"caption": answer}


if __name__ == '__main__':
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False, workers=1)
