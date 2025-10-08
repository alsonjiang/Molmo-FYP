import os, io, base64, time, traceback
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = (ROOT / "MolmoE-1B-0924-NF4").resolve()
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

# ── Quiet logs ───────────────────────────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ── Settings ─────────────────────────────────────────────────────────────────
# Keep outputs short while testing; you can raise later.
MAX_NEW_TOKENS = int(os.getenv("MOLMO_MAX_NEW_TOKENS", "32"))
# Downscale very large images to reduce vision backbone cost.
MAX_SIDE = int(os.getenv("MOLMO_MAX_SIDE", "768"))
# If you get any dtype mismatch, flip this to 1 to force full FP32 (slower, but safe)
FORCE_FP32 = os.getenv("MOLMO_FORCE_FP32", "0") == "1"

# ── Load processor/model fully on GPU ────────────────────────────────────────
processor = AutoProcessor.from_pretrained(MODEL_DIR.as_posix(), trust_remote_code=True, local_files_only=True)

# CRITICAL: put the whole model on cuda:0, no CPU/disk offload, with fp16 math.
dtype = torch.float32 if FORCE_FP32 else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR.as_posix(),
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map={"": 0},            # force all modules on GPU 0
    offload_folder=None,           # no CPU/disk offload
)
model.eval()

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
    # use_cache speeds multi-token decode; keep True unless you hit a KV bug
    use_cache=True,
)
if tok is not None:
    setattr(GENCFG, "stop_strings", "<|endoftext|>")

# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI()

class CaptionIn(BaseModel):
    image_b64: str
    prompt: Optional[str] = "Describe the image."

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
    # move to GPU; cast floats to model dtype (fp16/fp32)
    if t.is_floating_point():
        return t.to(EMBED_DEVICE, dtype=MODEL_DTYPE, non_blocking=True)
    return t.to(EMBED_DEVICE, non_blocking=True)

def _build_batch(img: Image.Image, prompt: str) -> Dict[str, Any]:
    batch = processor.process(images=[img], text=prompt)
    batch = {k: v for k, v in batch.items() if v is not None}

    """
    # ✅ Force pixel path; drop any pre-embedded 'images' keys
    if "pixel_values" in batch:
        batch.pop("images", None)
        batch.pop("image_values", None)
        batch.pop("image", None)
        
    # Now normalize just like before…
    has_image = False
    if "pixel_values" in batch and isinstance(batch["pixel_values"], torch.Tensor):
        x = batch["pixel_values"]
        if x.dim() == 3:  # [C,H,W] -> [1,C,H,W]
            x = x.unsqueeze(0)
        batch["pixel_values"] = _move_cast(x)
        has_image = True 
    if not has_image:
        raise HTTPException(status_code=422, detail="no pixel_values from processor")
    """
    batch["images"] = torch.unsqueeze(batch["images"],0)
    batch["image_input_idx"] = torch.unsqueeze(batch["image_input_idx"],0)
    batch["image_masks"] = torch.unsqueeze(batch["image_masks"],0)
    # Move non-image tensors; add batch dim for 1D
    for k, v in list(batch.items()):
        #if k == "pixel_values":
        #    continue
        print(f"{k} dimensions: {v.shape}")
        if isinstance(v, torch.Tensor):
            if v.dim() == 1:
                v = torch.unsqueeze(v,0)
            batch[k] = v.to(EMBED_DEVICE, non_blocking=True)

    # Drop broken caches and return
    if "past_key_values" in batch:
        pkv = batch["past_key_values"]
        if pkv is None or (isinstance(pkv, (list, tuple)) and any(x is None for x in pkv)):
            batch.pop("past_key_values", None)
    return batch


def _generate(batch: Dict[str, Any], timings: Dict[str, float] = None) -> str:
    t0 = time.time()
        with torch.inference_mode():
        if hasattr(model, "generate_from_batch"):
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                out = model.generate_from_batch(batch, GENCFG, tokenizer=tok, use_cache=False)
        else:
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
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

@app.on_event("startup")
def _warmup():
    try:
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        img = _resize_if_needed(img)
        batch = _build_batch(img, "ok")
        _ = _generate(batch)
        print("[warmup] Molmo ready on", EMBED_DEVICE, "dtype", MODEL_DTYPE)
    except Exception as e:
        print("[warmup] skipped:", e)

@app.get("/health")
def health():
    try:
        return {
            "ok": True,
            "model_dir": str(MODEL_DTYPE),
            "device": str(EMBED_DEVICE),
            "dtype": str(MODEL_DTYPE),
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_side": MAX_SIDE,
            "force_fp32": FORCE_FP32,
            "has_generate_from_batch": hasattr(model, "generate_from_batch"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

@app.get("/profile")
def profile():
    """Run a tiny in-memory image and return stage timings."""
    times = {}
    try:
        img = Image.new("RGB", (320, 240), (180, 180, 180))
        img = _resize_if_needed(img)
        _ = _build_batch(img, "ok", timings=times)
        _ = _generate(_, timings=times)
        times["ok"] = True
    except Exception as e:
        times["ok"] = False
        times["error"] = f"{type(e).__name__}: {e}"
    return times

@app.post("/caption")
def caption(inp: CaptionIn):
    img = _b64_to_pil(inp.image_b64)
    img = _resize_if_needed(img)
    batch = _build_batch(img, inp.prompt or "Describe the image.")
    text = _generate(batch)
    return {"caption": text}
