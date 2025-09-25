import os, base64, io, torch, traceback
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

# ───────── Paths ─────────
LOCAL_DIR = (Path(__file__).resolve().parents[1] / "MolmoE-1B-0924-NF4").resolve()
OFFLOAD_DIR = (Path(__file__).resolve().parent / "offload").resolve()
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ───────── Env (quiet TF) ─────────
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")

print(f"[molmo-service] Using model folder: {LOCAL_DIR}")
if not LOCAL_DIR.exists():
    raise FileNotFoundError(f"MOLMO_LOCAL_DIR not found: {LOCAL_DIR}")

LOCAL_DIR_STR = LOCAL_DIR.as_posix()
OFFLOAD_DIR_STR = OFFLOAD_DIR.as_posix()

# ───────── Load model/processor ─────────
processor = AutoProcessor.from_pretrained(
    LOCAL_DIR_STR, trust_remote_code=True, local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_DIR_STR,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype="auto",        # (Warning says deprecated, but still accepted)
    device_map="auto",
    offload_folder=OFFLOAD_DIR_STR,
)
model.eval()

# Input-embedding device (place inputs here under sharding)
try:
    EMBED_DEVICE = model.get_input_embeddings().weight.device
except Exception:
    EMBED_DEVICE = next(model.parameters()).device

# ───────── Generation defaults ─────────
tok = getattr(processor, "tokenizer", None)  # may be None
eos_id = getattr(model.config, "eos_token_id", None) or (getattr(tok, "eos_token_id", None) if tok else None)
pad_id = (getattr(tok, "pad_token_id", None) if tok else None) or eos_id

DEFAULT_GENCFG = GenerationConfig(
    max_new_tokens=64,
    do_sample=False,
    num_beams=1,
    eos_token_id=eos_id,
    pad_token_id=pad_id,
)
DEFAULT_STOP = "<|endoftext|>"
if tok is not None:
    setattr(DEFAULT_GENCFG, "stop_strings", DEFAULT_STOP)

if getattr(model, "generation_config", None) is None:
    model.generation_config = GenerationConfig.from_model_config(model.config)

def _fill_gc(g: GenerationConfig):
    if g.max_new_tokens is None: g.max_new_tokens = 64
    if g.do_sample is None: g.do_sample = False
    if g.num_beams is None: g.num_beams = 1
    if g.eos_token_id is None: g.eos_token_id = eos_id
    if g.pad_token_id is None: g.pad_token_id = pad_id
    if tok is not None and not hasattr(g, "stop_strings"):
        setattr(g, "stop_strings", DEFAULT_STOP)

_fill_gc(model.generation_config)

app = FastAPI()

# ───────── Schemas ─────────
class CaptionIn(BaseModel):
    image_b64: str
    prompt: str = "Describe the image."

# ───────── Helpers ─────────
def _normalize_images(key: str, value):
    """
    Normalize 'images' / 'pixel_values' / 'image_values' / 'image'.
    Molmo expects 'images' as embedded tokens [B, T, N, D], NOT raw pixels.

    • If key == 'images':
        - Tensor 3D  [T, N, D] -> [1, T, N, D]
        - Tensor 4D  [B, T, N, D] -> as-is
        - List/Tuple of [T, N, D] -> stack on batch -> [B, T, N, D]
        (device move only; DO NOT cast dtype)
    • Else (pixel paths like 'pixel_values'):
        - Tensor 3D  [C, H, W]   -> [1, C, H, W]
        - Tensor 4D  [B, C, H, W] -> as-is
        - List/Tuple of CHW -> stack on batch
        (device move only; no dtype cast here)
    """
    if key == "images":
        if isinstance(value, torch.Tensor):
            x = value.to(EMBED_DEVICE)
            if x.dim() == 3:      # [T, N, D]
                x = x.unsqueeze(0)  # -> [1, T, N, D]
            # if already 4D+ assume [B,T,N,D] and leave as-is
            return x
        else:
            tensors = []
            for item in value:
                if not isinstance(item, torch.Tensor):
                    raise TypeError("images list contains non-tensor element")
                t = item.to(EMBED_DEVICE)
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                tensors.append(t)
            return torch.cat(tensors, dim=0)  # [B,T,N,D]

    # Pixel path
    if isinstance(value, torch.Tensor):
        x = value.to(EMBED_DEVICE)
        if x.dim() == 3:          # CHW -> BCHW
            x = x.unsqueeze(0)
        return x

    tensors = []
    for item in value:
        if not isinstance(item, torch.Tensor):
            raise TypeError(f"{key} list contains non-tensor element")
        t = item.to(EMBED_DEVICE)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        tensors.append(t)
    return torch.cat(tensors, dim=0)

def _normalize_image_meta_like(batch: dict):
    """
    For ANY key containing 'image' (excluding actual image tensors),
    ensure a batch dimension is present. Aim for [B, N, P] if 2D/1D.
    Uses N from 'images' if available (token embeddings path).
    """
    image_tensor_keys = {"images", "pixel_values", "image_values", "image"}

    # discover N from token-embedding 'images' if present
    N = None
    if "images" in batch and isinstance(batch["images"], torch.Tensor):
        x = batch["images"]
        # [B,T,N,D] -> N is dim 2
        if x.dim() >= 4:
            try:
                N = int(x.shape[2])
            except Exception:
                N = None

    for k, v in list(batch.items()):
        if not isinstance(v, torch.Tensor):
            continue
        if k in image_tensor_keys:
            continue
        if "image" not in k.lower():
            continue

        t = v.to(EMBED_DEVICE)
        if t.dim() == 1:
            # [P] -> [1,1,P]
            batch[k] = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 2:
            # Prefer [N,P] -> [1,N,P]; if not matching, still add batch: [1,*,*]
            if (N is not None) and (t.shape[0] == N):
                batch[k] = t.unsqueeze(0)
            else:
                batch[k] = t.unsqueeze(0)
        else:
            # already batched or higher-dim; leave as-is
            batch[k] = t

def _prep_batch(pil_image: Image.Image, prompt: str):
    """
    Build a multimodal batch with the processor and shape corrections:
    - 'images' as token embeddings: [B, T, N, D]
    - image meta tensors (image_input_idx, image_masks, etc.): [B, N, P]
    - non-image tensors: add batch dim only for 1D (tokens)
    """
    batch = processor.process(images=[pil_image], text=prompt)
    batch = {k: v for k, v in batch.items() if v is not None}

    image_keys = {"pixel_values", "images", "image_values", "image"}

    for k, v in list(batch.items()):
        if k in image_keys:
            batch[k] = _normalize_images(k, v)
        elif isinstance(v, torch.Tensor):
            # non-image tensors: only add batch for 1D (tokens), leave 2D/3D as-is
            if v.dim() == 1:
                v = v.unsqueeze(0)
            batch[k] = v.to(EMBED_DEVICE)
        elif isinstance(v, (list, tuple)):
            new_list = []
            for item in v:
                if isinstance(item, torch.Tensor):
                    if item.dim() == 1:
                        item = item.unsqueeze(0)
                    item = item.to(EMBED_DEVICE)
                new_list.append(item)
            batch[k] = new_list

    # Make sure all image meta tensors gain a batch dim (and match N where possible)
    _normalize_image_meta_like(batch)

    # Drop broken cache entries if present
    if "past_key_values" in batch:
        val = batch["past_key_values"]
        if val is None or (isinstance(val, (list, tuple)) and any(x is None for x in val)):
            batch.pop("past_key_values", None)

    return batch

def _decode(gen_out):
    """
    Robustly decode whatever the model returns:
    - Tensor of token ids
    - dict with 'sequences' or 'text'
    - ModelOutput with .sequences
    """
    if isinstance(gen_out, dict):
        if "text" in gen_out and isinstance(gen_out["text"], str):
            return gen_out["text"].strip()
        if "generated_text" in gen_out and isinstance(gen_out["generated_text"], str):
            return gen_out["generated_text"].strip()
        if "sequences" in gen_out:
            seq = gen_out["sequences"]
            if isinstance(seq, torch.Tensor):
                if tok is not None and hasattr(tok, "batch_decode"):
                    return tok.batch_decode(seq, skip_special_tokens=True)[0].strip()
                if hasattr(processor, "batch_decode"):
                    return processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
                return str(seq.tolist())

    if hasattr(gen_out, "sequences"):
        seq = gen_out.sequences
        if tok is not None and hasattr(tok, "batch_decode"):
            return tok.batch_decode(seq, skip_special_tokens=True)[0].strip()
        if hasattr(processor, "batch_decode"):
            return processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
        if isinstance(seq, torch.Tensor):
            return str(seq.tolist())

    if isinstance(gen_out, torch.Tensor):
        if tok is not None and hasattr(tok, "batch_decode"):
            return tok.batch_decode(gen_out, skip_special_tokens=True)[0].strip()
        if hasattr(processor, "batch_decode"):
            return processor.batch_decode(gen_out, skip_special_tokens=True)[0].strip()
        return str(gen_out.tolist())

    if isinstance(gen_out, str):
        return gen_out.strip()

    return str(gen_out)

# ───────── Routes ─────────
@app.get("/health")
def health():
    return {
        "ok": True,
        "local_dir": str(LOCAL_DIR),
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "has_generate_from_batch": hasattr(model, "generate_from_batch"),
        "embed_device": str(EMBED_DEVICE),
    }

@app.post("/caption")
def caption(inp: CaptionIn):
    # Decode image
    try:
        img_bytes = base64.b64decode(inp.image_b64)
        if not img_bytes:
            raise ValueError("empty payload")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Build batch
    try:
        batch = _prep_batch(img, inp.prompt)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"processor_error: {type(e).__name__}: {e}")

    if not any(k in batch for k in ("input_ids", "pixel_values", "images", "image_values", "image")):
        raise HTTPException(status_code=422, detail="Processor produced empty batch")

    # Generate (tokenizer only if stop_strings present); disable cache to avoid past_kv bug
    try:
        with torch.inference_mode():
            use_tok = hasattr(DEFAULT_GENCFG, "stop_strings") and tok is not None
            gen_kwargs = dict(generation_config=DEFAULT_GENCFG, use_cache=False)
            if hasattr(model, "generate_from_batch"):
                if use_tok:
                    gen_out = model.generate_from_batch(batch, DEFAULT_GENCFG, tokenizer=tok, use_cache=False)
                else:
                    gen_out = model.generate_from_batch(batch, DEFAULT_GENCFG, use_cache=False)
            else:
                if use_tok:
                    gen_out = model.generate(**batch, tokenizer=tok, **gen_kwargs)
                else:
                    gen_out = model.generate(**batch, **gen_kwargs)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"generation_error: {type(e).__name__}: {e}")

    # Decode
    try:
        text = _decode(gen_out)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"decode_error: {type(e).__name__}: {e}")

    return {"caption": text}
