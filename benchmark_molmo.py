# bench_molmo_local.py
import os, time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "MolmoE-1B-0924-NF4"
IMG_PATH = ROOT / "images" / "clock_face.png"\

PROMPT = "Describe the image"

assert MODEL_DIR.exists(), f"missing model dir: {MODEL_DIR}"
assert IMG_PATH.exists(), f"missing image: {IMG_PATH}"

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print("[bench] load processor…")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_DIR.as_posix(), trust_remote_code=True, local_files_only=True)
print(f"[bench] processor loaded in {time.time()-t0:.2f}s")

# Force whole model on GPU in fp16 (fastest path on 8GB 4060). Flip to float32 if you must.
print("[bench] load model (fp16, full GPU)…")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR.as_posix(),
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map={"": 0},      # put everything on cuda:0; no CPU/disk offload
    offload_folder=None,
)
model.eval()
try:
    model.config.use_cache = False   # avoid past_kv bug in this checkpoint
except Exception:
    pass

print(f"[bench] model loaded in {time.time()-t0:.2f}s")
EMBED_DEVICE = next(model.parameters()).device
MODEL_DTYPE  = next(model.parameters()).dtype
print("[bench] device:", EMBED_DEVICE, "dtype:", MODEL_DTYPE)

tok   = getattr(processor, "tokenizer", None)
eos   = getattr(model.config, "eos_token_id", None) or (getattr(tok, "eos_token_id", None) if tok else None)
pad   = (getattr(tok, "pad_token_id", None) if tok else None) or eos
gen   = GenerationConfig(max_new_tokens=32, do_sample=False, num_beams=1, eos_token_id=eos, pad_token_id=pad)
if tok is not None:
    setattr(gen, "stop_strings", "<|endoftext|>")

def _move_cast_img(x: torch.Tensor) -> torch.Tensor:
    # Move to GPU; cast floats to the model dtype (fp16/fp32)
    if x.is_floating_point():
        return x.to(EMBED_DEVICE, dtype=MODEL_DTYPE, non_blocking=True)
    return x.to(EMBED_DEVICE, non_blocking=True)

def _move_only(x: torch.Tensor) -> torch.Tensor:
    return x.to(EMBED_DEVICE, non_blocking=True)

def _normalize_image_meta_like(batch: dict):
    """
    Any key containing 'image' (but not the actual image tensors) should have shape [B,N,P].
    If it's 1D -> [1,1,P]; if it's 2D [N,P] -> [1,N,P].
    """
    image_tensor_keys = {"pixel_values", "images", "image_values", "image"}

    # Infer N (number of image sequences) from 'images' if present
    N = None
    if "images" in batch and isinstance(batch["images"], torch.Tensor) and batch["images"].dim() >= 4:
        # images is [B,T,N,D]
        try:
            N = int(batch["images"].shape[2])
        except Exception:
            N = None

    for k, v in list(batch.items()):
        if k in image_tensor_keys:
            continue
        if "image" not in k.lower():
            continue
        if not isinstance(v, torch.Tensor):
            continue

        t = _move_only(v)
        if t.dim() == 1:
            # [P] -> [1,1,P]
            batch[k] = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 2:
            # [N,P] (or something close) -> [1,N,P]
            batch[k] = t.unsqueeze(0)
        else:
            batch[k] = t  # already batched

def build_batch(img: Image.Image, prompt: str) -> dict:
    b = processor.process(images=[img], text=prompt)
    b = {k: v for k, v in b.items() if v is not None}

    # Normalize image tensors: prefer pixel path; else 'images' path
    has_image = False
    for key in ("pixel_values", "image", "image_values", "images"):
        if key in b and isinstance(b[key], torch.Tensor):
            x = b[key]
            if key == "images":
                # token-embedding path: [T,N,D] or [B,T,N,D]
                if x.dim() == 3:
                    x = x.unsqueeze(0)
            else:
                # pixel path: [C,H,W] or [B,C,H,W]
                if x.dim() == 3:
                    x = x.unsqueeze(0)
            b[key] = _move_cast_img(x)
            has_image = True
    if not has_image:
        raise RuntimeError("processor produced no image tensors")

    # Move token/mask tensors; add batch for 1D
    for k, v in list(b.items()):
        if k in {"pixel_values", "images", "image_values", "image"}:
            continue
        if isinstance(v, torch.Tensor):
            if v.dim() == 1:
                v = v.unsqueeze(0)
            b[k] = _move_only(v)

    # Fix image meta shapes for 'images' path consumers
    _normalize_image_meta_like(b)

    # Defensive: drop broken caches
    if "past_key_values" in b:
        pkv = b["past_key_values"]
        if pkv is None or (isinstance(pkv, (list, tuple)) and any(x is None for x in pkv)):
            b.pop("past_key_values", None)

    return b

# Load image (downscale if huge just to keep vision pass cheap)
img = Image.open(IMG_PATH).convert("RGB")
m = max(img.size)
if m > 512:
    scale = 512.0 / m
    img = img.resize((max(1, int(img.size[0]*scale)), max(1, int(img.size[1]*scale))), Image.Resampling.LANCZOS)

print("[bench] processor.process …")
t0 = time.time()
batch = build_batch(img, PROMPT)
print(f"[bench] process+normalize took {time.time()-t0:.2f}s")
# Debug: print critical tensor shapes (uncomment if needed)
for k in ("images", "pixel_values", "image_input_idx", "image_masks"):
    if k in batch and isinstance(batch[k], torch.Tensor):
        print(f"  {k}: {tuple(batch[k].shape)}  dtype={batch[k].dtype}  device={batch[k].device}")

print("[bench] generate …")
t0 = time.time()
with torch.inference_mode():
    if hasattr(model, "generate_from_batch"):
        out = model.generate_from_batch(batch, gen, tokenizer=tok, use_cache=False)
    else:
        out = model.generate(**batch, generation_config=gen, use_cache=False)
dt = time.time() - t0
print(f"[bench] generate took {dt:.2f}s")

# Decode
def decode(gen_out):
    if isinstance(gen_out, dict) and "sequences" in gen_out and isinstance(gen_out["sequences"], torch.Tensor):
        seq = gen_out["sequences"]
        if tok and hasattr(tok, "batch_decode"):
            return tok.batch_decode(seq, skip_special_tokens=True)[0].strip()
        return str(seq.tolist())
    if hasattr(gen_out, "sequences"):
        seq = gen_out.sequences
        if tok and hasattr(tok, "batch_decode") and isinstance(seq, torch.Tensor):
            return tok.batch_decode(seq, skip_special_tokens=True)[0].strip()
        return str(seq)
    if isinstance(gen_out, torch.Tensor):
        if tok and hasattr(tok, "batch_decode"):
            return tok.batch_decode(gen_out, skip_special_tokens=True)[0].strip()
        return str(gen_out.tolist())
    if isinstance(gen_out, str):
        return gen_out.strip()
    return str(gen_out)

txt = decode(out)
print("\n--- CAPTION ---")
print(txt)
print("---------------")
