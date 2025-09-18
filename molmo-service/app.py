import os, base64, io, torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# -------- Paths --------
LOCAL_DIR = (Path(__file__).resolve().parents[1] / "MolmoE-1B-0924-NF4").resolve()
OFFLOAD_DIR = (Path(__file__).resolve().parent / "offload").resolve()
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -------- Env (quiet TF) --------
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

print(f"[molmo-service] Using model folder: {LOCAL_DIR}")
if not LOCAL_DIR.exists():
    raise FileNotFoundError(f"MOLMO_LOCAL_DIR not found: {LOCAL_DIR}")

LOCAL_DIR_STR = LOCAL_DIR.as_posix()
OFFLOAD_DIR_STR = OFFLOAD_DIR.as_posix()

# -------- Load model/processor --------
processor = AutoProcessor.from_pretrained(
    LOCAL_DIR_STR, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_DIR_STR,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype="auto",
    device_map="auto",
    offload_folder=OFFLOAD_DIR_STR,
)
model.eval()

app = FastAPI()

# -------- Schemas --------
class CaptionIn(BaseModel):
    image_b64: str
    prompt: str = "Describe the image."

# -------- Helpers --------
def _prep_batch(pil_image: Image.Image, prompt: str):
    """Processor -> move tensors to model device -> add batch dim."""
    batch = processor.process(images=[pil_image], text=prompt)
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(model.device).unsqueeze(0)
    return batch

# -------- Routes --------
@app.get("/health")
def health():
    return {"ok": True, "local_dir": str(LOCAL_DIR)}

@app.post("/caption")
def caption(inp: CaptionIn):
    try:
        img = Image.open(io.BytesIO(base64.b64decode(inp.image_b64))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    batch = _prep_batch(img, inp.prompt)

    # Use standard generate (no stop_strings) to avoid tokenizer requirement errors
    with torch.inference_mode():
        gen_ids = model.generate(
            **batch,
            max_new_tokens=64,
            do_sample=False,   # greedy; set True for sampling if you like
        )

    text = processor.batch_decode(gen_ids)[0].strip()
    return {"caption": text}
