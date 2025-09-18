import os, base64, io, torch
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

LOCAL_DIR = (Path(__file__).resolve().parents[1] / "MolmoE-1B-0924-NF4").resolve()

OFFLOAD_DIR = (Path(__file__).resolve().parent / "offload").resolve()
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Optional: keep TF quiet / disabled
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

print(f"[molmo-service] Using model folder: {LOCAL_DIR}")
if not LOCAL_DIR.exists():
    raise FileNotFoundError(f"MOLMO_LOCAL_DIR not found: {LOCAL_DIR}")

# Use POSIX string to avoid backslash issues in some hub helpers
LOCAL_DIR_STR = LOCAL_DIR.as_posix()
OFFLOAD_DIR_STR = OFFLOAD_DIR.as_posix()

processor = AutoProcessor.from_pretrained(LOCAL_DIR_STR, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_DIR_STR, trust_remote_code=True, local_files_only=True,
    torch_dtype="auto", device_map="auto", offload_folder=OFFLOAD_DIR_STR
)

app = FastAPI()

class CaptionIn(BaseModel):
    image_b64: str
    prompt: str = "Describe the image."

@app.get("/health")
def health():
    return {"ok": True, "local_dir": LOCAL_DIR}

@app.post("/caption")
def caption(inp: CaptionIn):
    img = Image.open(io.BytesIO(base64.b64decode(inp.image_b64))).convert("RGB")
    batch = processor.process(images=[img], text=inp.prompt)
    batch = {k:(v.to(model.device).unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k,v in batch.items() if v is not None}
    gen = model.generate_from_batch(batch, GenerationConfig(max_new_tokens=64, stop_strings=["<|endoftext|>"]))
    text = processor.batch_decode(gen)[0]
    return {"caption": text}
