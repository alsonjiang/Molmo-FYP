import os, time, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# --- Resolve paths ---
ROOT = Path(__file__).resolve().parents[1]     # ..\Molmo-FYP (repo root)
LOCAL_DIR = ROOT / "MolmoE-1B-0924-NF4"        # model folder in root
IMG_PATH = ROOT / "images" / "clock_face.png"  # image also in root
PROMPT = "Describe this image briefly."

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("TRANSFORMERS_NO_TORCHVISION", None)

def fail(msg): raise SystemExit(f"[fatal] {msg}")

def guess_vision_dtype(model: torch.nn.Module) -> torch.dtype:
    vb = getattr(model, "vision_backbone", None)
    if vb is not None:
        for _, p in vb.named_parameters():
            if p.is_floating_point():
                return p.dtype
    return torch.float16 if torch.cuda.is_available() else torch.float32

def move_and_fix_dtypes(batch: dict, device: torch.device, vision_dtype: torch.dtype):
    out = {}
    for k, v in batch.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            if k in ("images", "pixel_values"):
                out[k] = v.to(device=device, dtype=vision_dtype).unsqueeze(0)
            else:
                out[k] = v.to(device).unsqueeze(0)
        else:
            out[k] = v
    if "input_ids" in out and "cache_position" not in out:
        T = out["input_ids"].shape[1]
        out["cache_position"] = torch.arange(T, device=device).unsqueeze(0)
    return out

def main():
    if not LOCAL_DIR.exists():
        fail(f"Model folder not found: {LOCAL_DIR}")
    if not IMG_PATH.exists():
        fail(f"Test image not found: {IMG_PATH}")

    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    torch.set_float32_matmul_precision("high")

    # Load processor + model
    processor = AutoProcessor.from_pretrained(str(LOCAL_DIR), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(LOCAL_DIR),
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    device = next(model.parameters()).device
    vision_dtype = guess_vision_dtype(model)
    print("vision dtype:", vision_dtype)

    # Load test image from root
    img = Image.open(IMG_PATH).convert("RGB")

    # Preprocess
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()

    batch = processor.process(images=[img], text=PROMPT)
    batch = move_and_fix_dtypes(batch, device, vision_dtype)

    # Greedy decoding
    eos_id = getattr(processor.tokenizer, "eos_token_id", None)
    start_len = batch["input_ids"].shape[1]

    for _ in range(64):
        outputs = model(**batch, use_cache=False, return_dict=True)
        next_id = outputs.logits[:, -1, :].argmax(dim=-1)
        batch["input_ids"] = torch.cat([batch["input_ids"], next_id.unsqueeze(1)], dim=1)
        T = batch["input_ids"].shape[1]
        batch["cache_position"] = torch.arange(T, device=device).unsqueeze(0)
        if eos_id is not None and int(next_id[0].item()) == int(eos_id):
            break

    gen_tokens = batch["input_ids"][0, start_len:]
    text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    if device.type == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()

    print(">>>", text)
    print(f"[timing] total: {t1 - t0:.2f}s")

if __name__ == "__main__":
    main()
