# file: nf4_local_bulletproof.py
import os, time, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# ── CONFIG: point at your already-downloaded local NF4 folder ────────────────
LOCAL_DIR = Path.cwd() / "MolmoE-1B-0924-NF4"  # e.g. C:\Molmo-FYP\MolmoE-1B-0924-NF4

# Quiet TF spam (optional) and keep torchvision enabled
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("TRANSFORMERS_NO_TORCHVISION", None)

def fail(msg): raise SystemExit(f"[fatal] {msg}")

def guess_vision_dtype(model: torch.nn.Module) -> torch.dtype:
    """Find a floating-point param inside the vision backbone to infer dtype."""
    vb = getattr(model, "vision_backbone", None)
    if vb is not None:
        for _, p in vb.named_parameters():
            if p.is_floating_point():
                return p.dtype
    # Fallback
    return torch.float16 if torch.cuda.is_available() else torch.float32

def move_and_fix_dtypes(batch: dict, device: torch.device, vision_dtype: torch.dtype):
    """Move tensors to device; cast image tensors to the vision dtype."""
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
    # Provide cache_position if missing (some forks expect it)
    if "input_ids" in out and "cache_position" not in out:
        T = out["input_ids"].shape[1]
        out["cache_position"] = torch.arange(T, device=device).unsqueeze(0)
    return out

def main():
    if not LOCAL_DIR.exists():
        fail(f"Model folder not found: {LOCAL_DIR}")

    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    torch.set_float32_matmul_precision("high")

    # 1) Processor & model from local (offline)
    processor = AutoProcessor.from_pretrained(str(LOCAL_DIR), trust_remote_code=True, local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(LOCAL_DIR),
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"},
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        if "bitsandbytes" in str(e).lower():
            fail("bitsandbytes not found. Install it:  pip install -U bitsandbytes")
        raise

    device = next(model.parameters()).device
    model.config.use_cache = False  # avoid fork’s cache path entirely

    # 2) Infer the vision dtype and show it
    vision_dtype = guess_vision_dtype(model)
    print("vision dtype:", vision_dtype)

    # 3) Offline test image
    img = Image.new("RGB", (320, 240), "gray")

    # 4) Preprocess -> move to device -> CAST images to vision dtype
    batch = processor.process(images=[img], text="Describe this image briefly.")
    batch = move_and_fix_dtypes(batch, device, vision_dtype)

    # 5) Manual greedy decode (robust; no .generate* calls)
    max_new_tokens = 64
    eos_id = getattr(processor.tokenizer, "eos_token_id", None)
    start_len = batch["input_ids"].shape[1]

    for _ in range(max_new_tokens):
        outputs = model(**batch, use_cache=False, return_dict=True)
        next_id = outputs.logits[:, -1, :].argmax(dim=-1)  # greedy
        batch["input_ids"] = torch.cat([batch["input_ids"], next_id.unsqueeze(1)], dim=1)
        # keep cache_position in sync (harmless if unused)
        T = batch["input_ids"].shape[1]
        batch["cache_position"] = torch.arange(T, device=device).unsqueeze(0)
        if eos_id is not None and int(next_id[0].item()) == int(eos_id):
            break

    # 6) Decode only the new tokens
    gen_tokens = batch["input_ids"][0, start_len:]
    text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    print(">>>", text)

if __name__ == "__main__":
    main()

#expected output:
#The image displays a solid gray square with no visible content or features. 
#It appears to be a plain, unadorned gray square without any discernible elements, patterns, or variations in color or texture.