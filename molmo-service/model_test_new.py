import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
LOCAL_DIR = Path.cwd() / "MolmoE-1B-0924-NF4"
IMG_PATH = Path("test_images/clock_face.png")

# Use a chat-style prompt so the processor can mark Assistant tokens in response_mask
PROMPT = "Describe this image briefly."
MAX_NEW_TOKENS = 32

# Reduce TF verbosity; keep torchvision enabled
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.pop("TRANSFORMERS_NO_TORCHVISION", None)


def fail(msg: str):
    raise SystemExit(f"[fatal] {msg}")


def guess_vision_dtype(model: torch.nn.Module) -> torch.dtype:
    vb = getattr(model, "vision_backbone", None)
    if vb is not None:
        for _, p in vb.named_parameters():
            if p.is_floating_point():
                return p.dtype
    return torch.float16 if torch.cuda.is_available() else torch.float32


def move_and_fix_dtypes(batch: dict, device: torch.device, vision_dtype: torch.dtype):
    """
    Move tensors to the right device; cast only raw-pixel tensors to vision dtype.
    """
    out = {}
    for k, v in batch.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            if k in ("images", "pixel_values"):
                out[k] = v.to(device=device, dtype=vision_dtype)
            else:
                out[k] = v.to(device)
        else:
            out[k] = v
    return out


def normalize_batch_shapes(batch: dict):
    """
    Defensive shape fixes:
      - Text fields to [B, T] if they came as [T]
      - If raw pixels are under 'images', move to 'pixel_values'
      - Ensure aux vision tensors have batch dim
      - IMPORTANT: DO NOT drop response_mask/position_ids here
    """
    def ensure_2d(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0) if (isinstance(t, torch.Tensor) and t.dim() == 1) else t

    # Text-like tensors
    for k in ("input_ids", "attention_mask", "position_ids", "cache_position", "response_mask"):
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = ensure_2d(batch[k])

    # Vision: prefer raw pixels under 'pixel_values'; keep pretokenized as 'images'
    if "pixel_values" in batch and isinstance(batch["pixel_values"], torch.Tensor):
        pv = batch["pixel_values"]
        if pv.dim() == 3:  # [C,H,W] -> [1,C,H,W]
            batch["pixel_values"] = pv.unsqueeze(0)
    elif "images" in batch and isinstance(batch["images"], torch.Tensor):
        x = batch["images"]
        # If this looks like raw pixels -> move to 'pixel_values'
        if x.dim() == 4 and x.shape[1] in (1, 3):
            batch["pixel_values"] = x
            del batch["images"]
        elif x.dim() == 3 and x.shape[0] in (1, 3):  # [C,H,W]
            batch["pixel_values"] = x.unsqueeze(0)
            del batch["images"]
        elif x.dim() == 3:
            batch["images"] = x.unsqueeze(0)  # pre-tokenized [T,N,D] -> [1,T,N,D]
        elif x.dim() == 2:
            batch["images"] = x.unsqueeze(0).unsqueeze(0)  # [N,D] -> [1,1,N,D]

    # Aux vision tensors need a batch dim
    if "image_input_idx" in batch and isinstance(batch["image_input_idx"], torch.Tensor):
        if batch["image_input_idx"].dim() == 2:
            batch["image_input_idx"] = batch["image_input_idx"].unsqueeze(0)
    if "image_masks" in batch and isinstance(batch["image_masks"], torch.Tensor):
        if batch["image_masks"].dim() == 2:
            batch["image_masks"] = batch["image_masks"].unsqueeze(0)

    return batch


def ensure_nonempty_response_mask(batch: dict):
    """
    If response_mask exists but sums to 0 (no Assistant tokens yet),
    flip the last time-step to 1 so the first SDPA has a non-zero query length.
    """
    rm = batch.get("response_mask", None)
    if isinstance(rm, torch.Tensor) and rm.dim() == 2 and rm.numel() > 0:
        if int(rm.sum().item()) == 0:
            # Make the last position a query token
            if rm.dtype == torch.bool:
                rm[:, -1] = True
            else:
                rm[:, -1] = 1
            batch["response_mask"] = rm
    return batch


def main():
    if not LOCAL_DIR.exists():
        fail(f"Model folder not found: {LOCAL_DIR}")
    if not IMG_PATH.exists():
        fail(f"Image not found: {IMG_PATH}")

    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    torch.set_float32_matmul_precision("high")

    # Processor
    processor = AutoProcessor.from_pretrained(
        str(LOCAL_DIR), trust_remote_code=True, local_files_only=True
    )

    # Quantized 4-bit load
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(LOCAL_DIR),
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"},
        quantization_config=quantization_config,
    )

    device = next(model.parameters()).device
    model.eval()
    model.config.use_cache = True

    vision_dtype = guess_vision_dtype(model)
    print("vision dtype:", vision_dtype)

    # Load and cap image size to control VRAM
    img = Image.open(IMG_PATH).convert("RGB")
    target = None
    try:
        ip = getattr(processor, "image_processor", getattr(processor, "vision_processor", None))
        size = getattr(ip, "size", None)
        if isinstance(size, dict):
            if "shortest_edge" in size:
                target = int(size["shortest_edge"])
            elif "height" in size and "width" in size:
                target = max(int(size["height"]), int(size["width"]))
    except Exception:
        pass
    if target is None:
        target = 640
    w, h = img.size
    scale = target / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    # End-to-end timing start
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_e2e_start = time.perf_counter()

    # Preprocess (string prompt; this processor expects a str)
    batch = processor.process(images=[img], text=PROMPT)

    # Move to device/dtype and normalize shapes
    batch = move_and_fix_dtypes(batch, device, vision_dtype)
    batch = normalize_batch_shapes(batch)

    # --- Ensure masks exist and that the Assistant segment is non-empty ---
    B, T = batch["input_ids"].shape

    # attention_mask: default to all ones if missing
    if "attention_mask" not in batch or batch["attention_mask"] is None:
        batch["attention_mask"] = torch.ones((B, T), dtype=torch.long, device=device)

    # response_mask: create if missing OR fix if empty (sum==0)
    rm = batch.get("response_mask", None)
    if rm is None:
        # Create a boolean mask, mark the last token as Assistant query token
        rm = torch.zeros((B, T), dtype=torch.bool, device=device)
        rm[:, -1] = True
        batch["response_mask"] = rm
    else:
        # Normalize to [B, T] if needed
        if isinstance(rm, torch.Tensor) and rm.dim() == 1:
            rm = rm.unsqueeze(0)
        # If empty, flip last token on
        if int(rm.sum().item()) == 0:
            if rm.dtype == torch.bool:
                rm[:, -1] = True
            else:
                rm[:, -1] = 1
            batch["response_mask"] = rm


    # CRUCIAL: make sure we have at least one Assistant query token on first pass
    batch = ensure_nonempty_response_mask(batch)

    # (optional) one-shot shapes print for sanity — comment out after success
    print({k: tuple(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)})

    # --- decode-only timer start ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_dec_start = time.perf_counter()

    # Manual forward (avoid checkpoint-specific generate_from_batch quirks)
    with torch.inference_mode():
        # First pass: everything from the processor
        out = model(**batch, use_cache=True, return_dict=True)
        next_id = out.logits[:, -1, :].argmax(dim=-1)
        past = out.past_key_values

        generated = [next_id]
        for _ in range(MAX_NEW_TOKENS - 1):
            out = model(
                input_ids=generated[-1].unsqueeze(1),
                use_cache=True,
                past_key_values=past,
                return_dict=True,
            )
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1)
            generated.append(next_id)

        gen_tokens = torch.stack(generated, dim=1)[0].detach().cpu()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_dec_end = time.perf_counter()
    decode_time = t_dec_end - t_dec_start

    # Decode tokens
    text_out = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # End-to-end timing end
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_e2e_end = time.perf_counter()
    overall_time = t_e2e_end - t_e2e_start

    n_new = int(gen_tokens.numel())
    toks_per_s = (n_new / decode_time) if decode_time > 0 else float("inf")

    print(">>>", text_out)
    print(f"[timing] decode: {decode_time:.1f} s  |  tokens: {n_new}  |  {toks_per_s:.1f} tok/s")
    print(f"[timing] end-to-end: {overall_time:.1f} s")


if __name__ == "__main__":
    main()
