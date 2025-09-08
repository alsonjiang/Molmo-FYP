import os, time, torch, cv2
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

LOCAL_DIR = Path.cwd() / "MolmoE-1B-0924-NF4"  # your local snapshot folder

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
    """Move tensors to device; cast image tensors to the vision dtype; add cache_position."""
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

@torch.inference_mode()
def molmo_caption(model, processor, device, vision_dtype, pil_image, prompt: str, max_new_tokens: int = 48):
    """Greedy decode (manual) for robustness; returns plain text."""
    batch = processor.process(images=[pil_image], text=prompt)
    batch = move_and_fix_dtypes(batch, device, vision_dtype)

    eos_id = getattr(processor.tokenizer, "eos_token_id", None)
    start_len = batch["input_ids"].shape[1]

    # Greedy loop
    for _ in range(max_new_tokens):
        outputs = model(**batch, use_cache=False, return_dict=True)
        next_id = outputs.logits[:, -1, :].argmax(dim=-1)  # greedy
        batch["input_ids"] = torch.cat([batch["input_ids"], next_id.unsqueeze(1)], dim=1)
        T = batch["input_ids"].shape[1]
        batch["cache_position"] = torch.arange(T, device=device).unsqueeze(0)
        if eos_id is not None and int(next_id[0].item()) == int(eos_id):
            break

    gen_tokens = batch["input_ids"][0, start_len:]
    text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return text

def main():
    if not LOCAL_DIR.exists():
        fail(f"Model folder not found: {LOCAL_DIR}")

    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    torch.set_float32_matmul_precision("high")

    # 1) Load processor & model (offline)
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
    model.config.use_cache = False  # stick to stateless path
    vision_dtype = guess_vision_dtype(model)
    print("vision dtype:", vision_dtype)

    # 2) Optional quick smoke test with a gray image
    smoke = Image.new("RGB", (320, 240), "gray")
    txt = molmo_caption(model, processor, device, vision_dtype, smoke, "Describe this image briefly.")
    print("smoke test >>>", txt)

    # 3) OpenCV webcam loop
    cam_index = 0                       # change if you have multiple cameras
    infer_every_s = 2.0                 # run Molmo at most once every N seconds
    target_width = 640                  # capture width hint
    target_height = 480                 # capture height hint
    prompt = "Describe this scene briefly."

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # CAP_DSHOW on Windows to avoid MSMF slowness
    if not cap.isOpened():
        fail(f"Cannot open camera index {cam_index}")

    # Try to set a modest resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    last_infer = 0.0
    last_caption = "(initializing...)"

    print("Press [c] to force a caption, [q] to quit.")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[warn] Failed to read frame")
                continue

            # Show last caption overlaid
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 40), (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)
            cv2.putText(frame_bgr, last_caption[:110], (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Molmo Webcam", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            now = time.time()
            trigger = (now - last_infer) >= infer_every_s or key == ord('c')

            if trigger:
                # Convert to RGB PIL for the processor
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                t0 = time.time()
                try:
                    cap_text = molmo_caption(
                        model, processor, device, vision_dtype, pil_img, prompt, max_new_tokens=48
                    )
                    dt = time.time() - t0
                    last_caption = f"{cap_text}  [{dt:.2f}s]"
                    print(">>>", last_caption)
                except torch.cuda.OutOfMemoryError:
                    last_caption = "OOM during inference. Reduce resolution or tokens."
                    print("[error] CUDA OOM. Try lower resolution or max_new_tokens.")
                except Exception as e:
                    last_caption = f"Error: {e.__class__.__name__}"
                    print("[error]", e)

                last_infer = now

            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
