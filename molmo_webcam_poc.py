import os
# ---- Set env BEFORE importing cv2/transformers ----
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Prefer DirectShow; avoid sluggish MSMF on Windows
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_DSHOW", "1000")

import re
import time
from pathlib import Path
from threading import Thread, Event

import cv2
import torch
import numpy as np
from PIL import Image

# (Optional) small perf boost on NVIDIA
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from huggingface_hub import snapshot_download

# ----------------- Config -----------------
MODEL_ID    = "allenai/MolmoE-1B-0924"
LOCAL_DIR   = r"C:\FYP\models\MolmoE-1B-0924"
OFFLOAD_DIR = r"C:\FYP\offload_molmo"

WIN_NAME = "Molmo 1B Webcam"
WIN_W, WIN_H = 960, 720  # fixed window size

POINT_TAG_RE = re.compile(
    r'<point[^>]*x="([\d.]+)"\s*y="([\d.]+)"[^>]*>(.*?)</point>',
    re.IGNORECASE
)
# -------------------------------------------

def open_camera(index=0, width=640, height=480, fps=30, timeout_sec=5):
    """Prefer DirectShow, MJPG; small warm-up; bail fast if it can't open."""
    start = time.monotonic()
    backends = []
    if hasattr(cv2, "CAP_DSHOW"): backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):  backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)

    last_err = None
    for be in backends:
        try:
            cap = cv2.VideoCapture(index, be)
            while time.monotonic() - start < timeout_sec and not cap.isOpened():
                time.sleep(0.05)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FPS, fps)

            for _ in range(3):
                cap.read()
            return cap
        except Exception as e:
            last_err = e
            try: cap.release()
            except Exception: pass
    raise RuntimeError(f"Could not open camera within {timeout_sec}s. Last error: {last_err}")

def downscale_for_model(frame_bgr, max_side=512):
    """Limit longest side to <= max_side (faster preproc & less VRAM)."""
    h, w = frame_bgr.shape[:2]
    s = min(1.0, float(max_side) / max(h, w))
    if s < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return frame_bgr

def letterbox_resize(img_bgr, target_w, target_h, color=(0, 0, 0)):
    """Resize with padding to exactly (target_w, target_h) while keeping aspect."""
    h, w = img_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = color
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas

def to_pil(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def ensure_dirs():
    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)
    Path(OFFLOAD_DIR).mkdir(parents=True, exist_ok=True)

def load_model_blocking(_safe_prep_fn):
    """Downloads + loads Molmo; ties weights; warms up 1 token using the same safe prep as inference."""
    ensure_dirs()
    resolved_dir = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(resolved_dir, trust_remote_code=True)

    # Cap VRAM usage (int keys for GPU indices)
    max_memory = {"cpu": "48GiB"}
    if torch.cuda.is_available():
        max_memory.update({i: "2.8GiB" for i in range(torch.cuda.device_count())})

    model = AutoModelForCausalLM.from_pretrained(
        resolved_dir,
        trust_remote_code=True,
        use_safetensors=False,     # MolmoE-1B ships .bin
        device_map="auto",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        offload_folder=OFFLOAD_DIR,
        offload_state_dict=True,
        max_memory=max_memory,
    )

    # Tie weights (silence warning)
    try:
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    except Exception as e:
        print("ℹ️ tie_weights skipped:", repr(e))

    # Warm-up lightweight gen to trigger sharding/offload (using safe prep)
    try:
        with torch.inference_mode():
            dummy_img = Image.new("RGB", (64, 64), "gray")
            warm_inputs = _safe_prep_fn(processor, model, dummy_img, "hi")
            warm_cfg = GenerationConfig(max_new_tokens=1)
            _ = model.generate_from_batch(warm_inputs, warm_cfg, tokenizer=processor.tokenizer)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    except Exception as e:
        print("ℹ️ Warm-up skipped:", repr(e))

    try:
        print("[hf_device_map]", getattr(model, "hf_device_map", None))
    except Exception:
        pass

    return processor, model

def _prep_inputs(processor, model, pil_img, text):
    """Move only tensors to device, add batch dim; skip None; keep metadata."""
    raw = processor.process(images=[pil_img], text=text)
    prepped = {}
    for k, v in raw.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            prepped[k] = v.to(model.device).unsqueeze(0)
        else:
            prepped[k] = v
    return prepped

def _extract_sequences(out):
    """Normalize any return type to a (batch, seq_len) int tensor if possible."""
    seq = None
    if hasattr(out, "sequences"):
        seq = out.sequences
    elif isinstance(out, torch.Tensor):
        seq = out
    elif isinstance(out, dict):
        if isinstance(out.get("sequences"), torch.Tensor):
            seq = out["sequences"]
        elif "tokens" in out:
            t = out["tokens"]
            seq = t if isinstance(t, torch.Tensor) else torch.tensor(t)
    elif isinstance(out, (list, tuple)):
        first = out[0]
        seq = first if isinstance(first, torch.Tensor) else torch.tensor(out)

    if seq is None:
        return None
    if seq.ndim == 1:
        seq = seq.unsqueeze(0)
    return seq

def _generate(processor, model, inputs, prompt_text=None, max_new_tokens=32):
    """Robust decode via batch_decode; no reliance on input_ids size."""
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens)
    out = model.generate_from_batch(inputs, gen_cfg, tokenizer=processor.tokenizer)

    seq = _extract_sequences(out)
    if seq is None:
        return str(out)

    # Decode full sequence; trimming prompt text in string space (safer than token slicing).
    text = processor.tokenizer.batch_decode(seq, skip_special_tokens=True)[0].strip()
    if prompt_text:
        # Remove prompt prefix if the generator echoed it
        if text.lower().startswith(prompt_text.lower()):
            text = text[len(prompt_text):].lstrip()
    return text

@torch.inference_mode()
def caption_frame(processor, model, frame_bgr, prompt="Describe this image in one sentence."):
    pil_img = to_pil(frame_bgr)
    inputs = _prep_inputs(processor, model, pil_img, prompt)
    return _generate(processor, model, inputs, prompt_text=prompt, max_new_tokens=32)

@torch.inference_mode()
def point_and_caption(processor, model, frame_bgr):
    pil_img = to_pil(frame_bgr)
    prompt = (
        "Point to the single most salient object, then give a 6-12 word caption.\n"
        "Respond EXACTLY as:\n"
        '<point x="XX.x" y="YY.y">label</point>\n'
        "caption text\n"
        "Coordinates must be normalized in [0,100] with (0,0) at top-left."
    )
    inputs = _prep_inputs(processor, model, pil_img, prompt)
    return _generate(processor, model, inputs, prompt_text=prompt, max_new_tokens=32)

def parse_points(text):
    pts = []
    for m in POINT_TAG_RE.finditer(text):
        try:
            x = float(m.group(1)); y = float(m.group(2)); label = m.group(3).strip()
            pts.append((x, y, label))
        except Exception:
            continue
    return pts

def draw_annotation(frame_bgr, points, caption=None):
    h, w = frame_bgr.shape[:2]
    canvas = frame_bgr.copy()
    for (xn, yn, label) in points:
        x = int(np.clip(xn, 0, 100) / 100.0 * w)
        y = int(np.clip(yn, 0, 100) / 100.0 * h)
        cv2.circle(canvas, (x, y), 8, (0, 255, 0), thickness=-1)
        text = label if label else "point"
        cv2.putText(canvas, text, (x + 10, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if caption:
        wrapped = wrap_text(caption, max_width=48)
        y0 = 30
        for i, line in enumerate(wrapped):
            cv2.putText(canvas, line, (10, y0 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas

def wrap_text(s, max_width=48):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def main():
    # 1) Show window & camera immediately (fast feedback)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)

    try:
        cap = open_camera(index=0, width=640, height=480, fps=30, timeout_sec=5)
    except Exception as e:
        print("❌ Camera error:", e)
        return
    print("📷 Camera ready.")

    # 2) Load model in background (UI stays responsive)
    ready = Event()
    result = {"processor": None, "model": None, "error": None}
    def _worker():
        try:
            p, m = load_model_blocking(_prep_inputs)
            result["processor"], result["model"] = p, m
        except Exception as ex:
            result["error"] = ex
        finally:
            ready.set()
    Thread(target=_worker, daemon=True).start()
    print("⏳ Loading Molmo model in background…")

    last_caption, last_points = "", []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Failed to read frame from camera."); break

        # Compose display
        display = frame.copy()
        if not ready.is_set():
            hud = "Loading model… please wait"
        elif result["error"] is not None:
            hud = f"Model load failed: {type(result['error']).__name__}"
        else:
            hud = "C: caption   P: point+caption   Q/Esc: quit"
        cv2.putText(display, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if last_caption or last_points:
            display = draw_annotation(display, last_points, caption=last_caption)

        # Fixed-size window render
        display_out = letterbox_resize(display, WIN_W, WIN_H)
        cv2.imshow(WIN_NAME, display_out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # 'q' or ESC
            break

        # Enable keys only once model is ready
        if ready.is_set() and result["error"] is None:
            processor, model = result["processor"], result["model"]
            # Downscale before model for speed
            small = downscale_for_model(frame, max_side=512)

            if key == ord('c'):
                print("⏳ Captioning…")
                try:
                    last_caption = caption_frame(processor, model, small)
                    last_points = []
                    print("📝", last_caption)
                except Exception as e:
                    print("❌ Caption error:", repr(e))
            elif key == ord('p'):
                print("⏳ Pointing + caption…")
                try:
                    text = point_and_caption(processor, model, small)
                    pts = parse_points(text)
                    caption = text.split("\n", 1)[1].strip() if "\n" in text else None
                    last_points = pts
                    last_caption = caption or text
                    print("🧭 raw:", text)
                    if not pts:
                        print("ℹ️  No <point …> found; showing raw text.")
                except Exception as e:
                    print("❌ Point error:", repr(e))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
