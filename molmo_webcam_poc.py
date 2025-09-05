import os
# ---- Offline + perf env (set BEFORE imports) ----
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_DSHOW", "1000")

import re, time
from pathlib import Path
from threading import Thread, Event
import cv2, torch, numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# ----------------- Config -----------------
MODEL_DIR   = r"C:\models\MolmoE-1B-0924"   # local-only
OFFLOAD_DIR = r"C:\offload_molmo"           # SSD folder
WIN_NAME    = "Molmo 1B Webcam"
WIN_W, WIN_H = 960, 720

IMAGE_MAX_SIDE = 512    # 448 if you want even faster
MAX_CROPS      = 2      # 1 = fastest; 2 = good balance

POINT_TAG_RE = re.compile(r'<point[^>]*x="([\d.]+)"\s*y="([\d.]+)"[^>]*>(.*?)</point>', re.IGNORECASE)

# ===== CUDA diag =====
def print_cuda_diag(tag):
    print(f"\n=== CUDA diag ({tag}) ===")
    print("torch:", torch.__version__, "| cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory/(1024**3):.2f} GiB")
    print("=========================\n")

print_cuda_diag("startup")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ===== Camera & utils =====
def open_camera(index=0, width=640, height=480, fps=30, timeout_sec=5):
    start = time.monotonic()
    for be in [getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
               getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
               cv2.CAP_ANY]:
        try:
            cap = cv2.VideoCapture(index, be)
            while time.monotonic()-start < timeout_sec and not cap.isOpened():
                time.sleep(0.05)
            if not cap.isOpened():
                cap.release(); continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception: pass
            try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception: pass
            cap.set(cv2.CAP_PROP_FPS, fps)
            for _ in range(3): cap.read()
            return cap
        except Exception:
            try: cap.release()
            except Exception: pass
    raise RuntimeError("Could not open camera.")

def downscale_for_model(bgr, max_side=512):
    h, w = bgr.shape[:2]
    s = min(1.0, float(max_side)/max(h, w))
    return cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1.0 else bgr

def letterbox_resize(bgr, target_w, target_h, color=(0,0,0)):
    h, w = bgr.shape[:2]
    scale = min(target_w/w, target_h/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), color, np.uint8)
    x0 = (target_w-new_w)//2; y0 = (target_h-new_h)//2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

def to_pil(bgr): return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
def ensure_dirs(): Path(MODEL_DIR).mkdir(parents=True, exist_ok=True); Path(OFFLOAD_DIR).mkdir(parents=True, exist_ok=True)

# ===== Device & dtype =====
def _best_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

def _model_device(model):
    for p in model.parameters():
        return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Robust input prep (fixes NoneType) =====
def _finalize_masks(batch, tokenizer):
    if "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
        if ("attention_mask" not in batch) or (batch["attention_mask"] is None):
            pad_id = getattr(tokenizer, "pad_token_id", 0)
            batch["attention_mask"] = (batch["input_ids"] != pad_id).to(torch.long)
    return batch

def _prep_inputs(processor, model, pil_img, text):
    raw = processor.process(
        images=[pil_img],
        text=text,
        images_kwargs={"max_crops": MAX_CROPS},
    )
    for k, v in list(raw.items()):
        if isinstance(v, (list, tuple, np.ndarray)):
            raw[k] = torch.as_tensor(v)
    raw = _finalize_masks(raw, processor.tokenizer)
    dev = _model_device(model)
    out = {}
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0: v = v.unsqueeze(0)
            if v.ndim == 1: v = v.unsqueeze(0)
            out[k] = v.to(dev, non_blocking=True)
        else:
            out[k] = v
    return out

# ===== Generation =====
def _extract_sequences(out):
    if hasattr(out, "sequences"):
        seq = out.sequences
    elif isinstance(out, torch.Tensor):
        seq = out
    elif isinstance(out, dict):
        seq = out.get("sequences") if isinstance(out.get("sequences"), torch.Tensor) else None
        if seq is None and "tokens" in out:
            t = out["tokens"]; seq = t if isinstance(t, torch.Tensor) else torch.tensor(t)
    elif isinstance(out, (list, tuple)):
        first = out[0]; seq = first if isinstance(first, torch.Tensor) else torch.tensor(out)
    else:
        seq = None
    return seq.unsqueeze(0) if (isinstance(seq, torch.Tensor) and seq.ndim == 1) else seq

def _generate(processor, model, inputs, prompt_text=None, max_new_tokens=48):
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, stop_strings=["<|endoftext|>"])
    out = model.generate_from_batch(inputs, gen_cfg, tokenizer=processor.tokenizer)
    seq = _extract_sequences(out)
    if seq is None:
        return str(out)
    text = processor.tokenizer.batch_decode(seq, skip_special_tokens=True)[0].strip()
    if prompt_text and text.lower().startswith(prompt_text.lower()):
        text = text[len(prompt_text):].lstrip()
    return text

@torch.inference_mode()
def caption_frame(processor, model, frame_bgr, prompt="Describe this image in one sentence."):
    pil = to_pil(frame_bgr)
    return _generate(processor, model, _prep_inputs(processor, model, pil, prompt), prompt_text=prompt, max_new_tokens=48)

@torch.inference_mode()
def point_and_caption(processor, model, frame_bgr):
    pil = to_pil(frame_bgr)
    prompt = (
        "point_qa: Point to the single most salient object, then give a 6–12 word caption.\n"
        'Respond EXACTLY as:\n'
        '<point x="XX.x" y="YY.y">label</point>\n'
        'caption text\n'
        "Coordinates must be normalized in [0,100] with (0,0) at top-left."
    )
    return _generate(processor, model, _prep_inputs(processor, model, pil, prompt), prompt_text=prompt, max_new_tokens=64)

def parse_points(text):
    pts = []
    for m in POINT_TAG_RE.finditer(text):
        try: pts.append((float(m.group(1)), float(m.group(2)), m.group(3).strip()))
        except Exception: continue
    return pts

def draw_annotation(bgr, points, caption=None):
    h, w = bgr.shape[:2]
    canvas = bgr.copy()
    for (xn, yn, label) in points:
        x = int(np.clip(xn,0,100)/100.0*w); y = int(np.clip(yn,0,100)/100.0*h)
        cv2.circle(canvas, (x,y), 8, (0,255,0), -1)
        cv2.putText(canvas, label or "point", (x+10, max(20, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    if caption:
        y0=30
        for i, line in enumerate(_wrap_text(caption, 48)):
            cv2.putText(canvas, line, (10, y0+i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return canvas

def _wrap_text(s, maxw=48):
    words=s.split(); lines=[]; cur=""
    for w in words:
        if len(cur)+len(w)+1 <= maxw: cur=(cur+" "+w).strip()
        else: lines.append(cur); cur=w
    if cur: lines.append(cur); return lines

# ===== Loader: GPU-first, local-only =====
def _load_local_processor():
    return AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)

def _load_gpu_first():
    # Prefer safetensors if our one-time prep created it
    use_st = (Path(MODEL_DIR) / "model.safetensors").exists()
    dtype = _best_dtype()
    print(f"→ Loading model ({'safetensors' if use_st else 'bin'}) local-only, dtype={dtype} …")
    # Fastest path: load on CPU RAM without low_cpu_mem_usage, then move to GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,     # faster single-shot load
        use_safetensors=use_st
    )
    model.to("cuda")
    torch.cuda.synchronize()
    return model

def _load_offload_fallback():
    dtype = _best_dtype()
    print("↪ Using accelerate auto-offload (GPU+CPU+SSD)…")
    max_memory = {"cpu": "64GiB"}
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        max_memory.update({0: f"{max(1, total-1)}GiB"})  # leave ~1 GiB headroom
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=OFFLOAD_DIR,
        low_cpu_mem_usage=True,
    )
    return model

def load_model_blocking(_safe_prep_fn):
    ensure_dirs()
    if not (Path(MODEL_DIR) / "config.json").exists():
        raise FileNotFoundError(f"Model files not found in {MODEL_DIR}. Run the one-time prep first.")

    processor = _load_local_processor()
    try:
        model = _load_gpu_first()
        print("✅ Model device:", _model_device(model))
    except RuntimeError as e:
        if "CUDA out of memory" not in str(e):
            raise
        print("⚠️ OOM on full-GPU; falling back.")
        model = _load_offload_fallback()

    # warm-up
    try:
        with torch.inference_mode():
            dummy = Image.new("RGB", (64, 64), "gray")
            warm = _safe_prep_fn(processor, model, dummy, "hi")
            _ = model.generate_from_batch(
                warm,
                GenerationConfig(max_new_tokens=1, stop_strings=["<|endoftext|>"]),
                tokenizer=processor.tokenizer,
            )
            if torch.cuda.is_available(): torch.cuda.synchronize()
    except Exception as e:
        print("ℹ️ Warm-up skipped:", repr(e))

    print_cuda_diag("post-load")
    try: print("[hf_device_map]", getattr(model, "hf_device_map", None))
    except Exception: pass
    return processor, model

# ===== Main =====
def main():
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)
    try:
        cap = open_camera(0, 640, 480, 30, 5)
    except Exception as e:
        print("❌ Camera error:", e); return
    print("📷 Camera ready.")

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
    print("⏳ Loading Molmo model (local-only)…")

    last_caption, last_points = "", []
    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Camera read failed."); break

        display = frame.copy()
        hud = ("Loading model…" if not ready.is_set() else
               (f"Model load failed: {type(result['error']).__name__}" if result["error"] else
                "C: caption   P: point+caption   Q/Esc: quit"))
        cv2.putText(display, hud, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        if last_caption or last_points:
            display = draw_annotation(display, last_points, caption=last_caption)

        cv2.imshow(WIN_NAME, letterbox_resize(display, WIN_W, WIN_H))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break

        if ready.is_set() and result["error"] is None:
            processor, model = result["processor"], result["model"]
            small = downscale_for_model(frame, IMAGE_MAX_SIDE)

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
                    if not pts: print("ℹ️ No <point …> found; showing raw text.")
                except Exception as e:
                    print("❌ Point error:", repr(e))

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
