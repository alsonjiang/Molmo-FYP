import re
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image

import os
# ---- Runtime env tuning (do BEFORE importing transformers) ----
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# (Optional) small perf boost on NVIDIA
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from huggingface_hub import snapshot_download

MODEL_ID    = "allenai/MolmoE-1B-0924"
LOCAL_DIR   = r"C:\FYP\models\MolmoE-1B-0924"
OFFLOAD_DIR = r"C:\FYP\offload_molmo"

POINT_TAG_RE = re.compile(
    r'<point[^>]*x="([\d.]+)"\s*y="([\d.]+)"[^>]*>(.*?)</point>',
    re.IGNORECASE
)

def open_camera(index=0, width=640, height=480, fps=30):
    """
    Try faster Windows backends first (DirectShow), set MJPG, shrink frames, and warm up.
    """
    backends = []
    # Prefer DSHOW on Windows; fall back to MSMF, then ANY cross-platform
    if hasattr(cv2, "CAP_DSHOW"): backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):  backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)

    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if not cap.isOpened():
            continue
        # Smaller frames = faster preproc + less VRAM pressure
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Try to get MJPG to reduce CPU copy/convert cost on Windows webcams
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        # Reduce capture buffering if backend supports it
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # Try setting FPS (some drivers ignore)
        cap.set(cv2.CAP_PROP_FPS, fps)

        # Warm up a few frames so auto-exposure/white-balance settle
        for _ in range(5):
            cap.read()
        if cap.isOpened():
            return cap
        cap.release()
    return None


def downscale_for_model(frame_bgr, max_side=640):
    """
    Downscale the frame keeping aspect ratio so the longest side is <= max_side.
    Speeds up preproc and reduces VRAM without hurting a POC.
    """
    h, w = frame_bgr.shape[:2]
    s = min(1.0, float(max_side) / max(h, w))
    if s < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return frame_bgr


def to_pil(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def ensure_dirs():
    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)
    Path(OFFLOAD_DIR).mkdir(parents=True, exist_ok=True)

def load_model():
    ensure_dirs()

    resolved_dir = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(
        resolved_dir,
        trust_remote_code=True,
    )

    # Cap VRAM usage so accelerate shards more aggressively on 4GB GPUs
    max_memory = {"cpu": "48GiB"}
    if torch.cuda.is_available():
    # one entry per visible GPU, keys are integers: 0, 1, ...
        max_memory.update({i: "3GiB" for i in range(torch.cuda.device_count())})
    # e.g., on your laptop with a single GPU, this becomes {0: "3GiB", "cpu": "48GiB"}

    model = AutoModelForCausalLM.from_pretrained(
        resolved_dir,
        trust_remote_code=True,
        use_safetensors=False,          # MolmoE-1B ships .bin
        device_map="auto",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        offload_folder=OFFLOAD_DIR,
        offload_state_dict=True,
        max_memory=max_memory,
    )

    # 1) Explicitly tie weights to silence the warning
    try:
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    except Exception as e:
        print("ℹ️  tie_weights skipped:", repr(e))

    # 2) Warm-up: tiny forward/generation to load/shard/offload upfront
    try:
        with torch.inference_mode():
            dummy_img = Image.new("RGB", (64, 64), "gray")
            warm_inputs = processor.process(images=[dummy_img], text="hi")
            warm_inputs = {k: v.to(model.device).unsqueeze(0) for k, v in warm_inputs.items()}
            warm_cfg = GenerationConfig(max_new_tokens=1, temperature=0.0, top_p=1.0, stop_strings="<|endoftext|>")
            _ = model.generate_from_batch(warm_inputs, warm_cfg, tokenizer=processor.tokenizer)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    except Exception as e:
        print("ℹ️  Warm-up skipped:", repr(e))

    try:
        print("[hf_device_map]", getattr(model, "hf_device_map", None))
    except Exception:
        pass

    return processor, model

def _prep_inputs(processor, model, pil_img, text):
    """
    Robustly move only tensor inputs to the model device and add batch dim.
    Drop Nones or non-tensors (remote code may return extras).
    """
    raw = processor.process(images=[pil_img], text=text)
    prepped = {}
    for k, v in raw.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            prepped[k] = v.to(model.device).unsqueeze(0)
        else:
            # keep non-tensors as-is (some processors pass metadata)
            prepped[k] = v
    return prepped


def _generate(processor, model, inputs, max_new_tokens=96):
    """
    Works with both behaviors:
    - outputs include prompt+new tokens (need to slice)
    - outputs are only new tokens (offset=0)
    Also drops unsupported sampling args to avoid warnings.
    """
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        stop_strings="<|endoftext|>",
        # (Molmo's custom generate may ignore temperature/top_p)
    )

    output = model.generate_from_batch(
        inputs,
        gen_cfg,
        tokenizer=processor.tokenizer,
    )

    # Determine safe offset
    offset = 0
    inp_ids = inputs.get("input_ids", None)
    if isinstance(inp_ids, torch.Tensor) and inp_ids.ndim >= 2:
        offset = int(inp_ids.size(1))

    # output can be a tensor of token ids
    tokens = output[0]
    if offset > 0 and tokens.numel() >= offset:
        tokens = tokens[offset:]

    return processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()

@torch.inference_mode()
def caption_frame(processor, model, frame_bgr, prompt="Describe this image in one sentence."):
    pil_img = to_pil(frame_bgr)
    inputs = _prep_inputs(processor, model, pil_img, prompt)
    return _generate(processor, model, inputs, max_new_tokens=64)

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
    text = _generate(processor, model, inputs, max_new_tokens=96)
    return text

def parse_points(text):
    points = []
    for m in POINT_TAG_RE.finditer(text):
        try:
            x = float(m.group(1)); y = float(m.group(2))
            label = m.group(3).strip()
            points.append((x, y, label))
        except Exception:
            continue
    return points

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
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def main():
    processor, model = load_model()

    if torch.cuda.is_available():
        print(f"✅ CUDA: {torch.cuda.get_device_name(0)}  | dtype=fp16  | offload→{OFFLOAD_DIR}")
    else:
        print("⚠️  CUDA not detected — CPU will be slow.")

    cv2.namedWindow("Molmo 1B Webcam", cv2.WINDOW_NORMAL)

    # USE the faster open with smaller frames
    cap = open_camera(index=0, width=640, height=480, fps=30)
    if cap is None or not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✅ Ready. Keys: [C] caption, [P] point+caption, [Q/ESC] quit")

    last_caption = ""
    last_points = []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Failed to read frame from camera.")
            break

        # Downscale before sending to model to cut compute cost
        small = downscale_for_model(frame, max_side=640)

        display = frame.copy()
        hud = "C: caption   P: point+caption   Q/Esc: quit"
        cv2.putText(display, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if last_caption or last_points:
            display = draw_annotation(display, last_points, caption=last_caption)

        cv2.imshow("Molmo 1B Webcam", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            print("⏳ Captioning…")
            try:
                last_caption = caption_frame(processor, model, small)  # use downscaled
                last_points = []
                print("📝", last_caption)
            except Exception as e:
                print("❌ Caption error:", repr(e))
        elif key == ord('p'):
            print("⏳ Pointing + caption…")
            try:
                text = point_and_caption(processor, model, small)      # use downscaled
                pts = parse_points(text)
                caption = None
                if "\n" in text:
                    after = text.split("\n", 1)[1].strip()
                    caption = after if after else None
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
