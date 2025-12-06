#Activate whichever VLM you want to test at :8000 first
#this script tests VLM discriminative abilities on 2 images concatenated together
#call this script followed by the paths of the two images (eg. python compare_two_images.py ../images/me_01.png ../images/random1.png)
#output is printed to console

import os
import sys
import io
import time
import base64
from typing import Tuple, Optional

import cv2
import numpy as np
import requests
from PIL import Image

MOLMO_URL = os.getenv("MOLMO_URL", "http://localhost:8000/caption")

'''
PROMPT = (
    "You will see one image that contains two views side by side: LEFT and RIGHT.\n"
    "Your ONLY task is to decide whether LEFT and RIGHT show the SAME REAL-WORLD PERSON.\n"
    "\n"
    "Rules:\n"
    "- Focus only on the person's identity (face structure, features).\n"
    "- Ignore pose, angle, lighting, background, clothing and accessories.\n"
    "- If you are even slightly unsure, you MUST answer 'different person'.\n"
    "- Do NOT explain your reasoning.\n"
    "\n"
    "Answer with EXACTLY ONE LINE, with no extra words:\n"
    "same person\n"
    "or\n"
    "different person"
)
'''

PROMPT = (
    "Look at the features of the two persons and determine if they are the same person or not. "
    "Ignore environmental changes like lighting, colour of the clothes, and poses. "
    "Only compare the facial features such as the eyes, eyebrows, face shape, nose, mouth, ears, hair. "
    "Describe the two faces"
)

PANEL_SIZE = (384, 256)
GAP = 8
TIMEOUT_S = 80.0


def _encode_b64(frame_bgr: np.ndarray, q: int = 90) -> str:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _letterbox(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    th, tw = size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def compose_pair(left_bgr: np.ndarray,
                 right_bgr: np.ndarray,
                 panel_size: Tuple[int, int] = PANEL_SIZE,
                 gap: int = GAP) -> np.ndarray:
    L = _letterbox(left_bgr, panel_size)
    R = _letterbox(right_bgr, panel_size)
    H = max(L.shape[0], R.shape[0])
    W = L.shape[1] + gap + R.shape[1]
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:L.shape[0], :L.shape[1]] = L
    out[:R.shape[0], L.shape[1] + gap : L.shape[1] + gap + R.shape[1]] = R
    return out


def call_molmo_same_diff(pair_bgr: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
    payload = {
        "image_b64": _encode_b64(pair_bgr, q=90),
        "prompt": PROMPT,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(MOLMO_URL, json=payload, timeout=TIMEOUT_S)
    except Exception as e:
        print(f"[error] request to Molmo failed: {e}")
        return None, None

    dt_ms = (time.perf_counter() - t0) * 1000.0

    try:
        j = resp.json()
        # Expect Molmo app to return its reply in "caption" or "text"
        raw = (j.get("caption") or j.get("text") or "").strip()
    except Exception:
        raw = resp.text.strip()

    print(f"[debug] HTTP {resp.status_code}, latency={dt_ms:.1f} ms")
    print(f"[debug] raw reply: {raw!r}")

    if resp.status_code != 200:
        return None, dt_ms

    return raw, dt_ms


def test_single_pair(left_path: str, right_path: str):
    if not os.path.exists(left_path):
        print(f"[error] left image not found: {left_path}")
        return
    if not os.path.exists(right_path):
        print(f"[error] right image not found: {right_path}")
        return

    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None:
        print(f"[error] cv2 failed to read left image: {left_path}")
        return
    if right is None:
        print(f"[error] cv2 failed to read right image: {right_path}")
        return

    pair = compose_pair(left, right)

    cv2.imshow("Molmo pair (LEFT | RIGHT)", pair)
    cv2.waitKey(1)

    raw, dt_ms = call_molmo_same_diff(pair)

    print("────────────────────────────────────────")
    print("Molmo raw reply:")
    print(raw)
    print("────────────────────────────────────────")
    if dt_ms is not None:
        print(f"Latency: {dt_ms:.1f} ms")
    print("────────────────────────────────────────")
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_images.py left.jpg right.jpg")
        return
    left_path = sys.argv[1]
    right_path = sys.argv[2]
    test_single_pair(left_path, right_path)


if __name__ == "__main__":
    main()
