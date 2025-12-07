#Activate whichever VLM you want to test at :8000 first
#this script takes images as defined below and does a comparison of every possible pair in the dataset.
#USER_IMAGES = images of the same person
#OTHER_IMAGES = images of other people
#its output will be saved in a csv in this folder, as defined by OUT_CSV

import os
import io
import time
import base64
import csv
from typing import Tuple, Optional, List, Dict
from itertools import combinations

import cv2
import numpy as np
import requests
from PIL import Image

# ================== CONFIG ==================

MOLMO_URL = os.getenv("MOLMO_URL", "http://localhost:8000/caption")

# EDIT THESE PATHS TO MATCH YOUR DATA
USER_IMAGES = [
    "../images/me_1.png",
    "../images/me_2.png",
    "../images/me_3.png",
]

OTHER_IMAGES = [
    "../images/random1.png",
    "../images/random2.png",
    "../images/random3.png",
    "../images/random4.png",
    "../images/random5.png",
    "../images/random6.png",
    "../images/random7.png",
    "../images/random8.png",
    "../images/random9.png",
    "../images/random10.png",
    "../images/random11.png",
    "../images/random12.png",
    "../images/random13.png",
    "../images/random14.png",
    "../images/random15.png",
    "../images/random16.png",
    "../images/random17.png",
    "../images/random18.png",
    "../images/random19.png",
    "../images/random20.png",
    "../images/random21.png",
    "../images/random22.png",
    "../images/random23.png",
    "../images/random24.png",
    "../images/random25.png",
]

# How many times to query each pair (for consistency)
N_TRIALS = 3

# Where to store raw results
OUT_CSV = "results_moondream_reid_25images.csv"

# Prompt – use your strict identity prompt here
PROMPT = (
    "You will see an image containing two faces: LEFT and RIGHT.\n"
    "Your ONLY task is to decide whether LEFT and RIGHT show the SAME REAL PERSON.\n"
    "\n"
    "Rules:\n"
    "- Compare only facial features: bone structure, face shape, jawline, eyebrows, eyes, nose, mouth, ears.\n"
    "- Ignore lighting, pose, angle, expression, background, clothing, and hair style.\n"
    "- If you are NOT completely certain they are the same person, answer \"different person\".\n"
    "- Do NOT explain your reasoning.\n"
    "- Do NOT add any extra words.\n"
    "\n"
    "Answer with EXACTLY ONE of the following lines:\n"
    "same person\n"
    "different person"
)

PANEL_SIZE = (384, 256)
GAP = 8
TIMEOUT_S = 80.0

# ================== IMAGE UTILS ==================

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

# ================== MOLMO CALL + CLASSIFICATION ==================

def call_molmo_pair(pair_bgr: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
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
        raw = (j.get("caption") or j.get("text") or "").strip()
    except Exception:
        raw = resp.text.strip()

    print(f"[debug] HTTP {resp.status_code}, latency={dt_ms:.1f} ms, raw={raw!r}")

    if resp.status_code != 200:
        return None, dt_ms

    return raw, dt_ms


def classify_raw(raw: Optional[str]) -> str:
    """
    Map Molmo's raw text reply to 'same' or 'different'.
    Conservative default: anything ambiguous -> 'different'.
    """
    if not raw:
        return "different"

    lower = raw.strip().lower()

    # exact matches
    if lower == "same person":
        return "same"
    if lower == "different person":
        return "different"

    # tolerant matches
    if "same person" in lower and "different person" not in lower:
        return "same"
    if "different person" in lower:
        return "different"

    # fallback
    return "different"

# ================== DATA GENERATION ==================

def load_image(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        print(f"[error] missing image: {path}")
        return None
    img = cv2.imread(path)
    if img is None:
        print(f"[error] cv2 failed to read: {path}")
    return img


def build_pairs(user_imgs: List[str], other_imgs: List[str]) -> List[Dict]:
    """
    Build pair list:
      - identical self pairs (me1-me1, me2-me2, ...)
      - cross self pairs (me1-me2, me1-me3, ...)
      - self vs others (different pairs)
    """
    pairs = []

    # identical same-person pairs
    for p in user_imgs:
        pairs.append({
            "left": p,
            "right": p,
            "label": "same",
            "scenario": "identical"
        })

    # cross same-person pairs
    for p1, p2 in combinations(user_imgs, 2):
        pairs.append({
            "left": p1,
            "right": p2,
            "label": "same",
            "scenario": "same_cross"
        })

    # different-person pairs: each self vs all others
    for u in user_imgs:
        for o in other_imgs:
            pairs.append({
                "left": u,
                "right": o,
                "label": "diff",
                "scenario": "diff_self_vs_other"
            })

    return pairs

# ================== METRICS ==================

def compute_metrics(records: List[Dict]) -> None:
    total = len(records)
    correct = sum(1 for r in records if r["pred_label"] == r["label"])
    same_total = sum(1 for r in records if r["label"] == "same")
    diff_total = sum(1 for r in records if r["label"] == "diff")
    same_correct = sum(1 for r in records if r["label"] == "same" and r["pred_label"] == "same")
    diff_correct = sum(1 for r in records if r["label"] == "diff" and r["pred_label"] == "diff")

    # confusion
    tp = same_correct  # true same
    tn = diff_correct
    fp = sum(1 for r in records if r["label"] == "diff" and r["pred_label"] == "same")
    fn = sum(1 for r in records if r["label"] == "same" and r["pred_label"] == "diff")

    print("\n========== METRICS ==========")
    print(f"Total pairs (flattened across trials): {total}")
    print(f"Overall accuracy: {correct/total:.3f}")
    if same_total > 0:
        print(f"Same-pair accuracy: {same_correct/same_total:.3f}  ({same_correct}/{same_total})")
    if diff_total > 0:
        print(f"Diff-pair accuracy: {diff_correct/diff_total:.3f}  ({diff_correct}/{diff_total})")

    print("\nConfusion matrix (labels: same/diff):")
    print(f"  TP (true same):      {tp}")
    print(f"  TN (true diff):      {tn}")
    print(f"  FP (same but label diff): {fp}")
    print(f"  FN (diff but label same): {fn}")

    # per-pair consistency
    by_pair: Dict[str, List[str]] = {}
    for r in records:
        key = f"{r['left']}||{r['right']}"
        by_pair.setdefault(key, []).append(r["pred_label"])

    flips = 0
    for key, preds in by_pair.items():
        if len(set(preds)) > 1:
            flips += 1
    total_pairs = len(by_pair)
    if total_pairs > 0:
        print(f"\nPair-level consistency:")
        print(f"  Pairs with at least one flip (across {N_TRIALS} trials): {flips}/{total_pairs} "
              f"({flips/total_pairs:.3f})")

# ================== MAIN ==================

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True) if os.path.dirname(OUT_CSV) else None

    # Build pair list
    pairs = build_pairs(USER_IMAGES, OTHER_IMAGES)
    print(f"[info] total unique pairs: {len(pairs)}")
    records: List[Dict] = []

    for pair in pairs:
        left_path = pair["left"]
        right_path = pair["right"]
        label = pair["label"]
        scenario = pair["scenario"]

        left_img = load_image(left_path)
        right_img = load_image(right_path)
        if left_img is None or right_img is None:
            print(f"[warn] skipping pair {left_path} vs {right_path}")
            continue

        composed = compose_pair(left_img, right_img)

        for trial in range(N_TRIALS):
            raw, dt_ms = call_molmo_pair(composed)
            pred = classify_raw(raw)
            records.append({
                "left": left_path,
                "right": right_path,
                "label": label,
                "scenario": scenario,
                "trial": trial,
                "raw_reply": raw,
                "pred_label": pred,
                "latency_ms": dt_ms if dt_ms is not None else -1.0,
            })

    # Write CSV
    if records:
        fieldnames = list(records[0].keys())
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"[info] wrote {len(records)} rows to {OUT_CSV}")

    # Compute and print metrics
    compute_metrics(records)


if __name__ == "__main__":
    main()
