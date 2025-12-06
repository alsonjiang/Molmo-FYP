#Activate whichever VLM you want to test at :8000
#Activate the YOLO service at :9000
#this script takes frames from the webcam and sends to the VLM
#only the first 20 frames will be captioned
#output is saved to CSV, together with latency of both YOLO and the VLM


import os
import time
import io
import csv
import argparse
from typing import List, Dict, Any, Tuple, Optional

import cv2
import requests
import numpy as np
from PIL import Image

# ───────────────────────── Config ─────────────────────────
YOLO_URL  = os.getenv("YOLO_URL",  "http://localhost:9000/detect")
VLM_URL   = os.getenv("MOLMO_URL", "http://localhost:8000/caption")
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))

PROMPT = (
    "Describe the person's appearance. Focus only on clothing colour, hairstyle "
    "and any visible distinguishing features. Keep the answer concise."
)

DEFAULT_N_FRAMES = 20
WINDOW_NAME = "YOLO → VLM Live Caption Test"


# ─────────────────────── Utilities ────────────────────────
def encode_b64(frame_bgr, q: int = 80) -> str:
    """JPEG-encode BGR frame and return base64 string."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def make_session() -> requests.Session:
    s = requests.Session()
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry = Retry(
            total=2,
            backoff_factor=0.15,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        ad = HTTPAdapter(
            pool_connections=8,
            pool_maxsize=8,
            max_retries=retry,
        )
        s.mount("http://", ad)
        s.mount("https://", ad)
    except Exception:
        pass
    return s


def draw_boxes(frame_bgr, dets, persons_only=True):
    h, w = frame_bgr.shape[:2]
    for d in dets:
        cls = str(d.get("cls", "")).lower()
        cls_id = d.get("cls_id", None)
        conf = float(d.get("conf", 0.0))
        if persons_only and not (cls == "person" or cls_id == 0):
            continue
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls or 'obj'} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame_bgr,
            (x1, y1 - th - 6),
            (x1 + tw + 2, y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame_bgr,
            label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )


def crop(frame_bgr, xyxy) -> Tuple[bool, np.ndarray]:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    c = frame_bgr[y1:y2, x1:x2]
    return (c.size > 0, c)


# ─────────── YOLO helper ───────────
def yolo_detect_persons(bgr_img: np.ndarray, session: requests.Session) -> List[Dict[str, Any]]:
    """Call YOLO service, keep only 'person' detections."""
    try:
        t0 = time.perf_counter()
        resp = session.post(
            YOLO_URL,
            json={"image_b64": encode_b64(bgr_img, q=80)},
            timeout=5.0,
        )
        yolo_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        dets = resp.json().get("detections", [])
        persons = []
        for d in dets:
            cls = str(d.get("cls", "")).lower()
            cls_id = d.get("cls_id", None)
            if cls == "person" or cls_id == 0:
                persons.append(d)
        return persons, yolo_ms
    except Exception as e:
        print("[yolo error]", e, flush=True)
        return [], -1.0


# ─────────── VLM helper ───────────
def call_vlm_caption(session: requests.Session, crop_bgr: np.ndarray) -> Tuple[str, float]:
    """Send crop to VLM caption endpoint, return (caption, latency_ms)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "image_b64": img_b64,
        "prompt": PROMPT,
    }

    t0 = time.perf_counter()
    try:
        r = session.post(VLM_URL, json=payload, timeout=30.0)
        vlm_ms = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        j = r.json()
        caption = (j.get("caption") or j.get("text") or "").strip()
        return caption, vlm_ms
    except Exception as e:
        print("[vlm error]", e, flush=True)
        return "[ERROR]", -1.0


# ─────────── Main test loop ───────────
def run_live_caption_test(n_frames: int, csv_path: Optional[str] = None):
    session = make_session()
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[error] cannot open camera {CAM_INDEX}")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    results = []

    print(f"[info] Starting live caption test for {n_frames} frames.")
    print(f"[info] YOLO_URL = {YOLO_URL}")
    print(f"[info] VLM_URL  = {VLM_URL}")

    frame_idx = 0
    try:
        while frame_idx < n_frames:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            persons, yolo_ms = yolo_detect_persons(frame, session)
            vis = frame.copy()
            draw_boxes(vis, persons, persons_only=True)

            caption = ""
            vlm_ms = -1.0

            if persons:
                # Take highest-confidence person
                best = max(persons, key=lambda d: float(d.get("conf", 0.0)))
                okc, crop_img = crop(frame, best["xyxy"])
                if okc:
                    caption, vlm_ms = call_vlm_caption(session, crop_img)
                    frame_idx += 1
                    print(
                        f"[frame {frame_idx:03d}] YOLO={yolo_ms:.1f} ms | "
                        f"VLM={vlm_ms:.1f} ms | caption='{caption}'",
                        flush=True,
                    )
                    results.append({
                        "frame": frame_idx,
                        "yolo_ms": yolo_ms,
                        "vlm_ms": vlm_ms,
                        "caption": caption,
                    })
                    status = f"YOLO {yolo_ms:.0f} ms | VLM {vlm_ms:.0f} ms"
                else:
                    status = "NO CROP"
            else:
                status = "NO PERSON"

            cv2.putText(
                vis,
                status,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Summary
        if results:
            yolo_valid = [r["yolo_ms"] for r in results if r["yolo_ms"] >= 0]
            vlm_valid  = [r["vlm_ms"] for r in results if r["vlm_ms"] >= 0]
            print("\n===== SUMMARY =====")
            print(f"Frames processed: {len(results)}")
            if yolo_valid:
                print(
                    f"YOLO latency: mean={np.mean(yolo_valid):.1f} ms, "
                    f"min={np.min(yolo_valid):.1f}, max={np.max(yolo_valid):.1f}"
                )
            if vlm_valid:
                print(
                    f"VLM latency:  mean={np.mean(vlm_valid):.1f} ms, "
                    f"min={np.min(vlm_valid):.1f}, max={np.max(vlm_valid):.1f}"
                )
            print("Sample captions:")
            for r in results[:5]:
                print(f" - {r['caption']}")

            if csv_path:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["frame", "yolo_ms", "vlm_ms", "caption"])
                    writer.writeheader()
                    writer.writerows(results)
                print(f"[info] Results written to {csv_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ─────────── Entry point ───────────
if __name__ == "__main__":
    import base64
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=DEFAULT_N_FRAMES,
                        help="Number of frames to evaluate (default: 20)")
    parser.add_argument("--csv", type=str, default="live_caption_results.csv",
                        help="Output CSV filename (default: live_caption_results.csv)")
    args = parser.parse_args()

    run_live_caption_test(args.n_frames, csv_path=args.csv)
