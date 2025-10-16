import os, sys, time, io, base64, subprocess, threading, queue
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2, requests
from PIL import Image
import numpy as np

# ───────────────────────── Config ─────────────────────────
PROMPT = (
    "You will see a single image with two panels: LEFT = reference, RIGHT = live. "
    "Are they the same person? Reply with EXACTLY one of the following two phrases:\n"
    "same person\n"
    "different person\n"
    "Do not include any other words or explanations. Ignore small changes like pose, lighting, or minor accessories."
)
YOLO_URL  = os.getenv("YOLO_URL",  "http://localhost:9000/detect")
MOLMO_URL = os.getenv("MOLMO_URL", "http://localhost:8000/caption")
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
CONF_THR  = float(os.getenv("CONF_THR", "0.35"))
MOLMO_TIMEOUT_S = float(os.getenv("MOLMO_TIMEOUT_S", "80.0"))

COMPOSE_FILE = os.getenv("COMPOSE_FILE", str(Path(__file__).resolve().parents[1] / "robot-stack" / "docker-compose.yml"))
DOCKER_COMPOSE = os.getenv("DOCKER_COMPOSE", "docker compose")

WINDOW_NAME = "YOLO view"
PAIR_PREVIEW_WINDOW = "Molmo Input Pair (LEFT=ref | RIGHT=live)"

# Spatial-temporal debounce
DEDUPE_IOU_THR = 0.45
DEDUPE_COOLDOWN = 4.0
MIN_BOX_AREA_FRAC = 0.02
MIN_ASPECT = 0.25
MAX_ASPECT = 1.2

# Globals for reference
REF_PATH: Optional[str] = None
REF_CROP_BGR: Optional[np.ndarray] = None

# ─────────────────────── Utilities ────────────────────────
def draw_boxes(frame_bgr, dets, persons_only=True):
    h, w = frame_bgr.shape[:2]
    for d in dets:
        cls = str(d.get("cls", "")).lower()
        cls_id = d.get("cls_id", None)
        conf = float(d.get("conf", 0.0))
        if persons_only and not (cls == "person" or cls_id == 0):
            continue
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls or 'obj'} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 2, y1), (0, 255, 0), -1)
        cv2.putText(frame_bgr, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

def show_frame(frame_bgr, status_text=""):
    if status_text:
        cv2.putText(frame_bgr, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imshow(WINDOW_NAME, frame_bgr)
    cv2.waitKey(1)

def encode_b64(frame_bgr, q=85):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()

def crop(frame_bgr, xyxy) -> Tuple[bool, np.ndarray]:
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    h,w = frame_bgr.shape[:2]
    x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(w,x2), min(h,y2)
    c = frame_bgr[y1:y2, x1:x2]
    return (c.size > 0, c)

def box_area(xyxy):
    x1,y1,x2,y2 = xyxy
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

def box_iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,bx2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw * ih
    ua = box_area(a) + box_area(b) - inter
    return inter / ua if ua > 0 else 0.0

def stop_yolo_and_exit():
    if not COMPOSE_FILE:
        print(f"[orchestrator] Molmo exceeded {MOLMO_TIMEOUT_S:.1f}s → no COMPOSE_FILE set, exiting.", flush=True)
        sys.exit(1)
    print(f"[orchestrator] Molmo exceeded {MOLMO_TIMEOUT_S:.1f}s → stopping yolo-service and exiting.", flush=True)
    try:
        subprocess.run(f'{DOCKER_COMPOSE} -f "{COMPOSE_FILE}" stop yolo', shell=True)
    finally:
        sys.exit(1)

def start_cmd_reader():
    q = queue.Queue()
    def _reader():
        try:
            while True:
                line = input()
                q.put(line.strip())
        except EOFError:
            pass
    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return q

def make_session():
    s = requests.Session()
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(total=2, backoff_factor=0.15,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["POST","GET"])
        ad = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=retry)
        s.mount("http://", ad)
        s.mount("https://", ad)
    except Exception:
        pass
    return s

# ─────────── YOLO helpers (used for reference image too) ───────────
def yolo_detect_persons(bgr_img: np.ndarray, session) -> List[Dict[str, Any]]:
    try:
        resp = session.post(YOLO_URL, json={"image_b64": encode_b64(bgr_img)}, timeout=5.0)
        resp.raise_for_status()
        dets = resp.json().get("detections", [])
        persons = []
        h, w = bgr_img.shape[:2]
        img_area = float(h * w)
        for d in dets:
            cls = str(d.get("cls","")).lower()
            cls_id = d.get("cls_id", None)
            if not (cls == "person" or cls_id == 0):
                continue
            if float(d.get("conf",0.0)) < CONF_THR:
                continue
            x1,y1,x2,y2 = d["xyxy"]
            area = box_area((x1,y1,x2,y2))
            if area < MIN_BOX_AREA_FRAC * img_area:
                continue
            ww, hh = max(1.0, x2-x1), max(1.0, y2-y1)
            aspect = ww / hh
            if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
                continue
            persons.append(d)
        return persons
    except Exception as e:
        print("[yolo error - ref]", e, flush=True)
        return []

def best_person_crop(bgr_img: np.ndarray, session) -> np.ndarray:
    dets = yolo_detect_persons(bgr_img, session)
    if not dets:
        print("[startup] No person found in reference image; using full image.", flush=True)
        return bgr_img
    best = max(dets, key=lambda d: float(d.get("conf",0)))
    okc, c = crop(bgr_img, best["xyxy"])
    return c if okc else bgr_img

# ─────────── Pair compositing for Molmo comparison ───────────
def letterbox(img: np.ndarray, size: Tuple[int,int]) -> np.ndarray:
    th, tw = size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def compose_pair(left_bgr: np.ndarray, right_bgr: np.ndarray,
                 panel_size=(384, 256), gap=8) -> np.ndarray:
    # panel_size: (height, width) of EACH side
    lh, lw = panel_size
    L = letterbox(left_bgr, (lh, lw))
    R = letterbox(right_bgr, (lh, lw))
    H = max(L.shape[0], R.shape[0])
    W = L.shape[1] + gap + R.shape[1]
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:L.shape[0], :L.shape[1]] = L
    out[:R.shape[0], L.shape[1] + gap : L.shape[1] + gap + R.shape[1]] = R
    # annotate sides for clarity
    cv2.putText(out, "LEFT: reference", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(out, "RIGHT: live", (L.shape[1] + gap + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return out

# ────────────── Molmo worker (no normalizer + timer + preview) ───────────────
class MolmoWorker:
    def __init__(self, session, prompt):
        self.s = session
        self.prompt = prompt
        self.q = queue.Queue(maxsize=1)  # backpressure
        self.busy = threading.Event()
        self.alive = True
        self.last_text = ""          # raw text from Molmo (expected: 'same person'|'different person')
        self.last_ms: Optional[float] = None
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def set_prompt(self, p):
        self.prompt = p

    def submit(self, bgr_img_pair: np.ndarray):
        if not self.alive or self.busy.is_set():
            return False
        try:
            self.q.put_nowait(bgr_img_pair)
            self.busy.set()
            return True
        except queue.Full:
            return False

    def _loop(self):
        while self.alive:
            try:
                img = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                # Preview exactly what we send to Molmo
                preview = cv2.resize(img, (640, 320))
                cv2.imshow(PAIR_PREVIEW_WINDOW, preview)
                cv2.waitKey(1)

                start_t = time.perf_counter()
                payload = {"image_b64": encode_b64(img, q=90), "prompt": self.prompt}
                resp = self.s.post(MOLMO_URL, json=payload, timeout=MOLMO_TIMEOUT_S)
                resp.raise_for_status()
                raw = (resp.json().get("caption") or resp.json().get("text") or "").strip()
                self.last_text = raw  # no normalization by request
                self.last_ms = (time.perf_counter() - start_t) * 1000.0
                print(f"[molmo] {raw} | {self.last_ms:.1f} ms", flush=True)
            except requests.Timeout:
                self.last_ms = None
                print("[molmo] timeout", flush=True)
                stop_yolo_and_exit()
            except Exception as e:
                self.last_ms = None
                self.last_text = "err"
                print("[molmo error]", e, flush=True)
            finally:
                self.busy.clear()

    def close(self):
        self.alive = False
        self.busy.clear()

# ───────────────────────── Warm-up ─────────────────────────
def warmup_services(session, cap, ref_crop_bgr, prompt):
    # Warm Molmo by sending ref-vs-ref once
    try:
        pair = compose_pair(ref_crop_bgr, ref_crop_bgr)
        t0 = time.perf_counter()
        r = session.post(
            MOLMO_URL,
            json={"image_b64": encode_b64(pair, q=80), "prompt": prompt},
            timeout=min(10.0, MOLMO_TIMEOUT_S)
        )
        r.raise_for_status()
        print(f"[warmup] molmo responded in {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
    except Exception as e:
        print("[warmup] molmo request error:", e, flush=True)

    # Warm YOLO with one camera frame (if available)
    ok, f0 = cap.read()
    if ok:
        try:
            t1 = time.perf_counter()
            r = session.post(YOLO_URL, json={"image_b64": encode_b64(f0, q=75)}, timeout=5.0)
            r.raise_for_status()
            print(f"[warmup] yolo responded in {(time.perf_counter()-t1)*1000:.1f} ms", flush=True)
        except Exception as e:
            print("[warmup] yolo request error:", e, flush=True)

# ───────────────────────── Startup ─────────────────────────
def startup_reference(session) -> bool:
    global REF_PATH, REF_CROP_BGR
    print("=== Orchestrator Startup: Reference Image ===", flush=True)
    print("Enter path to a reference image of the target person (or press Enter to cancel):", flush=True)
    try:
        user_input = input("Reference image path: ").strip()
    except KeyboardInterrupt:
        print("\n[startup] Interrupted.")
        return False

    if not user_input:
        print("[startup] No reference provided. Exiting because comparison needs a reference.", flush=True)
        return False

    if not os.path.exists(user_input):
        print(f"[startup] File not found: {user_input}", flush=True)
        return False

    img = cv2.imread(user_input)
    if img is None:
        print(f"[startup] Could not read image: {user_input}", flush=True)
        return False

    REF_PATH = user_input
    REF_CROP_BGR = best_person_crop(img, session)
    print(f"[startup] Loaded reference from: {REF_PATH} | crop shape={REF_CROP_BGR.shape}", flush=True)
    return True

# ───────────────────────── Main ─────────────────────────
def main():
    print("=== Orchestrator CLI (Molmo Person Comparison) ===", flush=True)

    session = make_session()
    if not startup_reference(session):
        return

    cap = cv2.VideoCapture(CAM_INDEX)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if not cap.isOpened():
        print(f"[error] cannot open camera {CAM_INDEX}")
        sys.exit(2)

    # Optional camera hints
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    molmo = MolmoWorker(session, PROMPT)
    detect_on = True
    last_overlay = ""
    cmd_q = start_cmd_reader()

    # Warm up both services so the first real comparison is quick
    warmup_services(session, cap, REF_CROP_BGR, PROMPT)

    recent: List[Tuple[Tuple[float,float,float,float], float]] = []

    try:
        while True:
            # Commands
            while not cmd_q.empty():
                cmdline = cmd_q.get_nowait()
                if not cmdline: continue
                low = cmdline.lower()
                if low in ("quit", "exit", "q"):
                    print("[cmd] quitting…")
                    return
                elif low == "on":
                    detect_on = True; print("[cmd] detection ON")
                elif low == "off":
                    detect_on = False; print("[cmd] detection OFF")
                elif low == "status":
                    ms = f"{molmo.last_ms:.1f} ms" if molmo.last_ms is not None else "n/a"
                    print(f"[status] detection={'ON' if detect_on else 'OFF'} | molmo_busy={molmo.busy.is_set()} | last='{molmo.last_text}' | {ms}")
                elif low.startswith("prompt "):
                    p = cmdline[7:].strip()
                    molmo.set_prompt(p)
                    print(f"[cmd] updated prompt → {p}")
                else:
                    print(f"[cmd] unknown: {cmdline}")

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01); continue

            if not detect_on:
                show_frame(frame.copy(), status_text="DETECTION OFF")
                time.sleep(0.01); continue

            # YOLO call
            try:
                resp = session.post(YOLO_URL, json={"image_b64": encode_b64(frame)}, timeout=2.5)
                resp.raise_for_status()
                dets = resp.json().get("detections", [])
            except Exception as e:
                print("[yolo error]", e)
                show_frame(frame.copy(), status_text="YOLO ERROR")
                time.sleep(0.05); continue

            # Visualize
            vis = frame.copy()
            draw_boxes(vis, dets, persons_only=True)

            # Filter persons
            h, w = frame.shape[:2]
            img_area = float(h * w)
            persons = []
            for d in dets:
                cls = str(d.get("cls","")).lower(); cls_id = d.get("cls_id", None)
                if not (cls == "person" or cls_id == 0): continue
                if float(d.get("conf",0.0)) < CONF_THR: continue
                x1,y1,x2,y2 = d["xyxy"]
                area = box_area((x1,y1,x2,y2))
                if area < MIN_BOX_AREA_FRAC * img_area: continue
                ww, hh = max(1.0, x2-x1), max(1.0, y2-y1)
                aspect = ww / hh
                if not (MIN_ASPECT <= aspect <= MAX_ASPECT): continue
                persons.append(d)

            # Submit best person to Molmo
            if persons and REF_CROP_BGR is not None:
                best = max(persons, key=lambda d: float(d.get("conf",0)))
                box = tuple(map(float, best["xyxy"]))
                # temporal dedupe
                now = time.time()
                recent = [(b, t) for (b, t) in recent if now - t <= DEDUPE_COOLDOWN]
                if not any(box_iou(box, b) >= DEDUPE_IOU_THR for (b, t) in recent):
                    okc, live_crop = crop(frame, best["xyxy"])
                    if okc:
                        pair = compose_pair(REF_CROP_BGR, live_crop)
                        if molmo.submit(pair):
                            recent.append((box, now))

            # Overlay latest result + timing
            if molmo.last_text:
                if molmo.last_ms is not None:
                    last_overlay = f"{molmo.last_text} | {molmo.last_ms:.0f} ms"
                else:
                    last_overlay = f"{molmo.last_text}"
            else:
                last_overlay = ""

            show_frame(vis, status_text=last_overlay if last_overlay else "DETECTION ON")
            time.sleep(0.01)

    finally:
        try:
            molmo.close()
            cap.release()
            cv2.destroyWindow(PAIR_PREVIEW_WINDOW)
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[camera] closed")

if __name__ == "__main__":
    main()
