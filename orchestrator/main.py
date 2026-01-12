#This is the main script
#Activate either VLMs at :8000
#Activate YOLO at :9000

#Two MODES: SEMANTIC DESCRIPTION (CAPTION) or IDENTITY MATCHING (IDENTITY)
#python main.py identity -> provide reference image
#python main.py caption -> auto starts

import os, sys, time, io, base64, threading, queue
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2, requests
from PIL import Image
import numpy as np
from bs4 import BeautifulSoup

# ───────────────────────── Config ─────────────────────────

IDENTITY_PROMPT = (
    "You are a strict face verification system.\n"
    "You will see a single image with two panels: LEFT = reference, RIGHT = live.\n"
    "If BOTH panels clearly show the same real person's face, answer exactly:\n"
    "same person\n"
    "If they are different people, OR if either panel does NOT clearly show a human face "
    "(e.g. hand, object, blur, back of head, occlusion), OR if you are uncertain, "
    "answer exactly:\n"
    "different person\n"
    "Do not include any other words or explanations."
)

CAPTION_PROMPT = os.getenv(
    "CAPTION_PROMPT",
    "Describe the person in the image in one or two concise sentences."
)

POINTING_PROMPT = (    
    "Point to a person in the image"
)

YOLO_URL  = os.getenv("YOLO_URL",  "http://localhost:9000/detect")
VLM_URL   = os.getenv("MOLMO_URL", "http://localhost:8000/caption")
#MOONDREAM_URL   = os.getenv("MOONDREAM_URL", "http://localhost:8003/caption")
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
VLM_TIMEOUT_S = float(os.getenv("MOLMO_TIMEOUT_S", "80.0"))

WINDOW_NAME = "Orchestrator View"
PAIR_PREVIEW_WINDOW = "VLM Input Pair (LEFT=ref | RIGHT=live)"

# temporal dedupe on boxes (identity mode)
DEDUPE_IOU_THR = 0.45
DEDUPE_COOLDOWN = 2.0

# ───────────────────── Mode selection ─────────────────────

def resolve_mode() -> str:
    # CLI arg > env > default
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().lower()
        if arg in ("identity", "id", "match"):
            return "identity"
        if arg in ("caption", "cap"):
            return "caption"
        if arg in ("pointing", "point"):
            return "pointing"
        if arg in ("oneshot", "one", "single"):
            return "oneshot"


    env_mode = os.getenv("MODE", "").strip().lower()
    if env_mode in ("identity", "id", "match"):
        return "identity"
    if env_mode in ("caption", "cap"):
        return "caption"
    if env_mode in ("pointing", "point"):
        return "pointing"

    return "identity"


# ───────── Face detection (gate) ─────────
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_DETECTOR = cv2.CascadeClassifier(FACE_CASCADE_PATH)
MIN_FACE_FRAC = 0.05  # minimum face area fraction of crop to accept


def has_face(bgr: np.ndarray) -> bool:
    if bgr is None or bgr.size == 0:
        return False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return False
    h, w = gray.shape
    img_area = float(h * w)
    for (x, y, fw, fh) in faces:
        if (fw * fh) / img_area >= MIN_FACE_FRAC:
            return True
    return False


# ───────────────────── Utilities ─────────────────────

def encode_b64(frame_bgr, q=80):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


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
        label = f"person {conf:.2f}"
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


def show_frame(frame_bgr, status_text=""):
    if status_text:
        cv2.putText(
            frame_bgr,
            status_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )
    cv2.imshow(WINDOW_NAME, frame_bgr)
    cv2.waitKey(1)


def crop(frame_bgr, xyxy) -> Tuple[bool, np.ndarray]:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    c = frame_bgr[y1:y2, x1:x2]
    return (c.size > 0, c)


def box_area(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = box_area(a) + box_area(b) - inter
    return inter / ua if ua > 0 else 0.0


def make_session():
    s = requests.Session()
    return s


def yolo_detect_persons(bgr_img: np.ndarray, session) -> List[Dict[str, Any]]:
    try:
        resp = session.post(
            YOLO_URL,
            json={"image_b64": encode_b64(bgr_img, q=80)},
            timeout=5.0,
        )
        resp.raise_for_status()
        dets = resp.json().get("detections", [])
        persons = []
        for d in dets:
            cls = str(d.get("cls", "")).lower()
            cls_id = d.get("cls_id", None)
            if not (cls == "person" or cls_id == 0):
                continue
            persons.append(d)
        return persons
    except Exception as e:
        print("[yolo error]", e, flush=True)
        return []


def letterbox(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    th, tw = size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def compose_pair(left_bgr, right_bgr, panel_size=(384, 256), gap=8) -> np.ndarray:
    lh, lw = panel_size
    L = letterbox(left_bgr, (lh, lw))
    R = letterbox(right_bgr, (lh, lw))
    H = max(L.shape[0], R.shape[0])
    W = L.shape[1] + gap + R.shape[1]
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[: L.shape[0], : L.shape[1]] = L
    out[: R.shape[0], L.shape[1] + gap : L.shape[1] + gap + R.shape[1]] = R
    cv2.putText(out, "LEFT: reference", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(out, "RIGHT: live",
                (L.shape[1] + gap + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def classify_match(text: str) -> str:
    t = text.strip().lower()
    if t == "same person":
        return "SAME"
    if t == "different person":
        return "DIFFERENT"
    return "DIFFERENT"

def extract_points_from_molmo(text: str) -> List[List[float]]:
    try:
        html = text
        soup = BeautifulSoup(html, 'html.parser')
        tag = soup.find('point')

        if tag:
            coords = []
            coords.append([float(tag[f'x']),float(tag[f'y'])])
            label = tag['alt']
        else:
            tag = soup.find('points')
            # Extract values
            coords = []
            for i in range(1, 20):
                coords.append([float(tag[f'x{i}']),float(tag[f'y{i}'])])
            label = tag['alt']
            print(f'Extracted {len(coords)} points with label: {label}')
        return coords
    except:
        print('No points found')

def are_points_in_bbox(points: List[List[float]], box: List[float]) -> bool:
    x1, y1, x2, y2 = box
    for (px, py) in points:
        if not (x1 <= (px/100) <= x2 and y1 <= (py/100) <= y2):
            return False
    return True

def draw_points_on_frame(frame_bgr: np.ndarray, points: List[List[float]]):
    h, w = frame_bgr.shape[:2]
    for (px, py) in points:
        cx = int((px / 100) * w)
        cy = int((py / 100) * h)
        cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1)

# ────────────── VLM worker ───────────────

class VLMWorker:
    """
    Generic VLM worker with a 1-slot queue.
    In identity mode, caller interprets last_text via classify_match.
    In caption mode, caller treats last_text as caption.
    """
    def __init__(self, session, prompt: str):
        self.s = session
        self.prompt = prompt
        self.q = queue.Queue(maxsize=1)
        self.busy = threading.Event()
        self.alive = True

        self.last_text: str = ""
        self.last_ms: Optional[float] = None
        self.last_state: Optional[str] = None  # PENDING / OK / ERROR / TIMEOUT

        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def set_prompt(self, p: str):
        self.prompt = p

    def submit(self, img: np.ndarray) -> bool:
        if not self.alive or self.busy.is_set():
            return False
        try:
            self.last_text = ""
            self.last_ms = None
            self.last_state = "PENDING"
            self.q.put_nowait(img)
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
                start_t = time.perf_counter()
                payload = {
                    "image_b64": encode_b64(img, q=80),
                    "prompt": self.prompt,
                }
                resp = self.s.post(
                    VLM_URL,
                    json=payload,
                    timeout=VLM_TIMEOUT_S,
                )
                resp.raise_for_status()
                j = resp.json()
                raw = (j.get("caption") or j.get("text") or "").strip()
                self.last_text = raw
                self.last_ms = (time.perf_counter() - start_t) * 1000.0
                self.last_state = "OK"
                print(f"[vlm] {raw} | {self.last_ms:.1f} ms", flush=True)
            except requests.Timeout:
                self.last_ms = None
                self.last_text = ""
                self.last_state = "TIMEOUT"
                print("[vlm] timeout", flush=True)
            except Exception as e:
                self.last_ms = None
                self.last_text = "err"
                self.last_state = "ERROR"
                print("[vlm error]", e, flush=True)
            finally:
                self.busy.clear()

    def close(self):
        self.alive = False
        self.busy.clear()


# ───────────── Identity mode helpers ─────────────

def startup_reference(session) -> Optional[np.ndarray]:
    print("=== Identity mode: reference image selection ===", flush=True)
    path = input("Reference image path (or Enter to cancel): ").strip()
    if not path:
        print("[startup] no reference given, aborting.")
        return None
    if not os.path.exists(path):
        print(f"[startup] file not found: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"[startup] could not read image: {path}")
        return None

    persons = yolo_detect_persons(img, session)
    if not persons:
        print("[startup] no person found, using full image.")
        ref = img
    else:
        best = max(persons, key=lambda d: float(d.get("conf", 0)))
        okc, ref = crop(img, best["xyxy"])
        if not okc:
            ref = img

    if not has_face(ref):
        print("[startup] ERROR: no clear face in reference. Use frontal face.", flush=True)
        return None

    print(f"[startup] reference loaded, shape={ref.shape}", flush=True)
    return ref


# ──────────────── Main loops ────────────────

def run_oneshot_where_is_person(session):
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[error] cannot open camera {CAM_INDEX}")
        return

    ok, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()

    if not ok or frame is None:
        print("[error] failed to read frame")
        return

    prompt = "Give bounding box coordinates of the person in the frame."
    payload = {"image_b64": encode_b64(frame, q=80), "prompt": prompt}

    t0 = time.perf_counter()
    try:
        resp = session.post(VLM_URL, json=payload, timeout=VLM_TIMEOUT_S)
        resp.raise_for_status()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        j = resp.json()
        raw = (j.get("caption") or j.get("text") or resp.text).strip()
        print(f"[oneshot] latency: {dt_ms:.1f} ms")
        print(f"[oneshot] reply: {raw}")
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[oneshot] failed after {dt_ms:.1f} ms: {e}")

def run_identity_mode(session):
    ref_crop = startup_reference(session)
    if ref_crop is None:
        return

    cap = cv2.VideoCapture(CAM_INDEX)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if not cap.isOpened():
        print(f"[error] cannot open camera {CAM_INDEX}")
        sys.exit(2)

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    vlm = VLMWorker(session, IDENTITY_PROMPT)
    recent: List[Tuple[Tuple[float, float, float, float], float]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            persons = yolo_detect_persons(frame, session)
            vis = frame.copy()
            draw_boxes(vis, persons, persons_only=True)

            face_status = ""

            if not persons:
                face_status = "NO PERSON"
            else:
                best = max(persons, key=lambda d: float(d.get("conf", 0)))
                box = tuple(map(float, best["xyxy"]))
                okc, live_crop = crop(frame, best["xyxy"])
                if not okc:
                    face_status = "NO CROP"
                elif not has_face(live_crop):
                    face_status = "NO FACE"
                else:
                    pair = compose_pair(ref_crop, live_crop)
                    now = time.time()
                    recent = [(b, t) for (b, t) in recent if now - t <= DEDUPE_COOLDOWN]
                    if not any(box_iou(box, b) >= DEDUPE_IOU_THR for (b, t) in recent):
                        if vlm.submit(pair):
                            print("[orchestrator] submitted pair to VLM", flush=True)
                            recent.append((box, now))

            # interpret last_text as SAME/DIFFERENT
            if vlm.last_state == "OK":
                cls = classify_match(vlm.last_text)
                if vlm.last_ms is not None:
                    status = f"{cls} | {vlm.last_ms:.0f} ms"
                else:
                    status = cls
            elif vlm.last_state in ("PENDING", None):
                status = face_status or "PENDING"
            else:
                status = vlm.last_state

            if face_status:
                status = face_status

            show_frame(vis, status_text=status)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.01)
    finally:
        vlm.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[identity] camera closed")


def run_caption_mode(session):
    cap = cv2.VideoCapture(CAM_INDEX)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if not cap.isOpened():
        print(f"[error] cannot open camera {CAM_INDEX}")
        sys.exit(2)

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    vlm = VLMWorker(session, CAPTION_PROMPT)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            persons = yolo_detect_persons(frame, session)
            vis = frame.copy()
            draw_boxes(vis, persons, persons_only=True)

            status = "CAPTION ON"

            if not persons:
                status = "NO PERSON"
            else:
                best = max(persons, key=lambda d: float(d.get("conf", 0)))
                okc, person_crop = crop(frame, best["xyxy"])
                if not okc or not has_face(person_crop):
                    status = "NO FACE"
                else:
                    if not vlm.busy.is_set():
                        if vlm.submit(person_crop):
                            print("[orchestrator] submitted person crop for caption", flush=True)
                    if vlm.last_state == "OK":
                        status = "CAPTION OK"
                    elif vlm.last_state in ("ERROR", "TIMEOUT"):
                        status = vlm.last_state
                    elif vlm.last_state == "PENDING":
                        status = "CAPTIONING…"

            show_frame(vis, status_text=status)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.01)
    finally:
        vlm.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[caption] camera closed")

def run_pointing_mode(session):
    cap = cv2.VideoCapture(CAM_INDEX)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if not cap.isOpened():
        print(f"[error] cannot open camera {CAM_INDEX}")
        sys.exit(2)

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    vlm = VLMWorker(session, POINTING_PROMPT)
    
    old_points = [[0,0]]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            persons = yolo_detect_persons(frame, session)
            vis = frame.copy()
            draw_boxes(vis, persons, persons_only=True)
            draw_points_on_frame(vis, old_points)
            points = extract_points_from_molmo(vlm.last_text)
            
            if points:
                old_points = points
            status = "POINTING ON"
            if not persons:
                status = "NO PERSON"
            else:
                if not vlm.busy.is_set():
                    if vlm.submit(vis):
                        print("[orchestrator] submitted live feed for pointing", flush=True)
                if vlm.last_state == "OK":
                    best = max(persons, key=lambda d: float(d.get("conf", 0)))
                    if are_points_in_bbox(old_points, best["xyxy"]):
                        status = "POINTS OK"
                    else:
                        status = "POINTS OUTSIDE"
                elif vlm.last_state in ("ERROR", "TIMEOUT"):
                    status = vlm.last_state
                elif vlm.last_state == "PENDING":
                    status = "GENERATING POINTS..."

            show_frame(vis, status_text=status)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.01)
    finally:
        vlm.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[pointing] camera closed")

# ───────────────────────── Main ─────────────────────────

def main():
    mode = resolve_mode()
    print(f"=== Orchestrator starting in MODE = {mode} ===", flush=True)
    session = make_session()

    if mode == "caption":
        run_caption_mode(session)
    elif mode == "pointing":
        run_pointing_mode(session)
    
    elif mode == "oneshot":
        run_oneshot_where_is_person(session)

    else:
        run_identity_mode(session)


if __name__ == "__main__":
    main()
