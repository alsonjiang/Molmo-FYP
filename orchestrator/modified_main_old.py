import os, sys, time, io, base64, subprocess, threading, queue, math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2, requests
from PIL import Image

# ───────────────────────── Config ─────────────────────────
PROMPT = 'Describe the person only like so: Gender | Shirt Color'
YOLO_URL  = os.getenv("YOLO_URL",  "http://localhost:9000/detect")
MOLMO_URL = os.getenv("MOLMO_URL", "http://localhost:8000/caption")
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
CONF_THR  = float(os.getenv("CONF_THR", "0.35"))
MOLMO_TIMEOUT_S = float(os.getenv("MOLMO_TIMEOUT_S", "80.0"))

COMPOSE_FILE = os.getenv("COMPOSE_FILE", str(Path(__file__).resolve().parents[1] / "robot-stack" / "docker-compose.yml"))
DOCKER_COMPOSE = os.getenv("DOCKER_COMPOSE", "docker compose")

WINDOW_NAME = "YOLO view"

# Spatial-temporal debounce: don't send near-duplicate boxes to Molmo
DEDUPE_IOU_THR = 0.45     # overlap to consider "same"
DEDUPE_COOLDOWN = 4.0     # seconds before the same person crop can be sent again
MIN_BOX_AREA_FRAC = 0.02  # tiny boxes (e.g., background silhouettes) ignored
MIN_ASPECT = 0.25         # w/h sanity for a person
MAX_ASPECT = 1.2


# ─────────────── NEW: Label filtering functionality ────────────────

TARGET_LABEL = None  # Will store the target object label to detect

# ─────────────────────── Utilities ────────────────────────
def draw_boxes(frame_bgr, dets, persons_only=True):
    h, w = frame_bgr.shape[:2]
    for d in dets:
        cls = str(d.get("cls", ""))
        cls_id = d.get("cls_id", None)
        conf = float(d.get("conf", 0.0))
        if persons_only and not (cls.lower() == "person" or cls_id == 0):
            continue
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls or 'obj'} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
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

def crop(frame_bgr, xyxy) -> Tuple[bool, any]:
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
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
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

def restart_yolo():
    if not COMPOSE_FILE:
        print("[orchestrator] restart requested, but no COMPOSE_FILE set → skipping.", flush=True)
        return
    print("[orchestrator] Restarting yolo-service…", flush=True)
    subprocess.run(f'{DOCKER_COMPOSE} -f "{COMPOSE_FILE}" restart yolo', shell=True)

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

# ─────────────── NEW: Image processing and label extraction ────────────────

def extract_image_path_from_text(text):
    """Extract image path from user input text using | as delimiter"""
    try:
        # Split by | delimiter and look for image path
        parts = [part.strip() for part in text.split('|')]

        # Look for file extensions or valid paths
        for part in parts:
            # Check if it looks like a file path
            if any(ext in part.lower() for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']):
                return part
            # Check if it's a valid file path
            if os.path.exists(part):
                return part

        # If no delimiter found, assume the whole text is the path
        if os.path.exists(text.strip()):
            return text.strip()

        return None
    except Exception as e:
        print(f"[error] Failed to extract image path: {e}")
        return None

def load_image_and_extract_label(image_path, session):
    """Load image from path and extract label using Molmo"""
    global TARGET_LABEL

    try:
        # Load and encode image
        if not os.path.exists(image_path):
            print(f"[error] Image file not found: {image_path}")
            return False

        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"[error] Failed to load image: {image_path}")
            return False

        print(f"[startup] Loaded image: {image_path}")

        # Encode image to base64
        img_b64 = encode_b64(img, q=90)

        # Call Molmo to extract label
        label_prompt = "Respond with only the object name if there is a person (eg. 'person')"
        #label_prompt = "What is the main object in this image? Respond with only the object name (e.g., 'person', 'car', 'dog', etc.)"

        payload = {
            "image_b64": img_b64, 
            "prompt": label_prompt
        }

        print("[startup] Sending image to Molmo for label extraction...")
        resp = session.post(MOLMO_URL, json=payload, timeout=MOLMO_TIMEOUT_S)
        resp.raise_for_status()

        extracted_text = (resp.json().get("caption") or resp.json().get("text") or "").strip()

        if extracted_text:
            # Parse the response using | delimiter if present
            if '|' in extracted_text:
                label_parts = [part.strip() for part in extracted_text.split('|')]
                TARGET_LABEL = label_parts[0].lower()  # Use first part as primary label
            else:
                TARGET_LABEL = extracted_text.split(": ")[2].lower()

            print(f"[startup] Extracted target label: '{TARGET_LABEL}'")
            return True
        else:
            print("[error] Molmo returned empty response")
            return False

    except requests.Timeout:
        print("[error] Molmo timeout during label extraction")
        return False
    except Exception as e:
        print(f"[error] Failed to extract label: {e}")
        return False

def should_detect_object(detection):
    """Check if detection matches target label"""
    global TARGET_LABEL

    if TARGET_LABEL is None:
        # If no target label set, default to person detection
        cls = str(detection.get("cls", "")).lower()
        cls_id = detection.get("cls_id", None)
        return cls == "person" or cls_id == 0

    # Check if detection matches target label
    cls = str(detection.get("cls", "")).lower()
    return TARGET_LABEL in cls or cls in TARGET_LABEL

def startup_label_extraction():
    """Handle startup sequence for label extraction"""
    global TARGET_LABEL

    print("=== Orchestrator Startup ===", flush=True)
    print("Please provide an image to extract object label for detection filtering.", flush=True)
    print("Format: 'image_path' or 'description|image_path|other_info'", flush=True)
    print("Press Enter to skip and use default person detection.", flush=True)

    try:
        user_input = input("Enter image path: ").strip()

        if not user_input:
            print("[startup] No input provided, using default person detection")
            TARGET_LABEL = "person"
            return True

        # Extract image path from input
        image_path = extract_image_path_from_text(user_input)

        if image_path is None:
            print(f"[startup] Could not find valid image path in: '{user_input}'")
            print("[startup] Using default person detection")
            TARGET_LABEL = "person"
            return True

        # Create session for Molmo communication
        session = make_session()

        # Extract label using Molmo
        success = load_image_and_extract_label(image_path, session)

        if not success:
            print("[startup] Label extraction failed, using default person detection")
            TARGET_LABEL = "person"

        return True

    except KeyboardInterrupt:
        print("\n[startup] Interrupted by user")
        return False
    except Exception as e:
        print(f"[startup] Error during label extraction: {e}")
        print("[startup] Using default person detection")
        TARGET_LABEL = "person"
        return True

# ────────────────── HTTP session with retries ──────────────────
def make_session():
    s = requests.Session()
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(total=3, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST","GET"])
        s.mount("http://", HTTPAdapter(max_retries=retry))
        s.mount("https://", HTTPAdapter(max_retries=retry))
    except Exception:
        pass
    return s

# ────────────── Molmo worker (single in-flight) ───────────────
class MolmoWorker:
    def __init__(self, session, prompt):
        self.s = session
        self.prompt = prompt
        self.q = queue.Queue(maxsize=1)  # backpressure: only one pending
        self.busy = threading.Event()
        self.alive = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def set_prompt(self, p):
        self.prompt = p

    def submit(self, crop_bgr):
        if not self.alive or self.busy.is_set():
            return False
        try:
            self.q.put_nowait(crop_bgr)
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
                payload = {"image_b64": encode_b64(img, q=90), "prompt": self.prompt}
                resp = self.s.post(MOLMO_URL, json=payload, timeout=MOLMO_TIMEOUT_S)
                resp.raise_for_status()
                text = (resp.json().get("caption") or resp.json().get("text") or "").strip()
                print(f"[molmo] {text}", flush=True)
                #restart_yolo()
            except requests.Timeout:
                stop_yolo_and_exit()
            except Exception as e:
                print("[molmo error]", e, flush=True)
            finally:
                self.busy.clear()

    def close(self):
        self.alive = False
        self.busy.clear()

# ───────────────────────── Main ─────────────────────────

def main():
    global TARGET_LABEL

    # NEW: Startup label extraction sequence
    if not startup_label_extraction():
        print("[startup] Startup cancelled by user")
        return

    print("=== Orchestrator CLI ===", flush=True)
    print(f"Target detection label: '{TARGET_LABEL}'", flush=True)
    print("type 'off' to pause detection, 'on' to resume, 'prompt ' to change, 'q' to quit.", flush=True)

    cap = cv2.VideoCapture(CAM_INDEX)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if not cap.isOpened():

        print(f"[error] cannot open camera {CAM_INDEX}")

        sys.exit(2)

    # Optional camera hints (best-effort)

    cap.set(cv2.CAP_PROP_FPS, 30)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    session = make_session()

    prompt = PROMPT

    molmo = MolmoWorker(session, prompt)

    detect_on = True

    last_status = 0.0

    cmd_q = start_cmd_reader()

    # track recent sent boxes

    recent: List[Tuple[Tuple[float,float,float,float], float]] = [] # (xyxy, ts)

    try:
        while True:
            # Commands
            while not cmd_q.empty():
                cmdline = cmd_q.get_nowait()
                if not cmdline:
                    continue
                low = cmdline.lower()
                if low in ("quit", "exit", "q"):
                    print("[cmd] quitting…")
                    return
                elif low == "on":
                    detect_on = True; print("[cmd] detection ON")
                elif low == "off":
                    detect_on = False; print("[cmd] detection OFF")
                elif low == "status":
                    print(f"[status] detection={'ON' if detect_on else 'OFF'} | prompt='{prompt}' | molmo_busy={molmo.busy.is_set()} | target_label='{TARGET_LABEL}'")
                elif low.startswith("prompt "):
                    prompt = cmdline[7:].strip()
                    molmo.set_prompt(prompt)
                    print(f"[cmd] updated prompt → {prompt}")
                else:
                    print(f"[cmd] unknown: {cmdline}")
            ok, frame = cap.read()

            if not ok:
                time.sleep(0.01)
                continue

            if not detect_on:
                if time.time() - last_status > 2.0:
                    print("[status] detection OFF")
                    last_status = time.time()
                show_frame(frame.copy(), status_text="DETECTION OFF")
                time.sleep(0.01)
                continue

            # YOLO call

            try:
                resp = session.post(YOLO_URL, json={"image_b64": encode_b64(frame)}, timeout=2.5)
                resp.raise_for_status()
                dets = resp.json().get("detections", [])
            except Exception as e:
                print("[yolo error]", e)
                show_frame(frame.copy(), status_text="YOLO ERROR")
                time.sleep(0.05)
                continue

            vis = frame.copy()
            draw_boxes(vis, dets, persons_only=False)
            show_frame(vis, status_text=f"DETECTION ON (Target: {TARGET_LABEL})")

            # NEW: Filter based on target label instead of just persons
            
            h, w = frame.shape[:2]
            img_area = float(h * w)
            target_objects = []

            for d in dets:
                if should_detect_object(d) and float(d.get("conf",0)) >= CONF_THR:
                    x1,y1,x2,y2 = d["xyxy"]
                    area = box_area((x1,y1,x2,y2))
                    if area < MIN_BOX_AREA_FRAC * img_area:
                        continue                       
                    ww, hh = max(1.0, x2-x1), max(1.0, y2-y1)
                    aspect = ww / hh
                    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
                        continue
                        
                    target_objects.append(d)

            if not target_objects:
                continue

            best = max(target_objects, key=lambda d: float(d.get("conf",0)))
            box = tuple(map(float, best["xyxy"]))

            # temporal dedupe
            now = time.time()
            recent = [(b, t) for (b, t) in recent if now - t <= DEDUPE_COOLDOWN]

            if any(box_iou(box, b) >= DEDUPE_IOU_THR for (b, t) in recent):
                continue

            okc, c = crop(frame, best["xyxy"])

            if not okc:
                continue

            # backpressure: only submit if worker is idle

            if molmo.submit(c):
                recent.append((box, now))

            # else silently skip; worker is busy
            time.sleep(0.01)

    finally:

        try:
            molmo.close()
            cap.release()
            cv2.destroyAllWindows()

        except Exception:
            pass

        print("[camera] closed")

if __name__ == "__main__":

    main()
