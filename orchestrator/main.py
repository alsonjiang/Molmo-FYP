import os, sys, time, io, base64, subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2, requests
from PIL import Image
import threading, queue

PROMPT = 'Describe the person in the image. Answer "None" if there is no person.'

YOLO_URL  = os.getenv("YOLO_URL",  "http://localhost:9000/detect")
MOLMO_URL = os.getenv("MOLMO_URL", "http://localhost:8000/caption")
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
CONF_THR  = float(os.getenv("CONF_THR", "0.35"))
MOLMO_TIMEOUT_S = float(os.getenv("MOLMO_TIMEOUT_S", "5.0"))

COMPOSE_FILE = os.getenv("COMPOSE_FILE", str(Path(__file__).resolve().parents[1] / "robot-stack" / "docker-compose.yml"))
DOCKER_COMPOSE = os.getenv("DOCKER_COMPOSE", "docker compose")

WINDOW_NAME = "YOLO view"

def draw_boxes(frame_bgr, dets, persons_only=True):
    """Draw rectangles and labels on the frame."""
    import cv2
    h, w = frame_bgr.shape[:2]
    for d in dets:
        cls = str(d.get("cls", ""))
        cls_id = d.get("cls_id", None)
        conf = float(d.get("conf", 0.0))
        if persons_only and not (cls.lower()=="person" or cls_id == 0):
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
    """Show the frame and a status overlay; keep UI responsive."""
    import cv2, time
    if status_text:
        cv2.putText(frame_bgr, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imshow(WINDOW_NAME, frame_bgr)
    # Needed for the OS window to update; don't rely on this for commands (we use the input thread)
    cv2.waitKey(1)

def encode_b64(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def crop(frame_bgr, xyxy) -> Tuple[bool, any]:
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    h,w = frame_bgr.shape[:2]
    x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(w,x2), min(h,y2)
    c = frame_bgr[y1:y2, x1:x2]
    return (c.size > 0, c)

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
    """
    Background thread: blocks on input() and pushes lines into a Queue.
    Works on Windows CMD/PowerShell and Unix shells.
    """
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

def main():
    print("=== Orchestrator CLI ===", flush=True)
    print("type 'off' to pause detection, 'on' to resume, 'prompt <text>' to change, 'q' to quit.", flush=True)

    cap = cv2.VideoCapture(CAM_INDEX)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if not cap.isOpened():
        print(f"[error] cannot open camera {CAM_INDEX}"); sys.exit(2)

    prompt = PROMPT
    detect_on = True
    last_status = 0.0
    cmd_q = start_cmd_reader()

    try:
        while True:
            # 1) Drain any pending commands (non-blocking)
            while not cmd_q.empty():
                cmdline = cmd_q.get_nowait()
                if not cmdline:
                    continue
                low = cmdline.lower()
                if low in ("quit", "exit", "q"):
                    print("[cmd] quitting…")
                    return
                elif low == "on":
                    detect_on = True
                    print("[cmd] detection ON")
                elif low == "off":
                    detect_on = False
                    print("[cmd] detection OFF")
                elif low == "status":
                    print(f"[status] detection={'ON' if detect_on else 'OFF'} | prompt='{prompt}'")
                elif low.startswith("prompt "):
                    prompt = cmdline[7:].strip()
                    print(f"[cmd] updated prompt → {prompt}")
                else:
                    print(f"[cmd] unknown: {cmdline}")

            # Grab a frame
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # If detection is OFF, just show the camera
            if not detect_on:
                if time.time() - last_status > 2.0:
                    print("[status] detection OFF")
                    last_status = time.time()
                show_frame(frame.copy(), status_text="DETECTION OFF")
                time.sleep(0.02)
                continue

            # Call YOLO
            try:
                resp = requests.post(YOLO_URL, json={"image_b64": encode_b64(frame)}, timeout=5)
                resp.raise_for_status()
                dets = resp.json().get("detections", [])
            except Exception as e:
                print("[yolo error]", e)
                show_frame(frame.copy(), status_text="YOLO ERROR")
                time.sleep(0.2)
                continue

            # Draw ALL detections for visualization (set persons_only=True to show only persons)
            vis = frame.copy()
            draw_boxes(vis, dets, persons_only=False)
            show_frame(vis, status_text="DETECTION ON")

            # Filter for persons for decision logic
            persons = [d for d in dets if ((d.get("cls","").lower()=="person") or d.get("cls_id")==0)
                    and float(d.get("conf",0)) >= CONF_THR]
            if not persons:
                continue


            # Crop best and call Molmo
            best = max(persons, key=lambda d: float(d.get("conf",0)))
            okc, c = crop(frame, best["xyxy"])
            if not okc:
                continue

            try:
                resp = requests.post(MOLMO_URL, json={"image_b64": encode_b64(c), "prompt": prompt}, timeout=MOLMO_TIMEOUT_S)
                resp.raise_for_status()
                molmo = resp.json()
            except requests.Timeout:
                stop_yolo_and_exit()
            except Exception as e:
                print("[molmo error]", e)
                continue

            text = (molmo.get("caption") or molmo.get("text") or "").strip()
            print(f"[molmo] response: {text}", flush=True)

            # 7) Restart YOLO after a successful Molmo round
            restart_yolo()
            time.sleep(0.02)

    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        print("[camera] closed")

if __name__ == "__main__":
    main()
