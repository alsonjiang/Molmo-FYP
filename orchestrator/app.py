import os, sys, time, io, base64, subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2, requests
from PIL import Image

PROMPT = 'Is this a person? Reply Yes or No.'

YOLO_URL  = os.getenv("YOLO_URL",  "http://localhost:9000/detect")
MOLMO_URL = os.getenv("MOLMO_URL", "http://localhost:8000/caption")
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
CONF_THR  = float(os.getenv("CONF_THR", "0.35"))
MOLMO_TIMEOUT_S = float(os.getenv("MOLMO_TIMEOUT_S", "5.0"))

COMPOSE_FILE = os.getenv("COMPOSE_FILE", str(Path(__file__).resolve().parents[1] / "robot-stack" / "docker-compose.yml"))
DOCKER_COMPOSE = os.getenv("DOCKER_COMPOSE", "docker compose")

def encode_b64(frame_bgr):
    import numpy as np
    from PIL import Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def crop(frame_bgr, xyxy) -> Tuple[bool, any]:
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    h,w = frame_bgr.shape[:2]
    x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(w,x2), min(h,y2)
    c = frame_bgr[y1:y2, x1:x2]
    return (c.size>0, c)

def stop_yolo_and_exit():
    print(f"[orchestrator] Molmo exceeded {MOLMO_TIMEOUT_S:.1f}s → stopping yolo-service and exiting.", flush=True)
    try:
        subprocess.run(f'{DOCKER_COMPOSE} -f "{COMPOSE_FILE}" stop yolo', shell=True)
    finally:
        sys.exit(1)

def restart_yolo():
    print("[orchestrator] Restarting yolo-service…", flush=True)
    subprocess.run(f'{DOCKER_COMPOSE} -f "{COMPOSE_FILE}" restart yolo', shell=True)

def main():
    print("=== Orchestrator CLI ===", flush=True)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[error] cannot open camera {CAM_INDEX}"); sys.exit(2)

    prompt = PROMPT
    detect_on = True
    print("[tip] type 'off' to pause detection, 'on' to resume, 'prompt <text>' to change, 'q' to quit.", flush=True)

    import select
    last_status = 0.0
    while True:
        if select.select([sys.stdin], [], [], 0.0)[0]:
            cmd = sys.stdin.readline().strip()
            if cmd.lower() in ("q","quit","exit"): break
            if cmd.lower()=="off": detect_on=False; print("[cmd] detection OFF")
            elif cmd.lower()=="on": detect_on=True; print("[cmd] detection ON")
            elif cmd.lower().startswith("prompt "): prompt=cmd[7:]; print(f"[cmd] new prompt → {prompt}")
            else: print(f"[cmd] unknown: {cmd}")

        ok, frame = cap.read()
        if not ok: time.sleep(0.01); continue
        if not detect_on:
            if time.time()-last_status>2: print("[status] detection OFF"); last_status=time.time()
            time.sleep(0.02); continue

        # call YOLO
        try:
            dets = requests.post(YOLO_URL, json={"image_b64": encode_b64(frame)}, timeout=5).json().get("detections", [])
        except Exception as e:
            print("[yolo error]", e); time.sleep(0.2); continue

        persons = [d for d in dets if ((d.get("cls","").lower()=="person") or d.get("cls_id")==0) and float(d.get("conf",0))>=CONF_THR]
        if not persons: continue

        best = max(persons, key=lambda d: float(d.get("conf",0)))
        okc, c = crop(frame, best["xyxy"]) 
        if not okc: continue

        try:
            molmo = requests.post(MOLMO_URL, json={"image_b64": encode_b64(c), "prompt": prompt}, timeout=MOLMO_TIMEOUT_S).json()
        except requests.Timeout:
            stop_yolo_and_exit()
        except Exception as e:
            print("[molmo error]", e); continue

        text = (molmo.get("caption") or molmo.get("text") or "").strip()
        verdict = "YES" if "YES" in text.upper() and "NO" not in text.upper() else ("NO" if "NO" in text.upper() else "UNKNOWN")
        print(f"[molmo] verdict={verdict} | {text}", flush=True)

        restart_yolo()
        time.sleep(0.02)

    cap.release(); print("[camera] closed")

if __name__ == "__main__":
    main()
