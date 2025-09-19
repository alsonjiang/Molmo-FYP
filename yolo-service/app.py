import time, os, base64, cv2, torch, requests
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import base64, io, cv2
from PIL import Image
import numpy as np

app = FastAPI()

model = YOLO("yolo11n.pt")
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else "cpu"

CONF_THR = float(os.getenv("CONF_THR", "0.55"))
IOU_THR  = float(os.getenv("IOU_THR", "0.35"))

class DetectIn(BaseModel):
    image_b64: str

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "conf_thr": CONF_THR,
        "iou_thr": IOU_THR
    }

def b64_to_bgr(b64):
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

@app.post("/detect")
def detect(inp: DetectIn):
    frame = b64_to_bgr(inp.image_b64)
    t0 = time.time()
    r = model.predict(
        frame, imgsz=640,
        conf=CONF_THR, iou=IOU_THR,
        device=device, half=use_cuda,   # <— GPU + FP16 when available
        verbose=False
    )[0]
    dt_ms = (time.time() - t0) * 1000.0

    dets = []
    for b in r.boxes:
        cid = int(b.cls[0].item())
        cls = r.names[cid]
        if cls.lower() != "person":     # only person
            continue
        dets.append({
            "xyxy": [float(x) for x in b.xyxy[0].tolist()],
            "cls_id": cid, "cls": cls,
            "conf": float(b.conf[0].item())
        })
    return {"detections": dets, "inference_time_ms": round(dt_ms, 2)}
