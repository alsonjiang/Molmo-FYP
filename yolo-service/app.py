# yolo-service/app.py
import os, time, base64, io
import numpy as np
import torch
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import uvicorn

app = FastAPI()

# --- Config ---
WEIGHTS    = os.getenv("YOLO_WEIGHTS", "yolo11n.pt")
CONF_THR   = float(os.getenv("CONF_THR", "0.35"))   # server-side conf
IOU_THR    = float(os.getenv("IOU_THR",  "0.55"))
IMG_SIZE   = int(os.getenv("IMG_SIZE",  "640"))
FP16_AFTER = os.getenv("YOLO_FP16", "0") == "1"     # set YOLO_FP16=1 to run FP16 after fusing

use_cuda = torch.cuda.is_available()
device   = 0 if use_cuda else "cpu"

# --- Load model ---
model = YOLO(WEIGHTS)

# 1) Make sure we are in FP32 for fuse
try:
    # Some Ultralytics versions expose the underlying torch nn.Module as model.model
    if hasattr(model, "model") and model.model is not None:
        model.model.float()
    else:
        # fallback: this no-ops for some versions
        model.to("cpu")
except Exception as e:
    print("[yolo] float() set failed:", e)

# 2) Run a one-time warmup/fuse in FP32 (half=False)
try:
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model.predict(
        dummy,
        imgsz=IMG_SIZE,
        conf=CONF_THR,
        iou=IOU_THR,
        device=device,
        half=False,     # IMPORTANT: keep False during fuse/warmup
        verbose=False
    )
    print("[yolo] warmup+fuse complete (FP32)")
except Exception as e:
    print("[yolo] warmup/fuse failed:", e)

# 3) Optionally switch the fused model to FP16 for faster GPU inference
if use_cuda and FP16_AFTER:
    try:
        if hasattr(model, "model") and model.model is not None:
            model.model.half()
            print("[yolo] model cast to FP16 after fuse")
        else:
            print("[yolo] could not access model.model for half()")
    except Exception as e:
        print("[yolo] half() set failed:", e)

# NOTE: From this point on, the underlying model weights are already the right dtype.
# We will keep predict(half=False) so Ultralytics does not try to recast/fuse again.

class DetectIn(BaseModel):
    image_b64: str

def b64_to_bgr(b64: str):
    try:
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("cv2.imdecode returned None")
        return im
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {e}")

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": "cuda" if use_cuda else "cpu",
        "conf_thr": CONF_THR,
        "iou_thr": IOU_THR,
        "img_size": IMG_SIZE,
        "fp16_active": bool(use_cuda and FP16_AFTER),
    }

@app.post("/detect")
def detect(inp: DetectIn):
    frame = b64_to_bgr(inp.image_b64)

    t0 = time.time()
    # IMPORTANT: half=False because the underlying model is already fused and (maybe) half.
    r = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THR,
        iou=IOU_THR,
        device=device,
        half=False,      # <- leave False; dtype is controlled by model.model above
        verbose=False
    )[0]
    dt_ms = (time.time() - t0) * 1000.0

    dets = []
    names = r.names
    for b in r.boxes:
        cid = int(b.cls[0].item())
        cls = names[cid]
        if cls.lower() != "person":     # only person
            continue
        dets.append({
            "xyxy": [float(x) for x in b.xyxy[0].tolist()],
            "cls_id": cid, "cls": cls,
            "conf": float(b.conf[0].item())
        })
    return {"detections": dets, "inference_time_ms": round(dt_ms, 2)}

if __name__ == '__main__':
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "9000"))  # YOLO on 9000
    # Use 1 worker so the warmup/fuse runs once (multiple workers would rerun imports)
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False, workers=1, access_log=False) #turned off logging for this service
