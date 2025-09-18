from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import base64, io
from PIL import Image

model = YOLO("yolo11n.pt")   # auto-download on first run
app = FastAPI()

class DetectIn(BaseModel):
    image_b64: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/detect")
def detect(inp: DetectIn):
    img = Image.open(io.BytesIO(base64.b64decode(inp.image_b64))).convert("RGB")
    r = model.predict(img, verbose=False)[0]
    out = []
    for xyxy, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        out.append({
            "xyxy": [float(x) for x in xyxy],
            "cls_id": int(cls),
            "cls": r.names[int(cls)],
            "conf": float(conf)
        })
    return {"detections": out}
