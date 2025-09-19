import os, io, base64, time, cv2, requests
import numpy as np

YOLO_URL   = os.getenv("YOLO_URL", "http://localhost:9000/detect")
CAM_INDEX  = int(os.getenv("CAM_INDEX", "0"))
MAX_SIDE   = int(os.getenv("MAX_SIDE", "640"))     # max image side sent to YOLO
JPEG_Q     = int(os.getenv("JPEG_Q", "60"))        # JPEG quality (lower = faster/smaller)
DETECT_EVERY = int(os.getenv("DETECT_EVERY", "1")) # send every Nth frame (1 = every frame)
WINDOW_NAME = "YOLO Only View"

# Keep one session to reuse TCP connection
SESSION = requests.Session()
SESSION.headers.update({"Connection": "keep-alive"})

def resize_keep_aspect(bgr, max_side=640):
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def encode_b64_cv(bgr, quality=60):
    # Faster than PIL: jpeg encode directly from numpy
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf).decode("ascii")

def draw_boxes(frame_bgr, dets):
    h, w = frame_bgr.shape[:2]
    for d in dets:
        cls = str(d.get("cls", ""))
        conf = float(d.get("conf", 0.0))
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 2, y1), (0, 255, 0), -1)
        cv2.putText(frame_bgr, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def main():
    # Windows camera backend that’s faster/more reliable
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    # Reduce OpenCV internal thread overhead (helps sometimes)
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if not cap.isOpened():
        print(f"[error] cannot open camera {CAM_INDEX}")
        return

    frame_count = 0
    fps = 0.0
    t_fps = time.time()
    last_dets = []
    last_infer_ms = 0.0
    i = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # Downscale BEFORE sending over HTTP
            frame_small = resize_keep_aspect(frame, MAX_SIDE)

            # Send every Nth frame to YOLO (reduce HTTP load)
            i += 1
            send_to_server = (i % DETECT_EVERY == 0)

            if send_to_server:
                try:
                    b64 = encode_b64_cv(frame_small, JPEG_Q)
                    resp = SESSION.post(YOLO_URL, json={"image_b64": b64}, timeout=3)
                    resp.raise_for_status()
                    out = resp.json()
                    last_dets = out.get("detections", []) or []
                    last_infer_ms = float(out.get("inference_time_ms", 0.0))
                except Exception as e:
                    # On error, keep last_dets so display doesn’t flicker
                    # Print occasionally to avoid spamming
                    print("[yolo error]", e)
                    last_infer_ms = 0.0

            # Draw last results on the full frame for nicer display
            vis = frame.copy()
            draw_boxes(vis, last_dets)

            # Client FPS (end-to-end)
            frame_count += 1
            if frame_count >= 10:
                now = time.time()
                fps = frame_count / (now - t_fps)
                frame_count = 0
                t_fps = now

            # HUD
            cv2.putText(vis, f"FPS (client): {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if last_infer_ms:
                cv2.putText(vis, f"YOLO infer: {last_infer_ms:.1f} ms", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(vis, f"Detections: {len(last_dets)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[camera] closed")

if __name__ == "__main__":
    main()
