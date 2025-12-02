# test_yolo.py
import base64, requests, cv2

img = cv2.imread("../images/portrait.jpg")
_, buf = cv2.imencode(".jpg", img)
b64 = base64.b64encode(buf.tobytes()).decode()

r = requests.post(
    "http://localhost:9000/detect",  # or whatever YOLO_URL actually is
    json={"image_b64": b64},
    timeout=30,
)
print(r.status_code)
print(r.text[:500])