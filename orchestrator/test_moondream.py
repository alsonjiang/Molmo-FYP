import base64, requests

with open("../images/portrait.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

r = requests.post(
    "http://localhost:8000/caption",
    json={
        "image_b64": b64,
        "prompt": "Very briefly: what is in this image?",
    },
    timeout=120,  # give it plenty of time
)
print(r.status_code)
print(r.text)