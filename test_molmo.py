#!/usr/bin/env python3
"""
Test Molmo service end-to-end.

- Health check (/health)
- Debug ping (/debug_ping)
- Caption a real image (/caption)

Usage (from your project root):
  python test_molmo.py
  python test_molmo.py --url http://localhost:8000 --image C:/Molmo-FYP/images/clock_face.png
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests


def encode_image_b64(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with path.open("rb") as f:
        data = f.read()
    if not data:
        raise ValueError(f"Image file is empty: {path}")
    return base64.b64encode(data).decode("ascii")


def pretty(obj):
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def get_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text}


def main():
    default_root = Path.cwd()
    default_img = (default_root / "images" / "clock_face.png")

    ap = argparse.ArgumentParser(description="Molmo service tester")
    ap.add_argument("--url", default="http://localhost:8000", help="Molmo base URL (no trailing slash)")
    ap.add_argument("--image", default=str(default_img), help="Path to test image")
    ap.add_argument("--prompt", default="Describe the image.", help="Prompt to send")
    ap.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    health_url = f"{base}/health"
    ping_url = f"{base}/debug_ping"
    caption_url = f"{base}/caption"

    print(f"[cfg] MOLMO_URL={base}")
    print(f"[cfg] IMAGE={args.image}")
    print(f"[cfg] TIMEOUT={args.timeout}s\n")

    # 1) /health
    try:
        t0 = time.time()
        r = requests.get(health_url, timeout=args.timeout)
        dt = time.time() - t0
        print(f"[health] {r.status_code} in {dt:.2f}s")
        print(pretty(get_json(r)), "\n")
        if not r.ok:
            print("[fail] Molmo /health not OK — fix service before testing /caption")
            sys.exit(2)
    except Exception as e:
        print(f"[error] /health request failed: {e}")
        sys.exit(2)


    # 3) /caption with real image
    img_path = Path(args.image)
    try:
        img_b64 = encode_image_b64(img_path)
    except Exception as e:
        print(f"[error] failed to read image: {e}")
        sys.exit(2)

    payload = {"image_b64": img_b64, "prompt": args.prompt}
    try:
        t0 = time.time()
        r = requests.post(caption_url, json=payload, timeout=args.timeout)
        dt = time.time() - t0
        print(f"[caption] {r.status_code} in {dt:.2f}s")

        if r.status_code >= 400:
            # Show server-provided error details so you know exactly what's wrong
            try:
                print(pretty(r.json()))
            except Exception:
                print(r.text[:1000])
            sys.exit(3)

        data = r.json()
        caption = (data.get("caption") or data.get("text") or "").strip()
        print("\n--- CAPTION ---")
        print(caption or "<empty>")
        print("---------------")
        sys.exit(0)

    except requests.Timeout:
        print(f"[error] /caption timed out after {args.timeout:.1f}s")
        sys.exit(3)
    except Exception as e:
        print(f"[error] /caption request failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
