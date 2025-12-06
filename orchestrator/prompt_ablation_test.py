#Activate whichever VLM you want to test at :8000 first
#this script takes three images. One will be used for semantic description, Two will be used for identity comparison
#Its purpose is to test the effects of different prompts on the same set of images
#There are 3 prompts on semantics, and 3 prompts on identity. You can add more if you'd like
#The output is printed onto the console

import base64
import io
import os
import time
from typing import List, Tuple

import requests
from PIL import Image

# URL for your Molmo/Moondream service
MOLMO_URL = os.getenv("MOLMO_URL", "http://localhost:8000/caption")

# === CHANGE THESE PATHS ===
SINGLE_IMG_PATH = r"..\images\me_1.png"       # for semantic description tests
LEFT_IMG_PATH   = r"..\images\me_1.png"       # left image for identity comparison
RIGHT_IMG_PATH  = r"..\images\me_2.png"       # right image for identity comparison


# ===========================
# Utility functions
# ===========================

def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def image_to_b64(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def concat_side_by_side(left: Image.Image, right: Image.Image, gap: int = 8) -> Image.Image:
    """
    Create a simple LEFT|RIGHT panel for identity comparison.
    Resizes both to the same height while keeping aspect ratio.
    """
    # Make both same height
    h = 256
    def resize_keep_aspect(im: Image.Image, target_h: int) -> Image.Image:
        w, h0 = im.size
        scale = target_h / float(h0)
        new_w = max(1, int(round(w * scale)))
        return im.resize((new_w, target_h), Image.LANCZOS)

    left_resized = resize_keep_aspect(left, h)
    right_resized = resize_keep_aspect(right, h)

    w_left, h_left = left_resized.size
    w_right, h_right = right_resized.size

    panel_w = w_left + gap + w_right
    panel_h = max(h_left, h_right)

    panel = Image.new("RGB", (panel_w, panel_h), (0, 0, 0))
    panel.paste(left_resized, (0, 0))
    panel.paste(right_resized, (w_left + gap, 0))

    return panel


def call_vlm(image: Image.Image, prompt: str, timeout_s: float = 60.0) -> Tuple[int, str, float]:
    """
    Send a single image + prompt to /caption and return:
    (status_code, raw_text_response, latency_ms)
    """
    img_b64 = image_to_b64(image)
    payload = {
        "image_b64": img_b64,
        "prompt": prompt,
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(MOLMO_URL, json=payload, timeout=timeout_s)
    except Exception as e:
        return 0, f"[request failed: {repr(e)}]", 0.0
    dt_ms = (time.perf_counter() - t0) * 1000.0

    text = r.text
    return r.status_code, text, dt_ms


# ===========================
# Prompt ablation tests
# ===========================

def run_semantic_prompt_tests(img_path: str, prompts: List[str], repeats: int = 3):
    """
    Test how different prompts affect semantic descriptions for the same image.
    Also measures stability across repeated calls for each prompt.
    """
    print("\n================ SEMANTIC DESCRIPTION PROMPT TESTS ================\n")
    img = load_image(img_path)

    for idx, prompt in enumerate(prompts, start=1):
        print(f"\n--- Prompt {idx} ----------------------------------------------")
        print(f"Prompt text:\n{prompt}\n")

        for i in range(repeats):
            status, text, dt_ms = call_vlm(img, prompt)
            print(f"[Run {i+1}] HTTP {status}, latency={dt_ms:.1f} ms")
            print(f"[Run {i+1}] Raw response:\n{text}\n")


def run_identity_prompt_tests(left_path: str, right_path: str, prompts: List[str], repeats: int = 3):
    """
    Test how different prompts affect identity-comparison output on the same pair.
    The image pair is constructed once and reused.
    """
    print("\n================ IDENTITY COMPARISON PROMPT TESTS ================\n")
    left = load_image(left_path)
    right = load_image(right_path)
    panel = concat_side_by_side(left, right)

    # Optional: show the panel once
    try:
        panel.show(title="LEFT | RIGHT panel used for identity comparison")
    except Exception:
        pass

    for idx, prompt in enumerate(prompts, start=1):
        print(f"\n--- Prompt {idx} ----------------------------------------------")
        print(f"Prompt text:\n{prompt}\n")

        for i in range(repeats):
            status, text, dt_ms = call_vlm(panel, prompt)
            print(f"[Run {i+1}] HTTP {status}, latency={dt_ms:.1f} ms")
            print(f"[Run {i+1}] Raw response:\n{text}\n")


def main():
    # Example semantic prompts: generic vs engineered
    semantic_prompts = [
        # Generic / weak
        "Describe the image.",
        # Attribute-focused
        "Describe the person's appearance. Focus only on clothing colour, hairstyle and any visible distinguishing features. "
        "Keep the answer to one short sentence.",
        # Strongly constrained
        "Describe only the person's visible clothing colour, hairstyle and one distinctive feature. "
        "Do not mention background or speculate. Use at most 20 words."
    ]

    # Example identity prompts: generic vs strict
    identity_prompts = [
        # Generic
        "Are the two people in this image the same person?",
        # More guided
        "You see two people side by side. Compare their facial features and say whether they are the same person or different people.",
        # Strict engineered
        "You will see two faces side by side. Compare only facial features such as eyes, eyebrows, nose, mouth and face shape. "
        "Ignore lighting, background and pose. Answer with exactly one line: 'same person' or 'different person'."
    ]

    # Number of times to repeat each prompt to test stability
    repeats = 3

    # Run tests
    run_semantic_prompt_tests(SINGLE_IMG_PATH, semantic_prompts, repeats=repeats)
    run_identity_prompt_tests(LEFT_IMG_PATH, RIGHT_IMG_PATH, identity_prompts, repeats=repeats)


if __name__ == "__main__":
    main()
