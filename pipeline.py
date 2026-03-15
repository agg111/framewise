"""
Persistent Scene KV Cache for Robot Perception
Naive vs Cached VLM inference — same Nebius GPU, same model, different prompt strategy.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()
import base64
import cv2
import numpy as np
from openai import OpenAI

NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"

# Resize frames to this width before sending (keeps aspect ratio)
INFERENCE_WIDTH = 640

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=NEBIUS_API_KEY,
)

STATIC_SCENE_PROMPT = (
    "You are a robot perception module. The scene contains objects on a table. "
    "Your job is to identify NEW or MOVED objects in each frame update."
)


def resize_frame(frame: np.ndarray, width: int = INFERENCE_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def encode_frame(frame: np.ndarray) -> tuple[str, int]:
    """Encode a numpy BGR frame to base64 JPEG. Returns (b64_string, byte_size)."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode("utf-8")
    return b64, len(buf)


def extract_delta(prev: np.ndarray, curr: np.ndarray, threshold: int = 35) -> tuple[np.ndarray, bool]:
    """
    Return (delta_crop, has_change).
    Crops the bounding box around changed pixels.
    """
    # Blur before diff to suppress camera shake / compression noise
    prev_b = cv2.GaussianBlur(prev, (7, 7), 0)
    curr_b = cv2.GaussianBlur(curr, (7, 7), 0)

    diff = cv2.absdiff(prev_b, curr_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Ignore tiny contours (camera shake / noise) — require at least 1% of frame
    H, W = curr.shape[:2]
    min_area = 0.01 * H * W
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not contours:
        return curr, False  # no meaningful change detected

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    pad = 20
    H, W = curr.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)

    return curr[y1:y2, x1:x2], True


# ─── Naive mode ────────────────────────────────────────────────────────────────

def infer_naive(frame: np.ndarray, query: str) -> tuple[str, float, int]:
    """Send full frame every time. Returns (text, latency, bytes_sent)."""
    small = resize_frame(frame)
    b64, nbytes = encode_frame(small)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You are a robot perception module. Describe what you see."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": query},
                ],
            },
        ],
        max_tokens=128,
    )
    latency = time.perf_counter() - t0
    return resp.choices[0].message.content, latency, nbytes


# ─── Cached mode ───────────────────────────────────────────────────────────────

class CachedPipeline:
    """
    Sends STATIC_SCENE_PROMPT as a fixed system message (vLLM prefix cache hit).
    Each frame only sends the delta crop, not the full frame.
    """

    def __init__(self):
        self.prev_frame: np.ndarray | None = None
        self.last_response: str = ""
        self.last_delta: np.ndarray | None = None

    def infer(self, frame: np.ndarray, query: str) -> tuple[str, float, bool, int, np.ndarray]:
        """Returns (text, latency, used_delta, bytes_sent, delta_frame)."""
        small = resize_frame(frame)

        if self.prev_frame is None:
            # First frame — must call model to establish scene
            delta, has_change = small, False
            self.prev_frame = small.copy()
        else:
            delta, has_change = extract_delta(self.prev_frame, small)
            self.prev_frame = small.copy()
            if not has_change:
                # Scene unchanged — skip API call entirely, return cached response
                return self.last_response + " [CACHED — no change]", 0.0, False, 0, small

        b64, nbytes = encode_frame(delta)

        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": STATIC_SCENE_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": query},
                    ],
                },
            ],
            max_tokens=128,
        )
        latency = time.perf_counter() - t0
        self.last_response = resp.choices[0].message.content
        return self.last_response, latency, has_change, nbytes, delta
