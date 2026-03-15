"""
Persistent Scene KV Cache for Robot Perception
Naive vs Cached VLM inference — same Nebius GPU, same model, different prompt strategy.

Two-level change detection:
  Level 1 — Pixel diff (OpenCV, free, instant)
  Level 2 — Semantic diff (Nebius embeddings, catches false positives from pixel noise)
"""

import os
import re
import json
import time
import base64
import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
SMALL_MODEL_ID = "google/gemma-3-27b-it"  # 27B — smaller model comparison baseline
MINIMAX_MODEL_ID = "google/gemma-3-27b-it"
EMBED_MODEL = "BAAI/bge-en-icl"

# Resize frames to this width before sending (keeps aspect ratio)
INFERENCE_WIDTH = 640

# Cosine similarity threshold — above this means "scene is semantically the same"
SEMANTIC_SIMILARITY_THRESHOLD = 0.92

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=NEBIUS_API_KEY,
)

STATIC_SCENE_PROMPT = (
    "You are a robot perception module. The scene contains objects on a table. "
    "Your job is to identify NEW or MOVED objects in each frame update."
)

DELTA_QUERY = (
    "Something changed in the scene. What is the new or moved object? "
    "Reply in 2 lines: 1) What changed. 2) What action a robot arm should take."
)

CONFIDENCE_THRESHOLD = 80  # % — below this triggers human confirmation request

# ─── RAG: Robot Safety Knowledge Base ─────────────────────────────────────────

SAFETY_RULES = [
    "Fragile objects near table edge: gently push toward center to prevent falling.",
    "Sharp or pointed objects detected: use gripper carefully, avoid contact with blade or tip.",
    "Liquid containers or bottles: keep upright, move slowly to prevent spilling.",
    "Electronics or screens detected: handle with care, avoid static discharge and pressure.",
    "Small or lightweight objects: use precision gripper with low grip force.",
    "Stacked or unstable objects: stabilize the stack before any manipulation.",
    "Glasses or eyewear on table: pick up gently by the frame, avoid lens contact.",
    "Food items detected: use clean gripper, avoid contamination.",
    "Multiple objects clustered: clear space around target before gripping.",
    "Unknown or unrecognized object: pause and request human confirmation before acting.",
]

class RobotKnowledgeBase:
    """
    Embeds safety rules at startup using Nebius embeddings.
    On change detection, retrieves the most relevant rule via cosine similarity.
    """
    def __init__(self):
        self.rules = SAFETY_RULES
        self.embeddings: list[list[float]] = []
        self._build()

    def _build(self):
        for rule in self.rules:
            emb = get_embedding(rule)
            if emb:
                self.embeddings.append(emb)
            else:
                self.embeddings.append([])

    def query(self, scene_description: str, top_k: int = 2) -> list[str]:
        """Return the top_k most relevant safety rules for this scene."""
        query_emb = get_embedding(scene_description)
        if not query_emb or not self.embeddings:
            return []
        scores = []
        for i, emb in enumerate(self.embeddings):
            if emb:
                scores.append((cosine_similarity(query_emb, emb), i))
        scores.sort(reverse=True)
        return [self.rules[i] for _, i in scores[:top_k]]


# Singleton — initialized once at startup
robot_kb: RobotKnowledgeBase | None = None

def get_robot_kb() -> RobotKnowledgeBase:
    global robot_kb
    if robot_kb is None:
        robot_kb = RobotKnowledgeBase()
    return robot_kb

SCENE_INVENTORY_PROMPT = (
    "You are a robot perception module. Look at this scene and return a JSON object with this exact format: "
    '{{"objects": ["object1", "object2", ...], "new_object": "name of any newly placed object or null", "robot_action": "recommended action or null"}}. '
    "Return only valid JSON, no other text."
)


# ─── Utilities ─────────────────────────────────────────────────────────────────

def resize_frame(frame: np.ndarray, width: int = INFERENCE_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def encode_frame(frame: np.ndarray) -> tuple[str, int]:
    """Encode a numpy BGR frame to base64 JPEG. Returns (b64_string, byte_size)."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode("utf-8")
    return b64, len(buf)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ─── Nebius Embeddings ─────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float] | None:
    """Get text embedding from Nebius using BAAI/bge-en-icl."""
    try:
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return resp.data[0].embedding
    except Exception:
        return None


# ─── Pixel-level change detection ─────────────────────────────────────────────

def extract_delta(prev: np.ndarray, curr: np.ndarray, threshold: int = 35) -> tuple[np.ndarray, bool, tuple | None]:
    """
    Returns (delta_crop, has_change, bbox).
    Uses Gaussian blur to suppress camera shake.
    """
    prev_b = cv2.GaussianBlur(prev, (7, 7), 0)
    curr_b = cv2.GaussianBlur(curr, (7, 7), 0)

    diff = cv2.absdiff(prev_b, curr_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = curr.shape[:2]
    min_area = 0.01 * H * W
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not contours:
        return curr, False, None

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    pad = 20
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)

    return curr[y1:y2, x1:x2], True, (x1, y1, x2, y2)


def draw_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int], label: str = "CHANGE DETECTED") -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(out, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


# ─── Scene inventory ───────────────────────────────────────────────────────────

def get_scene_inventory(frame: np.ndarray) -> dict:
    small = resize_frame(frame)
    b64, _ = encode_frame(small)
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SCENE_INVENTORY_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Analyze this scene."},
                ]},
            ],
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except Exception:
        return {}


# ─── Reranking ────────────────────────────────────────────────────────────────

RERANK_MODEL = "BAAI/bge-en-icl"

def rerank_actions(actions: list[str], query: str = "safest and most actionable robot instruction") -> tuple[str, list[float]]:
    """
    Use Nebius reranking API to score robot action candidates.
    Returns (best_action, scores).
    """
    try:
        resp = client.post(
            "/rerank",
            body={
                "model": RERANK_MODEL,
                "query": query,
                "documents": actions,
                "top_n": len(actions),
            },
            cast_to=object,
        )
        results = resp.get("results", []) if isinstance(resp, dict) else []
        if not results:
            return actions[0], []
        scored = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
        scores = [r.get("relevance_score", 0) for r in scored]
        best_idx = scored[0].get("index", 0)
        return actions[best_idx], scores
    except Exception as e:
        return actions[0], []


# ─── Batch inference ───────────────────────────────────────────────────────────

def submit_batch(frames: list[np.ndarray], query: str) -> str:
    """
    Submit all frames as a Nebius batch job.
    Returns batch_id.
    """
    import tempfile, json as _json

    lines = []
    for i, frame in enumerate(frames):
        small = resize_frame(frame)
        b64, _ = encode_frame(small)
        req = {
            "custom_id": f"frame-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_ID,
                "messages": [
                    {"role": "system", "content": STATIC_SCENE_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": query},
                    ]},
                ],
                "max_tokens": 128,
            },
        }
        lines.append(_json.dumps(req))

    # Write JSONL to temp file and upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines))
        tmp_path = f.name

    with open(tmp_path, "rb") as f:
        file_resp = client.files.create(file=f, purpose="batch")

    batch_resp = client.batches.create(
        input_file_id=file_resp.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch_resp.id


def get_batch_results(batch_id: str) -> tuple[str, list[dict]]:
    """
    Poll batch status. Returns (status, results_list).
    results_list is empty until batch completes.
    """
    import json as _json

    batch = client.batches.retrieve(batch_id)
    status = batch.status

    if status != "completed":
        return status, []

    content = client.files.content(batch.output_file_id)
    results = []
    for line in content.text.strip().split("\n"):
        if line:
            obj = _json.loads(line)
            frame_id = obj.get("custom_id", "")
            try:
                text = obj["response"]["body"]["choices"][0]["message"]["content"]
            except Exception:
                text = "error"
            results.append({"frame": frame_id, "text": text})

    return status, results


# ─── MiniMax mode ─────────────────────────────────────────────────────────────

def infer_minimax(delta_frame: np.ndarray, query: str) -> tuple[str, float, int]:
    """Run MiniMax-M2.1 on the delta crop. Returns (text, latency, bytes_sent)."""
    b64, nbytes = encode_frame(delta_frame)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MINIMAX_MODEL_ID,
        messages=[
            {"role": "system", "content": STATIC_SCENE_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": query},
            ]},
        ],
        max_tokens=128,
    )
    latency = time.perf_counter() - t0
    return resp.choices[0].message.content, latency, nbytes


# ─── Streaming inference ───────────────────────────────────────────────────────

def infer_streaming(frame: np.ndarray, query: str):
    """
    Stream tokens from Nebius for a single frame.
    Yields (token_chunk, is_done, elapsed_ms).
    """
    small = resize_frame(frame)
    b64, nbytes = encode_frame(small)
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": STATIC_SCENE_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": query},
            ]},
        ],
        max_tokens=128,
        stream=True,
    )
    accumulated = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        accumulated += delta
        elapsed = (time.perf_counter() - t0) * 1000
        is_done = chunk.choices[0].finish_reason is not None
        yield accumulated, is_done, elapsed, nbytes


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
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": query},
            ]},
        ],
        max_tokens=128,
    )
    latency = time.perf_counter() - t0
    return resp.choices[0].message.content, latency, nbytes


# ─── Small model (naive, full frame) ───────────────────────────────────────────

def infer_small_model(frame: np.ndarray, query: str) -> tuple[str, float, int]:
    """Run Qwen-7B on the full frame every tick — the 'just use a smaller model' baseline."""
    small = resize_frame(frame)
    b64, nbytes = encode_frame(small)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=SMALL_MODEL_ID,
        messages=[
            {"role": "system", "content": "You are a robot perception module. Describe what you see."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": query},
            ]},
        ],
        max_tokens=128,
    )
    latency = time.perf_counter() - t0
    return resp.choices[0].message.content, latency, nbytes


# ─── Cached mode with two-level change detection ───────────────────────────────

class CachedPipeline:
    """
    Two-level change detection:
      Level 1: Pixel diff (OpenCV) — fast, free
      Level 2: Semantic diff (Nebius embeddings) — catches pixel-noise false positives

    Only calls the VLM when BOTH levels agree there is a real change.
    """

    def __init__(self):
        self.prev_frame: np.ndarray | None = None
        self.last_response: str = ""
        self.last_embedding: list[float] | None = None
        self.prev_b64: str | None = None  # for temporal reasoning

    def infer(self, frame: np.ndarray, query: str) -> tuple[str, float, str, float, int, np.ndarray, np.ndarray]:
        """
        Returns:
          (text, latency, change_gate, semantic_similarity, bytes_sent, delta_frame, annotated_frame)

        change_gate: 'first_frame' | 'pixel+semantic' | 'pixel_only_cached' | 'no_change_cached'
        """
        small = resize_frame(frame)

        # ── First frame: must call model ──
        if self.prev_frame is None:
            self.prev_frame = small.copy()
            b64, nbytes = encode_frame(small)
            self.prev_b64 = b64
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": STATIC_SCENE_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": query},
                    ]},
                ],
                max_tokens=128,
            )
            latency = time.perf_counter() - t0
            self.last_response = resp.choices[0].message.content
            self.last_embedding = get_embedding(self.last_response)
            return self.last_response, latency, "first_frame", 1.0, nbytes, small, small

        # ── Level 1: Pixel diff ──
        delta, pixel_change, bbox = extract_delta(self.prev_frame, small)
        self.prev_frame = small.copy()

        if not pixel_change:
            return self.last_response + " [CACHED — no change]", 0.0, "no_change_cached", 1.0, 0, small, small

        # ── Level 2: Semantic diff via embeddings ──
        # Get a quick one-sentence description to embed (cheaper than full inference)
        sem_similarity = 0.0
        try:
            b64_small, _ = encode_frame(small)
            quick_resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_small}"}},
                        {"type": "text", "text": "Describe the objects on the table in one sentence."},
                    ]},
                ],
                max_tokens=40,
            )
            current_desc = quick_resp.choices[0].message.content.strip()
            current_emb = get_embedding(current_desc)

            if current_emb and self.last_embedding:
                sem_similarity = cosine_similarity(self.last_embedding, current_emb)
                if sem_similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                    # Pixel said change, but semantically the same — false positive
                    annotated = draw_bbox(small, bbox, f"PIXEL NOISE (sim={sem_similarity:.2f})") if bbox else small
                    return (self.last_response + f" [CACHED — semantic sim={sem_similarity:.2f}]",
                            0.0, "pixel_only_cached", sem_similarity, 0, delta, annotated)
        except Exception:
            pass  # If embedding fails, fall through to full inference

        # ── Both levels triggered: real change — run full inference with temporal context ──
        b64, nbytes = encode_frame(delta)

        # Temporal reasoning: include previous frame so model can reason about change over time
        temporal_query = (
            "Here are two frames from a robot's camera.\n"
            "Frame 1 (previous): what the scene looked like before.\n"
            "Frame 2 (current delta): what changed.\n"
            "1) What object appeared or moved?\n"
            "2) What specific action should the robot arm take?"
        )

        messages = [{"role": "system", "content": STATIC_SCENE_PROMPT}]
        if self.prev_b64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Frame 1 — previous scene:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.prev_b64}"}},
                    {"type": "text", "text": "Frame 2 — current change detected:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": temporal_query},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": DELTA_QUERY},
                ],
            })

        # Update prev_b64 for next temporal comparison
        self.prev_b64, _ = encode_frame(small)

        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=150,
        )
        latency = time.perf_counter() - t0
        self.last_response = resp.choices[0].message.content
        self.last_embedding = get_embedding(self.last_response)

        annotated = draw_bbox(small, bbox, f"CHANGE (sim={sem_similarity:.2f})") if bbox else small
        return self.last_response, latency, "pixel+semantic", sem_similarity, nbytes, delta, annotated
