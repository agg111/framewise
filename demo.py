"""
Gradio demo — side-by-side naive vs cached inference on a video file.
Run: python demo.py --video path/to/video.mp4
"""

import argparse
import os
import re
import cv2
import numpy as np
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from pipeline import infer_naive, CachedPipeline, get_scene_inventory, infer_minimax, DELTA_QUERY

load_dotenv()

executor = ThreadPoolExecutor(max_workers=4)

QUERY = "What objects do you see? Have any objects moved or appeared?"

cached_pipeline = CachedPipeline()

naive_latencies = []
cached_latencies = []
minimax_latencies = []
naive_bytes_list = []
cached_bytes_list = []

# ─── Tavily ────────────────────────────────────────────────────────────────────

def tavily_lookup(response_text: str) -> str:
    """Extract the new object from response and search Tavily for context."""
    try:
        from tavily import TavilyClient
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "Tavily API key not set."

        # Extract first object name from response (simple heuristic)
        match = re.search(r'(?:pair of |a |an )?([\w\s\-]+?)(?:\s+(?:resting|placed|on|is|was|that|with)|\.|,)', response_text, re.IGNORECASE)
        query = match.group(1).strip() if match else "new object detected by robot"
        search_query = f"robot manipulation {query} safety handling"

        client = TavilyClient(api_key=api_key)
        result = client.search(search_query, max_results=2)
        snippets = [r.get("content", "")[:200] for r in result.get("results", [])]
        return f"Tavily [{query}]:\n" + "\n\n".join(snippets) if snippets else "No results."
    except Exception as e:
        return f"Tavily error: {e}"


# ─── Frame processing ──────────────────────────────────────────────────────────

def process_frame(frame_bgr: np.ndarray):
    # Fire naive + cached in parallel first
    fut_naive = executor.submit(infer_naive, frame_bgr, QUERY)
    fut_cached = executor.submit(cached_pipeline.infer, frame_bgr, QUERY)
    naive_text, naive_lat, naive_bytes = fut_naive.result()
    cached_text, cached_lat, change_gate, sem_sim, cached_bytes, delta_frame, annotated_frame = fut_cached.result()

    # Fire MiniMax on the delta crop in parallel with post-processing
    fut_minimax = executor.submit(infer_minimax, delta_frame, DELTA_QUERY if change_gate == "pixel+semantic" else QUERY)

    real_change = change_gate == "pixel+semantic"
    speedup = naive_lat / cached_lat if cached_lat > 0 else 1.0
    bytes_saved_pct = (1 - cached_bytes / naive_bytes) * 100 if naive_bytes > 0 else 0

    gate_label = {
        "first_frame":       "first frame",
        "no_change_cached":  "SKIP — no pixel change",
        "pixel_only_cached": f"SKIP — pixel noise (semantic sim={sem_sim:.2f})",
        "pixel+semantic":    f"CHANGE — both gates triggered (sim={sem_sim:.2f})",
    }.get(change_gate, change_gate)

    label_naive = f"NAIVE  |  {naive_lat*1000:.0f} ms  |  {naive_bytes/1024:.1f} KB sent"
    label_cached = (
        f"CACHED (Qwen) |  {cached_lat*1000:.0f} ms  |  {cached_bytes/1024:.1f} KB  |  "
        f"{'∞' if cached_lat == 0 else f'{speedup:.1f}x faster'}  |  {bytes_saved_pct:.0f}% less data"
    )

    # Get MiniMax result
    minimax_text, minimax_lat, _ = fut_minimax.result()
    label_minimax = f"Gemma-3-27B (Nebius) |  {minimax_lat*1000:.0f} ms  |  {cached_bytes/1024:.1f} KB sent"

    # Scene inventory + Tavily — only fire on real semantic change
    inventory_text = f"Gate: {gate_label}\nSemantic similarity: {sem_sim:.3f}"
    tavily_text = ""
    if real_change:
        inv = get_scene_inventory(frame_bgr)
        if inv:
            objects = ", ".join(inv.get("objects", []))
            new_obj = inv.get("new_object") or "—"
            action = inv.get("robot_action") or "—"
            inventory_text = f"Gate: {gate_label}\nObjects: {objects}\nNew: {new_obj}\nRobot action: {action}"
        tavily_text = tavily_lookup(cached_text)

    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    delta_rgb = cv2.cvtColor(delta_frame, cv2.COLOR_BGR2RGB)
    return (naive_text, label_naive, cached_text, label_cached,
            minimax_text, label_minimax,
            naive_lat, cached_lat, minimax_lat, naive_bytes, cached_bytes,
            delta_rgb, annotated_rgb, inventory_text, tavily_text)


# ─── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    SAMPLE_EVERY = 10
    sampled = frames[::SAMPLE_EVERY]
    print(f"Loaded {len(frames)} frames, sampling {len(sampled)} frames (every {SAMPLE_EVERY})")

    with gr.Blocks(title="Scene KV Cache — Robot Perception") as demo:
        gr.Markdown(
            "## Persistent Scene KV Cache for Robot Perception\n"
            "**Same VLM · Same GPU · Only delta sent when scene is static**\n\n"
            f"Video: {len(sampled)} sampled frames from {len(frames)/30:.1f}s clip"
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Full Frame — Naive sends this every tick")
                frame_img = gr.Image(label="Current Frame", type="numpy", height=280)
            with gr.Column():
                gr.Markdown("#### Annotated Frame — Cached detects change here")
                annotated_img = gr.Image(label="Change Detection", type="numpy", height=280)
            with gr.Column():
                gr.Markdown("#### Delta Crop — Cached sends only this")
                delta_img = gr.Image(label="What Cached Mode Sends", type="numpy", height=280)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Naive — Qwen, full frame")
                naive_label = gr.Label(label="Latency + Payload")
                naive_out = gr.Textbox(label="Response", lines=4)
            with gr.Column():
                gr.Markdown("### 2. Cached — Qwen, delta crop")
                cached_label = gr.Label(label="Latency + Payload")
                cached_out = gr.Textbox(label="Response", lines=4)
            with gr.Column():
                gr.Markdown("### 3. Gemma-3-27B — delta crop")
                minimax_label = gr.Label(label="Latency + Payload")
                minimax_out = gr.Textbox(label="Response", lines=4)

        with gr.Row():
            inventory_out = gr.Textbox(label="Scene Inventory + Robot Action (fires on change)", lines=3, interactive=False)
            tavily_out = gr.Textbox(label="Tavily — Object Context (fires on change, optional)", lines=3, interactive=False)

        with gr.Row():
            stats_box = gr.Textbox(label="Running Stats", interactive=False)

        with gr.Row():
            run_btn = gr.Button("Run Next Frame", variant="primary")
            run_all_btn = gr.Button("Run All Frames")

        frame_idx = gr.State(0)

        def next_frame(idx):
            if idx >= len(sampled):
                return None, None, None, "—", "Done", "—", "Done", "—", "Done", "", "", "All frames complete", idx

            frame = sampled[idx]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            (n_text, n_lbl, c_text, c_lbl,
             mm_text, mm_lbl,
             n_lat, c_lat, mm_lat, n_bytes, c_bytes,
             delta_rgb, annotated_rgb, inventory_text, tavily_text) = process_frame(frame)

            naive_latencies.append(n_lat * 1000)
            cached_latencies.append(c_lat * 1000)
            minimax_latencies.append(mm_lat * 1000)
            naive_bytes_list.append(n_bytes)
            cached_bytes_list.append(c_bytes)

            avg_n = np.mean(naive_latencies)
            avg_c = np.mean(cached_latencies)
            avg_mm = np.mean(minimax_latencies)
            avg_nb = np.mean(naive_bytes_list) / 1024
            avg_cb = np.mean(cached_bytes_list) / 1024
            total_saved_kb = (sum(naive_bytes_list) - sum(cached_bytes_list)) / 1024

            stats = (
                f"Frames: {len(naive_latencies)}/{len(sampled)}  |  "
                f"Naive: {avg_n:.0f}ms  Cached(Qwen): {avg_c:.0f}ms  MiniMax: {avg_mm:.0f}ms  "
                f"Speedup: {avg_n/avg_c:.2f}x\n"
                f"Payload — Naive: {avg_nb:.1f}KB  Cached: {avg_cb:.1f}KB  "
                f"Total saved: {total_saved_kb:.1f}KB  ({(1-avg_cb/avg_nb)*100:.0f}% less data)"
            )

            return (frame_rgb, annotated_rgb, delta_rgb,
                    n_text, n_lbl, c_text, c_lbl, mm_text, mm_lbl,
                    inventory_text, tavily_text, stats, idx + 1)

        outputs = [frame_img, annotated_img, delta_img,
                   naive_out, naive_label, cached_out, cached_label, minimax_out, minimax_label,
                   inventory_out, tavily_out, stats_box, frame_idx]

        run_btn.click(fn=next_frame, inputs=[frame_idx], outputs=outputs)

        def run_all(idx):
            while idx < len(sampled):
                results = next_frame(idx)
                idx = results[-1]
                yield results

        run_all_btn.click(fn=run_all, inputs=[frame_idx], outputs=outputs)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to demo video file")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui(args.video)
    demo.launch(server_port=args.port, share=False)
