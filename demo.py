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
from pipeline import (infer_naive, CachedPipeline, get_scene_inventory, infer_minimax,
                      DELTA_QUERY, rerank_actions, submit_batch, get_batch_results,
                      infer_streaming, get_robot_kb, CONFIDENCE_THRESHOLD)

load_dotenv()

executor = ThreadPoolExecutor(max_workers=4)

QUERY = "What objects do you see? Have any objects moved or appeared?"

cached_pipeline = CachedPipeline()

naive_latencies = []
cached_latencies = []
minimax_latencies = []
naive_bytes_list = []
cached_bytes_list = []

# Gate stats
gate_counts = {"first_frame": 0, "no_change_cached": 0, "pixel_only_cached": 0, "pixel+semantic": 0}

# Nebius pricing: ~$0.001 per 1000 tokens. Approx 1 token per 4 bytes of image.
COST_PER_TOKEN = 0.000001  # $0.000001 per token

def bytes_to_cost(total_bytes: int) -> float:
    tokens = total_bytes / 4
    return tokens * COST_PER_TOKEN

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

    skipped = cached_lat == 0
    label_naive = (
        f"Latency:  {naive_lat*1000:.0f} ms\n"
        f"Payload:  {naive_bytes/1024:.1f} KB"
    )
    label_cached = (
        f"Latency:  {'SKIPPED (0 ms)' if skipped else f'{cached_lat*1000:.0f} ms'}\n"
        f"Payload:  {cached_bytes/1024:.1f} KB  |  "
        f"Speedup: {'∞' if skipped else f'{speedup:.1f}x'}  |  "
        f"Data saved: {bytes_saved_pct:.0f}%"
    )

    # Get MiniMax result
    minimax_text, minimax_lat, _ = fut_minimax.result()
    label_minimax = (
        f"Latency:  {minimax_lat*1000:.0f} ms\n"
        f"Payload:  {cached_bytes/1024:.1f} KB  (delta crop)"
    )

    # Scene inventory + RAG + Reranking + Confidence gate + Tavily — only on real change
    inventory_text = f"Gate: {gate_label}\nSemantic similarity: {sem_sim:.3f}"
    tavily_text = ""
    if real_change:
        inv = get_scene_inventory(frame_bgr)
        if inv:
            objects = ", ".join(inv.get("objects", []))
            new_obj = inv.get("new_object") or "—"
            action = inv.get("robot_action") or "—"
            inventory_text = f"Gate: {gate_label}\nObjects: {objects}\nNew: {new_obj}\nRobot action: {action}"

        # Confidence gate + speculative routing decision
        conf_match = re.findall(r'(\d{1,3})%', minimax_text)
        confidences = [int(c) for c in conf_match if int(c) <= 100]
        avg_conf = sum(confidences) / len(confidences) if confidences else 100
        if avg_conf >= CONFIDENCE_THRESHOLD:
            spec_decision = f"DRAFT ACCEPTED — Gemma answer used (conf={avg_conf:.0f}% ≥ {CONFIDENCE_THRESHOLD}%)"
            spec_final = minimax_text
        else:
            spec_decision = f"ESCALATED → Qwen-72B (conf={avg_conf:.0f}% < {CONFIDENCE_THRESHOLD}%)"
            spec_final = cached_text
        inventory_text += f"\n\nSpeculative Routing: {spec_decision}\n→ {spec_final[:120]}"

        if avg_conf < CONFIDENCE_THRESHOLD:
            inventory_text += f"\n⚠️  LOW CONFIDENCE — HUMAN CONFIRMATION NEEDED"

        # RAG: retrieve relevant safety rules from knowledge base
        kb = get_robot_kb()
        rules = kb.query(cached_text, top_k=2)
        if rules:
            inventory_text += "\n\nRAG Safety Rules:\n" + "\n".join(f"• {r}" for r in rules)

        # Rerank: pick best robot action between Qwen and Gemma
        best_action, scores = rerank_actions([cached_text, minimax_text])
        winner = "Qwen" if best_action == cached_text else "Gemma"
        score_str = f"{scores[0]:.3f} vs {scores[1]:.3f}" if len(scores) >= 2 else "—"
        inventory_text += f"\n\nReranker: {winner} wins (scores: {score_str})\n→ {best_action[:120]}"

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
    print("Initializing Robot Knowledge Base (embedding safety rules)...")
    get_robot_kb()
    print("Knowledge base ready.")

    with gr.Blocks(title="Scene KV Cache — Robot Perception") as demo:
        gr.Markdown(
            "## Persistent Scene KV Cache for Robot Perception\n"
            "**Same VLM · Same GPU · Only delta sent when scene is static**\n\n"
            f"Video: {len(sampled)} sampled frames from {len(frames)/30:.1f}s clip"
        )
        with gr.Tabs():
            # ── Tab 1: Live Streaming ──────────────────────────────────────────
            with gr.Tab("Live Streaming"):
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
                        naive_label = gr.Textbox(label="Latency + Payload", lines=2, max_lines=2, interactive=False)
                        naive_out = gr.Textbox(label="Response", lines=8, max_lines=12)
                    with gr.Column():
                        gr.Markdown("### 2. Cached — Qwen, delta crop")
                        cached_label = gr.Textbox(label="Latency + Payload", lines=2, max_lines=2, interactive=False)
                        cached_out = gr.Textbox(label="Response", lines=8, max_lines=12)
                    with gr.Column():
                        gr.Markdown("### 3. Gemma-3-27B — delta crop")
                        minimax_label = gr.Textbox(label="Latency + Payload", lines=2, max_lines=2, interactive=False)
                        minimax_out = gr.Textbox(label="Response", lines=8, max_lines=12)

                with gr.Row():
                    inventory_out = gr.Textbox(label="Scene Inventory + Robot Action + Reranker Winner", lines=6, max_lines=10, interactive=False)
                    tavily_out = gr.Textbox(label="Tavily — Object Context (fires on change)", lines=6, max_lines=10, interactive=False)

                with gr.Row():
                    cost_box = gr.Textbox(label="Cost Analysis", lines=3, max_lines=4, interactive=False)
                    gate_box = gr.Textbox(label="Gate Stats — Change Detection Breakdown", lines=3, max_lines=4, interactive=False)

                with gr.Row():
                    stream_out = gr.Textbox(label="Streaming Output (token-by-token from Nebius)", lines=4, max_lines=8, interactive=False)

                with gr.Row():
                    stats_box = gr.Textbox(label="Running Stats", lines=3, max_lines=4, interactive=False)

                with gr.Row():
                    run_btn = gr.Button("Run Next Frame", variant="primary")
                    run_all_btn = gr.Button("Run All Frames")
                    stream_btn = gr.Button("Stream Current Frame (Nebius)", variant="secondary")

                frame_idx = gr.State(0)

                def _compute_frame(idx):
                    """Run inference on frame[idx], return (results_tuple, new_idx) or (None, idx) if done."""
                    if idx >= len(sampled):
                        return None, idx

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

                    naive_cost = bytes_to_cost(sum(naive_bytes_list))
                    cached_cost = bytes_to_cost(sum(cached_bytes_list))
                    cost_saved = naive_cost - cached_cost
                    frames_done = len(naive_latencies)
                    scale = (10 * 3600) / max(frames_done, 1)

                    cost_text = (
                        f"So far ({frames_done} frames) — Naive: ${naive_cost:.4f}  Cached: ${cached_cost:.4f}  Saved: ${cost_saved:.4f}\n"
                        f"Projected (10 FPS, 1hr) — Naive: ${naive_cost*scale:.2f}  Cached: ${cached_cost*scale:.2f}  Saved: ${cost_saved*scale:.2f}"
                    )

                    skipped = sum(1 for l in cached_latencies if l == 0)
                    gate_text = (
                        f"Processed: {frames_done}  |  Skipped: {skipped}  |  VLM called: {frames_done-skipped}\n"
                        f"Skip rate: {skipped/frames_done*100:.0f}%  —  {skipped/frames_done*100:.0f}% fewer API calls"
                    )

                    stats = (
                        f"Frames: {frames_done}/{len(sampled)}\n"
                        f"Avg Latency  —  Naive (Qwen): {avg_n:.0f}ms  |  Cached (Qwen): {avg_c:.0f}ms  |  Gemma-27B: {avg_mm:.0f}ms  |  Speedup: {avg_n/avg_c:.2f}x\n"
                        f"Avg Payload  —  Naive: {avg_nb:.1f}KB  |  Cached: {avg_cb:.1f}KB  |  Total saved: {total_saved_kb:.1f}KB  ({(1-avg_cb/avg_nb)*100:.0f}% less data)"
                    )

                    result = (frame_rgb, annotated_rgb, delta_rgb,
                              n_text, n_lbl, c_text, c_lbl, mm_text, mm_lbl,
                              inventory_text, tavily_text, cost_text, gate_text, stats, idx + 1)
                    return result, idx + 1

                # outputs has 16 entries: 15 existing + stream_out
                outputs = [frame_img, annotated_img, delta_img,
                           naive_out, naive_label, cached_out, cached_label, minimax_out, minimax_label,
                           inventory_out, tavily_out, cost_box, gate_box, stats_box, frame_idx,
                           stream_out]

                # 16 outputs total: 15 main + stream_out
                # For streaming-only updates, skip all 15 main outputs
                _SKIP15 = tuple(gr.update() for _ in range(15))

                def next_frame(idx):
                    result, new_idx = _compute_frame(idx)
                    if result is None:
                        yield (None, None, None,
                               "—", "Done", "—", "Done", "—", "Done",
                               "", "", "", "", "All frames complete", idx, "")
                        return

                    # Yield main results immediately, stream box cleared
                    yield (*result, "")

                    # Auto-stream the same frame token-by-token
                    frame = sampled[idx]
                    accumulated = ""
                    for text, is_done, elapsed_ms, _ in infer_streaming(frame, QUERY):
                        accumulated = f"[{elapsed_ms:.0f}ms] {text}"
                        yield (*_SKIP15, accumulated)

                run_btn.click(fn=next_frame, inputs=[frame_idx], outputs=outputs)

                def run_all(idx):
                    # Run all frames fast (no per-frame streaming); stream the last frame at the end
                    last_frame_bgr = None
                    last_idx = idx
                    while idx < len(sampled):
                        result, idx = _compute_frame(idx)
                        if result is None:
                            break
                        last_frame_bgr = sampled[idx - 1]
                        last_idx = idx
                        yield (*result, "")
                    # Stream the final frame
                    if last_frame_bgr is not None:
                        accumulated = ""
                        for text, is_done, elapsed_ms, _ in infer_streaming(last_frame_bgr, QUERY):
                            accumulated = f"[{elapsed_ms:.0f}ms] {text}"
                            yield (*_SKIP15, accumulated)

                run_all_btn.click(fn=run_all, inputs=[frame_idx], outputs=outputs)

                def stream_frame(idx):
                    frame_i = max(0, idx - 1)
                    frame = sampled[min(frame_i, len(sampled) - 1)]
                    for text, is_done, elapsed_ms, _ in infer_streaming(frame, QUERY):
                        yield f"[{elapsed_ms:.0f}ms] {text}"

                stream_btn.click(fn=stream_frame, inputs=[frame_idx], outputs=[stream_out])

            # ── Tab 2: Batch Analysis (cloud-native) ──────────────────────────
            with gr.Tab("Batch Analysis (Nebius Cloud)"):
                gr.Markdown(
                    "### Submit all frames as a Nebius batch job\n"
                    "Fire-and-forget cloud inference — no blocking, results retrieved async.\n"
                    "This is how you'd run robot perception at scale in production."
                )
                with gr.Row():
                    batch_submit_btn = gr.Button("Submit Batch Job to Nebius", variant="primary")
                    batch_check_btn = gr.Button("Check Batch Status")

                batch_id_box = gr.Textbox(label="Batch Job ID", interactive=False)
                batch_status_box = gr.Textbox(label="Status", interactive=False)
                batch_results_box = gr.Textbox(label="Results (per frame)", lines=15, interactive=False)

                def submit_batch_job():
                    try:
                        batch_id = submit_batch(sampled, QUERY)
                        return batch_id, f"Submitted {len(sampled)} frames. Status: in_progress", ""
                    except Exception as e:
                        return "", f"Error: {e}", ""

                def check_batch_status(batch_id):
                    if not batch_id:
                        return "No batch ID", ""
                    try:
                        status, results = get_batch_results(batch_id)
                        if status != "completed":
                            return f"Status: {status} — check again in a moment", ""
                        output = "\n\n".join([f"[{r['frame']}]\n{r['text']}" for r in results])
                        return f"Status: completed ({len(results)} frames)", output
                    except Exception as e:
                        return f"Error: {e}", ""

                batch_submit_btn.click(
                    fn=submit_batch_job,
                    outputs=[batch_id_box, batch_status_box, batch_results_box]
                )
                batch_check_btn.click(
                    fn=check_batch_status,
                    inputs=[batch_id_box],
                    outputs=[batch_status_box, batch_results_box]
                )


    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to demo video file")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui(args.video)
    demo.launch(server_port=args.port, share=True)
