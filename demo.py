"""
Gradio demo — side-by-side naive vs cached inference on a video file.
Run: python demo.py --video path/to/video.mp4
"""

import argparse
import cv2
import numpy as np
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
from pipeline import infer_naive, CachedPipeline

executor = ThreadPoolExecutor(max_workers=2)

QUERY = "What objects do you see? Have any objects moved or appeared?"

cached_pipeline = CachedPipeline()

naive_latencies = []
cached_latencies = []
naive_bytes_list = []
cached_bytes_list = []


def process_frame(frame_bgr: np.ndarray):
    # Run both modes in parallel for a fair latency comparison
    fut_naive = executor.submit(infer_naive, frame_bgr, QUERY)
    fut_cached = executor.submit(cached_pipeline.infer, frame_bgr, QUERY)
    naive_text, naive_lat, naive_bytes = fut_naive.result()
    cached_text, cached_lat, used_delta, cached_bytes, delta_frame = fut_cached.result()

    speedup = naive_lat / cached_lat if cached_lat > 0 else 1.0
    bytes_saved_pct = (1 - cached_bytes / naive_bytes) * 100 if naive_bytes > 0 else 0

    label_naive = f"NAIVE  |  {naive_lat*1000:.0f} ms  |  {naive_bytes/1024:.1f} KB sent"
    label_cached = (
        f"CACHED |  {cached_lat*1000:.0f} ms  ({'delta' if used_delta else 'full'})  |  "
        f"{cached_bytes/1024:.1f} KB sent  |  {speedup:.2f}x faster  |  {bytes_saved_pct:.0f}% less data"
    )

    delta_rgb = cv2.cvtColor(delta_frame, cv2.COLOR_BGR2RGB)
    return naive_text, label_naive, cached_text, label_cached, naive_lat, cached_lat, naive_bytes, cached_bytes, delta_rgb


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
                gr.Markdown("#### Full Frame (Naive sends this)")
                frame_img = gr.Image(label="Current Frame", type="numpy", height=300)
            with gr.Column():
                gr.Markdown("#### Delta Crop (Cached sends this)")
                delta_img = gr.Image(label="What Cached Mode Sends", type="numpy", height=300)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Naive — full frame every tick")
                naive_label = gr.Label(label="Latency + Payload")
                naive_out = gr.Textbox(label="Response", lines=5)
            with gr.Column():
                gr.Markdown("### Cached — static prefix + delta crop")
                cached_label = gr.Label(label="Latency + Payload")
                cached_out = gr.Textbox(label="Response", lines=5)

        with gr.Row():
            stats_box = gr.Textbox(label="Running Stats", interactive=False)

        with gr.Row():
            run_btn = gr.Button("Run Next Frame", variant="primary")
            run_all_btn = gr.Button("Run All Frames")

        frame_idx = gr.State(0)

        def next_frame(idx):
            if idx >= len(sampled):
                return None, None, "—", "Done", "—", "Done", "All frames complete", idx

            frame = sampled[idx]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            n_text, n_lbl, c_text, c_lbl, n_lat, c_lat, n_bytes, c_bytes, delta_rgb = process_frame(frame)

            naive_latencies.append(n_lat * 1000)
            cached_latencies.append(c_lat * 1000)
            naive_bytes_list.append(n_bytes)
            cached_bytes_list.append(c_bytes)

            avg_n = np.mean(naive_latencies)
            avg_c = np.mean(cached_latencies)
            avg_nb = np.mean(naive_bytes_list) / 1024
            avg_cb = np.mean(cached_bytes_list) / 1024
            total_saved_kb = (sum(naive_bytes_list) - sum(cached_bytes_list)) / 1024

            stats = (
                f"Frames: {len(naive_latencies)}/{len(sampled)}  |  "
                f"Latency — Naive: {avg_n:.0f}ms  Cached: {avg_c:.0f}ms  Speedup: {avg_n/avg_c:.2f}x\n"
                f"Payload — Naive avg: {avg_nb:.1f}KB  Cached avg: {avg_cb:.1f}KB  "
                f"Total saved: {total_saved_kb:.1f}KB  ({(1-avg_cb/avg_nb)*100:.0f}% less data)"
            )

            return frame_rgb, delta_rgb, n_text, n_lbl, c_text, c_lbl, stats, idx + 1

        run_btn.click(
            fn=next_frame,
            inputs=[frame_idx],
            outputs=[frame_img, delta_img, naive_out, naive_label, cached_out, cached_label, stats_box, frame_idx],
        )

        def run_all(idx):
            while idx < len(sampled):
                results = next_frame(idx)
                idx = results[-1]
                yield results

        run_all_btn.click(
            fn=run_all,
            inputs=[frame_idx],
            outputs=[frame_img, delta_img, naive_out, naive_label, cached_out, cached_label, stats_box, frame_idx],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to demo video file")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui(args.video)
    demo.launch(server_port=args.port, share=False)
