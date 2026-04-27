# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import argparse
import imageio.v3 as iio
import numpy as np
from matplotlib import cm
from torchinfo import summary

from cotracker.predictor import CoTrackerOnlinePredictor


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument("--trail_length", type=int, default=10, help="Number of past frames shown as trail")
    parser.add_argument("--linewidth", type=int, default=3, help="Track point radius and trail line width")
    parser.add_argument("--no_show", action="store_true", help="Disable live OpenCV preview window")
    parser.add_argument("--save_video", action="store_true", help="Save rendered video to ./saved_videos/")

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    summary(model)
    window_frames = []
    # Mutable state shared with render_frames (nonlocal is illegal inside if __name__)
    state = {
        "track_buffer": [],     # list of (N, 2) float arrays, one per rendered frame
        "colors_bgr": None,     # (N, 3) uint8 BGR, assigned after first prediction
        "display_ptr": 0,
        "video_writer": None,
    }

    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2:]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )

    def render_frames(pred_tracks, pred_visibility):
        """Draw tracks on newly predicted frames and display/save them.

        The online model accumulates predictions, so pred_tracks covers all
        frames from t=0 up to the current call. We only render the frames
        starting from display_ptr (the ones not yet shown).
        """
        if pred_tracks is None:
            return

        tracks_np = pred_tracks[0].cpu().numpy()      # (T_total, N, 2)
        vis_np = pred_visibility[0].cpu().numpy()     # (T_total, N) or (T_total, N, 1)
        if vis_np.ndim == 3:
            vis_np = vis_np[..., 0]
        T_total, N = tracks_np.shape[:2]

        display_ptr = state["display_ptr"]
        if display_ptr >= T_total:
            return

        # Assign rainbow colors once, keyed by y-position at query frame
        if state["colors_bgr"] is None:
            cmap = cm.get_cmap("gist_rainbow")
            qf = min(args.grid_query_frame, T_total - 1)
            y_vals = tracks_np[qf, :, 1]
            y_min, y_max = float(y_vals.min()), float(y_vals.max())
            norm = (y_vals - y_min) / (y_max - y_min + 1e-6)
            rgb = (np.array([cmap(float(v))[:3] for v in norm]) * 255).astype(np.uint8)
            state["colors_bgr"] = rgb[:, ::-1].copy()  # RGB -> BGR

        colors_bgr = state["colors_bgr"]
        track_buffer = state["track_buffer"]

        # Initialise VideoWriter before the first frame
        if args.save_video and state["video_writer"] is None:
            h, w = window_frames[display_ptr].shape[:2]
            seq_name = os.path.splitext(os.path.basename(args.video_path))[0]
            os.makedirs("./saved_videos", exist_ok=True)
            save_path = f"./saved_videos/{seq_name}_opencv.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            state["video_writer"] = cv2.VideoWriter(save_path, fourcc, 10.0, (w, h))
            print(f"Saving rendered video to {save_path}")

        new_count = T_total - display_ptr
        for idx in range(new_count):
            t = display_ptr + idx
            frame_rgb = window_frames[t]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            curr_tracks = tracks_np[t]   # (N, 2)
            curr_vis = vis_np[t]          # (N,)

            # Draw fading trail lines
            trail = track_buffer[-args.trail_length:] if args.trail_length > 0 else []
            trail_len = len(trail)
            for n in range(N):
                color = tuple(int(v) for v in colors_bgr[n])
                pts = [
                    (int(past[n, 0]), int(past[n, 1]))
                    for past in trail
                    if past[n, 0] > 0 and past[n, 1] > 0
                ]
                for s in range(len(pts) - 1):
                    fade = (s + 1) / max(trail_len, 1)
                    faded = tuple(int(c * fade * 0.85) for c in color)
                    cv2.line(frame_bgr, pts[s], pts[s + 1], faded, max(1, args.linewidth - 1))

            # Draw current point (filled circle if visible, hollow if not)
            for n in range(N):
                px, py = int(curr_tracks[n, 0]), int(curr_tracks[n, 1])
                if px > 0 and py > 0:
                    color = tuple(int(v) for v in colors_bgr[n])
                    thickness = -1 if curr_vis[n] else 1
                    cv2.circle(frame_bgr, (px, py), args.linewidth * 2, color, thickness)

            track_buffer.append(curr_tracks.copy())

            if not args.no_show:
                cv2.imshow("CoTracker Online", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.save_video and state["video_writer"] is not None:
                state["video_writer"].write(frame_bgr)

        state["display_ptr"] += new_count

    # Iterate over video frames, processing one window at a time
    is_first_step = True
    for i, frame in enumerate(iio.imiter(args.video_path, plugin="FFMPEG")):
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
            )
            is_first_step = False
            render_frames(pred_tracks, pred_visibility)
        window_frames.append(frame)

    # Process the final partial window
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1:],
        is_first_step,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
    )
    render_frames(pred_tracks, pred_visibility)

    print("Tracking complete.")

    if not args.no_show:
        print("Press any key to close the preview window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if state["video_writer"] is not None:
        state["video_writer"].release()
        print("Video saved.")
