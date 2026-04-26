#!/usr/bin/env python3
"""
eval_frames.py — Generate overlapping-frame composite images from motion clips (MP4/GIF).

**strobe** (default layout **``concat``**): Plain horizontal strip — each sampled frame is
copied into a wide canvas; frame *i+1* starts ``(1−overlap)×width`` pixels to the right, so
the new image **overwrites** the overlap band (including the empty right-side background of
the previous panel). No alpha, blur, or masks.

Other strobe layouts: ``spread_h`` / ``spread_v`` (mask+blend), ``adjoin`` (soft overlap),
``in_place``. **``--neat``** helps mask-based layouts only.

**blend**: Linear alpha stack (ghost trail), initialised from the reference frame instead of
a flat colour.

**max**: Per-pixel maximum versus the reference—quick heuristic for dark scenes; may stack
bright HUD pixels.

Follows the same config-driven structure as eval_analyzer.py: when called with a run
directory or a parent folder, the script finds every ``.../env_rgb_array/`` (recursive
walk), picks the longest-duration video in each, and writes ``overlap_frames.png`` next to
that ``env_rgb_array`` folder (the run directory).

Usage examples
--------------
# GIF: simple hard-paste strip (default about 40% width overlap between frames)
python eval_frames.py path/to/clip.gif

# Abutted panels (no overlap), like a contact sheet
python eval_frames.py clip.gif --overlap 0

# Mask-based paper strip (advanced)
python eval_frames.py clip.gif --layout spread_h --neat --no-labels

# Soft adjoining panels (alpha matte)
python eval_frames.py clip.gif --layout adjoin --overlap 0.35 --neat

# Strobe with all poses stacked in the camera centre (no spatial shift)
python eval_frames.py clip.gif --layout in_place

# Vertical spread instead of horizontal
python eval_frames.py clip.gif --layout spread_v --spread-pixels 120

# Ghost trail (legacy blend) on MP4
python eval_frames.py clip.mp4 --mode blend --alpha-start 0.25 --alpha-end 0.9

# Strobe with tuning
python eval_frames.py clip.gif --n-frames 10 --diff-thresh 18 --blur-sigma 1.0 --morph-kernel 3

# Reference = first frame of clip (default). Use middle of clip as static background:
python eval_frames.py run_dir/ --ref-frac 0.5

# Batch: several run dirs, or one root whose subtree contains many .../env_rgb_array/
python eval_frames.py case--h8/limp_walk case-l-h8/limp_walk
python eval_frames.py checkpoints/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_EXTENSIONS = {".gif", ".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_N_FRAMES = 10
DEFAULT_ALPHA_START = 0.20
DEFAULT_ALPHA_END = 1.00
DEFAULT_MODE = "strobe"
DEFAULT_DIFF_THRESH = 18
DEFAULT_BLUR_SIGMA = 0.0
DEFAULT_MORPH_KERNEL = 3
DEFAULT_REF_FRAC = 0.0
DEFAULT_LAYOUT = "concat"
DEFAULT_SPREAD_PX = 140
DEFAULT_SPREAD_MARGIN = 48
DEFAULT_FEATHER_SIGMA = 10.0
DEFAULT_EDGE_TONE = 0.38
DEFAULT_MASK_CROP_TOP = 76
DEFAULT_MASK_CROP_BOTTOM = 52
DEFAULT_ANCHOR = "body"
DEFAULT_MATTE = "smooth"
DEFAULT_OVERLAP_FRAC = 0.40
OUTPUT_FILENAME = "overlap_frames.png"
BACKGROUND_COLOR = (255, 255, 255)  # RGB white (blend mode when no video ref)


# ─────────────────────────────────────────────────────────────────────────────
# Video / GIF resolution
# ─────────────────────────────────────────────────────────────────────────────

def video_duration_seconds(path: Path) -> float:
    """Best-effort duration for picking the longest clip in a folder."""
    cap = cv2.VideoCapture(str(path))
    try:
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps and fps > 1e-6 and n > 0:
                return float(n / fps)
    finally:
        cap.release()
    if path.suffix.lower() == ".gif":
        try:
            _frames, n, fps = load_gif_all(path)
            if fps > 0 and n > 0:
                return float(n / fps)
        except Exception:
            pass
    return 0.0


def find_longest_video_in_env_rgb_folder(env_dir: Path) -> Path | None:
    """In env_rgb_array/, choose the video with the largest estimated duration."""
    if not env_dir.is_dir():
        return None
    candidates = [
        q for q in env_dir.iterdir()
        if q.is_file() and q.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda q: (video_duration_seconds(q), q.name.lower()))


def collect_jobs_under_dir(root: Path) -> list[tuple[Path, Path]]:
    """(video_path, run_dir) for each env_rgb_array under root with at least one video."""
    jobs: list[tuple[Path, Path]] = []
    for env in sorted(root.rglob("env_rgb_array"), key=lambda x: str(x).lower()):
        if not env.is_dir():
            continue
        vid = find_longest_video_in_env_rgb_folder(env)
        if vid is not None:
            jobs.append((vid, env.parent))
    return jobs


def resolve_input_jobs(arg: str) -> list[tuple[Path, Path]]:
    """
    Each job is (video_path, output_dir). output_dir is the run folder (parent of
    env_rgb_array/). A directory argument is scanned recursively for env_rgb_array/.
    """
    p = Path(arg)
    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
        return [(p, p.parent)]
    if p.is_dir():
        jobs = collect_jobs_under_dir(p)
        if not jobs:
            raise FileNotFoundError(
                f"No video file found under '{p}'. "
                f"Expected {VIDEO_EXTENSIONS} inside any .../env_rgb_array/ "
                f"(searched recursively)."
            )
        return jobs
    raise FileNotFoundError(f"'{p}' is neither a video file nor a directory.")


def _rgb_rgba_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def load_gif_all(path: Path) -> tuple[list[np.ndarray], int, float]:
    """
    Load all GIF frames as BGR uint8. Returns (frames, total_count, fps estimate).
    """
    raw = iio.imread(path, index=None)
    if raw.ndim == 3:
        raw = raw[np.newaxis, ...]
    n = int(raw.shape[0])
    frames = [_rgb_rgba_to_bgr(raw[i]) for i in range(n)]

    fps = 10.0
    try:
        meta = iio.immeta(path, plugin="pillow")
        dur = meta.get("duration")
        if dur is not None and float(dur) > 0:
            # Pillow GIF: duration is ms per frame in modern Pillow
            fps = 1000.0 / float(dur)
    except Exception:
        pass

    return frames, n, fps


def sample_frame_indices(total: int, n_frames: int,
                         start_frac: float, end_frac: float) -> list[int]:
    if total <= 0:
        return []
    first = int(start_frac * (total - 1))
    last = int(end_frac * (total - 1))
    if n_frames == 1:
        return [first]
    return [int(first + i * (last - first) / (n_frames - 1))
            for i in range(n_frames)]


def extract_frames_opencv(path: Path, n_frames: int,
                          start_frac: float, end_frac: float) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total <= 0:
        raise ValueError(f"Video '{path}' reports {total} frames.")

    indices = sample_frame_indices(total, n_frames, start_frac, end_frac)
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(path))
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  [warn] could not read frame {idx}, skipping", file=sys.stderr)
            continue
        frames.append(frame)
    cap.release()
    return frames


def extract_frames_from_list(all_frames: list[np.ndarray], n_frames: int,
                             start_frac: float, end_frac: float) -> list[np.ndarray]:
    total = len(all_frames)
    indices = sample_frame_indices(total, n_frames, start_frac, end_frac)
    return [all_frames[i].copy() for i in indices]


def extract_frames(path: Path, n_frames: int,
                   start_frac: float = 0.0,
                   end_frac: float = 1.0,
                   gif_cache: list[np.ndarray] | None = None,
                   ) -> list[np.ndarray]:
    """
    Extract n_frames evenly spaced frames from [start_frac, end_frac].
    GIF: pass gif_cache from load_gif_all to avoid reloading; otherwise loads internally.
    Returns list of BGR uint8 arrays.
    """
    if gif_cache is not None:
        return extract_frames_from_list(gif_cache, n_frames, start_frac, end_frac)
    if path.suffix.lower() == ".gif":
        all_bgr, _, _ = load_gif_all(path)
        return extract_frames_from_list(all_bgr, n_frames, start_frac, end_frac)
    return extract_frames_opencv(path, n_frames, start_frac, end_frac)


def read_frame_opencv(path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {idx} from {path}")
    return frame


def video_metadata(path: Path,
                   gif_frames: list[np.ndarray] | None = None,
                   gif_fps: float | None = None,
                   ) -> tuple[int, float]:
    """Frame count and FPS for label timing."""
    if gif_frames is not None:
        return len(gif_frames), (gif_fps or 10.0)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total, fps


# ─────────────────────────────────────────────────────────────────────────────
# Composite rendering
# ─────────────────────────────────────────────────────────────────────────────

def motion_mask(
    frame: np.ndarray,
    _ref: np.ndarray,
    ref_gray: np.ndarray,
    diff_thresh: int,
    blur_sigma: float,
    morph_kernel: int,
    kblur: int,
) -> np.ndarray:
    """Binary uint8 mask (255 = foreground) where frame differs from ref."""
    fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(fg, ref_gray)
    if blur_sigma > 0:
        diff = cv2.GaussianBlur(diff, (kblur, kblur), blur_sigma)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    if morph_kernel > 0:
        mk = morph_kernel | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def apply_mask_roi_gate(mask: np.ndarray, top_px: int, bottom_px: int) -> np.ndarray:
    """Zero out fixed HUD/timeline bands so overlays are not pasted into the strip."""
    if top_px <= 0 and bottom_px <= 0:
        return mask
    m = mask.copy()
    h, _w = m.shape[:2]
    t = min(max(0, top_px), h)
    b = min(max(0, bottom_px), h)
    if t > 0:
        m[:t, :] = 0
    if b > 0:
        m[h - b :, :] = 0
    return m


def mask_largest_blob(mask: np.ndarray) -> np.ndarray:
    """Keep largest connected foreground region (stabilises centroid vs HUD specks)."""
    if not np.any(mask):
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 80.0:
        return mask
    out = np.zeros_like(mask)
    cv2.drawContours(out, [best], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(mask, out)


def finalize_strobe_mask(mask: np.ndarray, anchor: str, crop_top: int, crop_bot: int) -> np.ndarray:
    """Apply ROI gate then optional largest-blob extraction for body anchoring."""
    m = apply_mask_roi_gate(mask, crop_top, crop_bot)
    if anchor == "body":
        m = mask_largest_blob(m)
    return m


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Image-space centroid of binary mask; falls back to geometric centre."""
    h, w = mask.shape[:2]
    m = cv2.moments(mask)
    if m["m00"] > 1e-3:
        return m["m10"] / m["m00"], m["m01"] / m["m00"]
    return w / 2.0, h / 2.0


def tile_ref_to_canvas(ref: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    """Repeat ref to cover canvas size (preserves tiled floor look)."""
    rh, rw = ref.shape[:2]
    reps_y = max(1, int(np.ceil(canvas_h / rh)))
    reps_x = max(1, int(np.ceil(canvas_w / rw)))
    tiled = np.tile(ref, (reps_y, reps_x, 1))
    return tiled[:canvas_h, :canvas_w, :].copy()


def stencil_to_alpha(
    stencil: np.ndarray,
    feather_sigma: float,
    matte_mode: str,
) -> np.ndarray:
    """
    Float alpha (H,W,1) from uint8 stencil. gaussian=frozen blob+blur;
    smooth=distance-transform interior + feather (cleaner fringe).
    """
    feath = max(0.0, float(feather_sigma))
    sm = stencil.astype(np.float32)

    if matte_mode == "smooth":
        bin01 = (sm > 127).astype(np.uint8)
        if np.any(bin01):
            dist = cv2.distanceTransform(bin01, cv2.DIST_L2, 5)
            dmax = float(dist.max())
            if dmax > 1e-3:
                a = (dist / dmax).astype(np.float32)
                if feath > 1e-3:
                    a = cv2.GaussianBlur(a, (0, 0), feath * 0.42)
                return np.clip(a[..., np.newaxis], 0.0, 1.0)

    if feath <= 1e-3:
        return (sm / 255.0)[..., np.newaxis]
    gb = cv2.GaussianBlur(sm, (0, 0), feath)[..., np.newaxis] / 255.0
    return gb


def try_parse_step_reward_hud(frame: np.ndarray) -> tuple[str | None, str | None]:
    """
    Try to read 'Step N' / 'reward: …' from the MuJoCo overlay (top strip).
    Requires optional ``pytesseract`` + system Tesseract; otherwise returns (None, None).
    """
    try:
        import pytesseract  # type: ignore import-not-found
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return None, None

    h, w = frame.shape[:2]
    crop = frame[4 : min(80, h - 4), 4 : min(w - 4, 680)]
    if crop.size == 0:
        return None, None

    crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    pil = Image.fromarray(gray)
    try:
        txt = pytesseract.image_to_string(pil, config="--psm 6").strip()
    except Exception:
        return None, None

    txt_clean = " ".join(txt.replace("\n", " ").split())
    step_val: str | None = None
    rew_val: str | None = None
    m_step = re.search(r"Step\s*[:\s]*(\d+)", txt_clean, re.I)
    if m_step:
        step_val = m_step.group(1)
    m_rew = re.search(r"reward\s*[:\s]*([^\s,]+)", txt_clean, re.I)
    if m_rew:
        rew_val = m_rew.group(1).strip()
    return step_val, rew_val


def draw_pose_captions(
    img: np.ndarray,
    canvas_positions: list[tuple[float, float]],
    captions: list[tuple[str, str]],
    spread_px: int,
    ref_h: int,
    position: str = "below",
    stride_px: float | None = None,
) -> None:
    """
    Draw compact step/reward captions near each pose centre; box width capped by spread.
    Mutates img in place (BGR uint8).
    """
    if not captions or not canvas_positions:
        return

    h, w = img.shape[:2]
    span = float(stride_px if stride_px is not None else spread_px)
    fs = float(np.clip(span / 380.0, 0.18, 0.36))
    thickness = max(1, int(round(fs * 2)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_tw = max(96, int(span * 0.82))
    gap_y = int(ref_h * 0.055)

    for (cx, cy), (step_lbl, rew_lbl) in zip(canvas_positions, captions):
        line2 = rew_lbl if rew_lbl.lower().startswith("reward") else f"reward {rew_lbl}"
        lines = [f"step {step_lbl}", line2]
        sizes = [cv2.getTextSize(t, font, fs, thickness)[0] for t in lines]
        tw = min(max_tw, max(s[0] for s in sizes) + 12)
        th_block = sizes[0][1] + sizes[1][1] + 18

        x1 = int(np.clip(cx - tw / 2, 4, w - tw - 4))
        x2 = x1 + tw

        if position == "above":
            y1 = int(cy - gap_y - th_block)
            y2 = int(cy - gap_y)
        else:
            y1 = int(cy + gap_y)
            y2 = int(cy + gap_y + th_block)

        if y2 >= h - 3:
            y2 = h - 3
            y1 = max(4, y2 - th_block)
        if y1 <= 3:
            y1 = 4
            y2 = min(h - 4, y1 + th_block)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        overlay = np.full_like(roi, (22, 22, 28))
        cv2.addWeighted(overlay, 0.84, roi, 0.16, 0, roi)

        y_cursor = y1 + 10 + sizes[0][1]
        for text in lines:
            ts = cv2.getTextSize(text, font, fs, thickness)[0]
            tx = int(np.clip(x1 + (tw - ts[0]) / 2, 4, w - ts[0] - 4))
            cv2.putText(
                img,
                text[:56],
                (tx, min(h - 5, y_cursor)),
                font,
                fs,
                (236, 236, 240),
                thickness,
                cv2.LINE_AA,
            )
            y_cursor += ts[1] + 5


def build_strobe_composite(
    frames: list[np.ndarray],
    ref: np.ndarray,
    diff_thresh: int,
    blur_sigma: float,
    morph_kernel: int,
    layout: str = "in_place",
    spread_px: int = DEFAULT_SPREAD_PX,
    spread_margin: int = DEFAULT_SPREAD_MARGIN,
    feather_sigma: float = DEFAULT_FEATHER_SIGMA,
    edge_tone: float = DEFAULT_EDGE_TONE,
    pose_captions: list[tuple[str, str]] | None = None,
    caption_position: str = "below",
    mask_crop_top: int = 0,
    mask_crop_bottom: int = 0,
    anchor: str = "full",
    matte_mode: str = "gaussian",
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
) -> np.ndarray:
    """
    Static reference ref; paste motion regions from each sampled frame.

    layout=in_place: same as original (stack on ref, fixed camera centre).
    layout=spread_h | spread_v | adjoin: tiled ref + soft-mask layers.
    Temporal order: older -> newer (newer wins on overlap).
    """
    if not frames:
        raise ValueError("No frames to composite.")

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    kblur = max(3, int(blur_sigma * 4) | 1) if blur_sigma > 0 else 0

    masks = [
        finalize_strobe_mask(
            motion_mask(f, ref, ref_gray, diff_thresh, blur_sigma, morph_kernel, kblur),
            anchor,
            mask_crop_top,
            mask_crop_bottom,
        )
        for f in frames
    ]

    if layout == "in_place":
        canvas = ref.copy()
        for f, mask in zip(frames, masks):
            m3 = cv2.merge([mask, mask, mask]) > 0
            canvas = np.where(m3, f, canvas)
        return canvas

    caption_stride: float | None = None
    if layout == "adjoin":
        rw = float(ref.shape[1])
        caption_stride = max(1.0, rw * (1.0 - float(np.clip(overlap_frac, 0.02, 0.9))))

    return _strobe_spread_strip(
        frames,
        masks,
        ref,
        layout,
        spread_px,
        spread_margin,
        feather_sigma,
        edge_tone,
        pose_captions,
        caption_position,
        matte_mode,
        overlap_frac,
        caption_stride,
    )


def _warp_layer_to_canvas(
    frame: np.ndarray,
    mask: np.ndarray,
    dx: float,
    dy: float,
    ox: float,
    oy: float,
    canvas_h: int,
    canvas_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense foreground float layer [0,1] and uint8 stencil mask on canvas coordinates."""
    layer = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    stencil = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return layer, stencil

    dst_x = np.round(xs + dx - ox).astype(np.int32)
    dst_y = np.round(ys + dy - oy).astype(np.int32)
    valid = (dst_x >= 0) & (dst_x < canvas_w) & (dst_y >= 0) & (dst_y < canvas_h)
    if not np.any(valid):
        return layer, stencil

    dyv = dst_y[valid]
    dxv = dst_x[valid]
    syv = ys[valid]
    sxv = xs[valid]
    layer[dyv, dxv] = frame[syv, sxv].astype(np.float32) / 255.0
    stencil[dyv, dxv] = 255
    return layer, stencil


def _strobe_spread_strip(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    ref: np.ndarray,
    layout: str,
    spread_px: int,
    margin: int,
    feather_sigma: float,
    edge_tone: float,
    pose_captions: list[tuple[str, str]] | None,
    caption_position: str,
    matte_mode: str,
    overlap_frac: float,
    caption_stride: float | None,
) -> np.ndarray:
    """Spread or adjoin poses; soft-mask blend with optional smooth matte."""
    n = len(frames)
    centroids = [mask_centroid(m) for m in masks]

    rw, rh = ref.shape[1], ref.shape[0]

    if layout == "spread_h":
        if spread_px <= 0:
            raise ValueError("--spread-pixels must be positive for spread_h.")
        cy_tgt = float(np.median([c[1] for c in centroids]))
        deltas = [
            (margin + i * spread_px - centroids[i][0], cy_tgt - centroids[i][1])
            for i in range(n)
        ]
    elif layout == "spread_v":
        if spread_px <= 0:
            raise ValueError("--spread-pixels must be positive for spread_v.")
        cx_tgt = float(np.median([c[0] for c in centroids]))
        deltas = [
            (cx_tgt - centroids[i][0], margin + i * spread_px - centroids[i][1])
            for i in range(n)
        ]
    elif layout == "adjoin":
        ovl = float(np.clip(overlap_frac, 0.02, 0.92))
        stride = max(1.0, float(rw) * (1.0 - ovl))
        cy_tgt = float(np.median([c[1] for c in centroids]))
        deltas = [
            (
                margin + i * stride + float(rw) / 2.0 - centroids[i][0],
                cy_tgt - centroids[i][1],
            )
            for i in range(n)
        ]
    else:
        raise ValueError(f"Unknown spread layout: {layout}")

    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    for mask, (dx, dy) in zip(masks, deltas):
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            continue
        min_x = min(min_x, float((xs + dx).min()))
        max_x = max(max_x, float((xs + dx).max()))
        min_y = min(min_y, float((ys + dy).min()))
        max_y = max(max_y, float((ys + dy).max()))

    if not np.isfinite(min_x):
        rh, rw = ref.shape[:2]
        min_x, max_x = 0.0, float(rw)
        min_y, max_y = 0.0, float(rh)

    ox = min_x - margin
    oy = min_y - margin
    canvas_w = int(np.ceil(max_x - ox + margin))
    canvas_h = int(np.ceil(max_y - oy + margin))
    canvas_w = max(canvas_w, 1)
    canvas_h = max(canvas_h, 1)

    tiled = tile_ref_to_canvas(ref, canvas_w, canvas_h)
    canvas_f = tiled.astype(np.float32) / 255.0

    tone = float(np.clip(edge_tone, 0.0, 0.95))
    feath = max(0.0, float(feather_sigma))

    matt = matte_mode if matte_mode in ("gaussian", "smooth") else "gaussian"

    for f, mask, (dx, dy) in zip(frames, masks, deltas):
        layer, stencil = _warp_layer_to_canvas(
            f, mask, dx, dy, ox, oy, canvas_h, canvas_w,
        )
        alpha = stencil_to_alpha(stencil, feath, matt)

        fg_eff = layer * (1.0 - tone * (1.0 - alpha))
        canvas_f = canvas_f * (1.0 - alpha) + fg_eff * alpha

    out = (canvas_f * 255.0).clip(0, 255).astype(np.uint8)

    positions_canvas = [
        (centroids[i][0] + deltas[i][0] - ox, centroids[i][1] + deltas[i][1] - oy)
        for i in range(n)
    ]

    if pose_captions is not None and len(pose_captions) == n:
        rh0 = ref.shape[0]
        draw_pose_captions(
            out,
            positions_canvas,
            pose_captions,
            spread_px,
            rh0,
            position=caption_position,
            stride_px=caption_stride,
        )

    return out


def build_composite_blend(
    frames: list[np.ndarray],
    alpha_start: float,
    alpha_end: float,
    base: np.ndarray,
) -> np.ndarray:
    """Linear alpha stack; starts from base (reference frame), not a flat colour."""
    if not frames:
        raise ValueError("No frames to composite.")

    canvas = base.astype(np.float32) / 255.0
    n = len(frames)
    for i, frame in enumerate(frames):
        t = i / (n - 1) if n > 1 else 1.0
        alpha = alpha_start + t * (alpha_end - alpha_start)
        frame_f = frame.astype(np.float32) / 255.0
        canvas = (1.0 - alpha) * canvas + alpha * frame_f

    return (canvas * 255.0).clip(0, 255).astype(np.uint8)


def build_composite(
    frames: list[np.ndarray],
    alpha_start: float,
    alpha_end: float,
    bg_color: tuple[int, int, int] = BACKGROUND_COLOR,
    base_frame: np.ndarray | None = None,
) -> np.ndarray:
    """
    Legacy whole-frame blend onto flat bg_color, unless base_frame is given
    (then same as build_composite_blend without separate entry).
    """
    if not frames:
        raise ValueError("No frames to composite.")

    if base_frame is not None:
        return build_composite_blend(frames, alpha_start, alpha_end, base_frame)

    h, w = frames[0].shape[:2]
    canvas_bgr = np.array([bg_color[2], bg_color[1], bg_color[0]], dtype=np.float32) / 255.0
    canvas = np.full((h, w, 3), canvas_bgr, dtype=np.float32)

    n = len(frames)
    for i, frame in enumerate(frames):
        t = i / (n - 1) if n > 1 else 1.0
        alpha = alpha_start + t * (alpha_end - alpha_start)
        frame_f = frame.astype(np.float32) / 255.0
        canvas = (1.0 - alpha) * canvas + alpha * frame_f

    return (canvas * 255.0).clip(0, 255).astype(np.uint8)


def build_composite_max(ref: np.ndarray, frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel max; ref seeds the output so static dim regions stay put."""
    out = ref.copy()
    for f in frames:
        out = np.maximum(out, f)
    return out


def add_frame_ticks(composite: np.ndarray,
                    n_frames: int,
                    fps: float,
                    frame_indices: list[int],
                    font_scale: float = 0.45) -> np.ndarray:
    """Stamp time / frame labels at the bottom."""
    result = composite.copy()
    h, w = result.shape[:2]
    for i, fi in enumerate(frame_indices):
        t_sec = fi / fps if fps > 0 else fi
        label = f"f{fi}" if fps <= 0 else f"{t_sec:.2f}s"
        x = int(i / max(n_frames - 1, 1) * (w - 40)) + 5
        y = h - 6
        cv2.putText(result, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (80, 80, 80), 1, cv2.LINE_AA)
    return result


def vstack_with_labels(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """Stack images vertically with a label banner above each."""
    max_w = max(img.shape[1] for img in images)
    banner_h = 28
    rows = []
    for img, label in zip(images, labels):
        h, w = img.shape[:2]
        if w < max_w:
            pad = np.full((h, max_w - w, 3), 255, dtype=np.uint8)
            img = np.hstack([img, pad])
        banner = np.full((banner_h, max_w, 3), 230, dtype=np.uint8)
        cv2.putText(banner, label, (6, banner_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        rows.append(banner)
        rows.append(img)
    return np.vstack(rows)


def build_concat_overlap(frames: list[np.ndarray], overlap_frac: float) -> np.ndarray:
    """
    Horizontally stack full frames: each row of pixels is a hard BGR copy; later frames
    overwrite the shared overlap in reading order. Stride = W - overlap_px.
    ``overlap_frac=0`` abuts frames with no overlap; ~0.4–0.5 is typical for reusing
    the right background of the previous panel.
    """
    if not frames:
        raise ValueError("No frames to composite.")
    h0, w0 = frames[0].shape[:2]
    out_frames: list[np.ndarray] = []
    for f in frames:
        if f.shape[0] != h0 or f.shape[1] != w0:
            f = cv2.resize(f, (w0, h0), interpolation=cv2.INTER_AREA)
        out_frames.append(f)

    ovl = float(np.clip(overlap_frac, 0.0, 0.99))
    overlap_px = int(min(w0 - 1, max(0, round(w0 * ovl))))
    stride = max(1, w0 - overlap_px)
    n = len(out_frames)
    canvas_w = stride * (n - 1) + w0
    canvas = np.zeros((h0, canvas_w, 3), dtype=np.uint8)
    for i, f in enumerate(out_frames):
        x0 = i * stride
        canvas[:, x0 : x0 + w0] = f
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Single-input pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_one(
    mp4_path: Path,
    out_path: Path,
    n_frames: int,
    alpha_start: float,
    alpha_end: float,
    start_frac: float,
    end_frac: float,
    add_labels: bool,
    mode: str,
    diff_thresh: int,
    blur_sigma: float,
    morph_kernel: int,
    ref_frac: float,
    layout: str,
    spread_px: int,
    spread_margin: int,
    feather_sigma: float,
    edge_tone: float,
    no_pose_captions: bool,
    caption_position: str,
    mask_crop_top: int,
    mask_crop_bottom: int,
    anchor: str,
    matte_mode: str,
    overlap_frac: float,
) -> None:
    ref_frac = float(np.clip(ref_frac, 0.0, 1.0))

    gif_all: list[np.ndarray] | None = None
    gif_fps: float | None = None

    if mp4_path.suffix.lower() == ".gif":
        gif_all, total, gif_fps = load_gif_all(mp4_path)
        frames = extract_frames_from_list(gif_all, n_frames, start_frac, end_frac)
        ref_idx = int(ref_frac * (total - 1))
        ref = gif_all[ref_idx].copy()
    else:
        cap = cv2.VideoCapture(str(mp4_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total <= 0:
            print(f"  [error] bad frame count for {mp4_path}", file=sys.stderr)
            return

        frames = extract_frames_opencv(mp4_path, n_frames, start_frac, end_frac)
        ref_idx = int(ref_frac * (total - 1))
        ref = read_frame_opencv(mp4_path, ref_idx)

    if not frames:
        print(f"  [error] no frames extracted from {mp4_path}", file=sys.stderr)
        return

    first_i = int(start_frac * (total - 1))
    last_i = int(end_frac * (total - 1))
    if n_frames == 1:
        frame_indices = [first_i]
    else:
        frame_indices = [
            int(first_i + i * (last_i - first_i) / (n_frames - 1))
            for i in range(n_frames)
        ]

    pose_caps: list[tuple[str, str]] | None = None
    if (
        mode == "strobe"
        and layout in ("spread_h", "spread_v", "adjoin")
        and not no_pose_captions
    ):
        pose_caps = []
        for fi, fr in zip(frame_indices[: len(frames)], frames):
            ps, pr = try_parse_step_reward_hud(fr)
            pose_caps.append(
                (ps if ps is not None else str(fi), pr if pr is not None else "N/A"),
            )

    if mode == "strobe":
        print(f"  Processing: {mp4_path.name}  ({n_frames} frames, mode=strobe, "
              f"layout={layout}, thresh={diff_thresh}, ref_idx={ref_idx})")
        if layout == "concat":
            composite = build_concat_overlap(frames, overlap_frac)
        else:
            composite = build_strobe_composite(
                frames,
                ref,
                diff_thresh,
                blur_sigma,
                morph_kernel,
                layout=layout,
                spread_px=spread_px,
                spread_margin=spread_margin,
                feather_sigma=feather_sigma,
                edge_tone=edge_tone,
                pose_captions=pose_caps,
                caption_position=caption_position,
                mask_crop_top=mask_crop_top,
                mask_crop_bottom=mask_crop_bottom,
                anchor=anchor,
                matte_mode=matte_mode,
                overlap_frac=overlap_frac,
            )
    elif mode == "blend":
        print(f"  Processing: {mp4_path.name}  ({n_frames} frames, mode=blend, "
              f"alpha {alpha_start:.2f}->{alpha_end:.2f})")
        composite = build_composite_blend(frames, alpha_start, alpha_end, ref)
    elif mode == "max":
        print(f"  Processing: {mp4_path.name}  ({n_frames} frames, mode=max)")
        composite = build_composite_max(ref, frames)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total, fps = video_metadata(mp4_path, gif_frames=gif_all, gif_fps=gif_fps)

    if add_labels:
        composite = add_frame_ticks(
            composite, n_frames, fps, frame_indices[: len(frames)],
        )

    cv2.imwrite(str(out_path), composite)
    print(f"  Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Overlapping-frame composite from motion video (GIF/MP4/…).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("inputs", nargs="+",
                   help="Video file(s) or folder(s); folders are scanned recursively for "
                        "env_rgb_array/; longest video per folder is used.")
    p.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES,
                   help=f"Number of frames to sample (default: {DEFAULT_N_FRAMES}).")
    p.add_argument(
        "--mode",
        choices=("strobe", "blend", "max"),
        default=DEFAULT_MODE,
        help=f"strobe=mask paste on ref (default); blend=alpha trail on ref; "
             f"max=per-pixel max vs ref (default: {DEFAULT_MODE}).",
    )
    p.add_argument("--alpha-start", type=float, default=DEFAULT_ALPHA_START,
                   help="blend mode: opacity of earliest frame (default: 0.20).")
    p.add_argument("--alpha-end", type=float, default=DEFAULT_ALPHA_END,
                   help="blend mode: opacity of newest frame (default: 1.00).")
    p.add_argument("--start-frac", type=float, default=0.0,
                   help="Start sampling at this fraction of the clip (0–1).")
    p.add_argument("--end-frac", type=float, default=1.0,
                   help="End sampling at this fraction of the clip (0–1).")
    p.add_argument(
        "--ref-frac",
        type=float,
        default=DEFAULT_REF_FRAC,
        help="Which frame (by fraction of clip length) is the static background / "
             f"blend base (default: {DEFAULT_REF_FRAC} = first frame).",
    )
    p.add_argument("--diff-thresh", type=int, default=DEFAULT_DIFF_THRESH,
                   help=f"strobe: absdiff threshold 0–255 (default: {DEFAULT_DIFF_THRESH}).")
    p.add_argument("--blur-sigma", type=float, default=DEFAULT_BLUR_SIGMA,
                   help="strobe: Gaussian blur on diff before threshold (0=off).")
    p.add_argument(
        "--morph-kernel",
        type=int,
        default=DEFAULT_MORPH_KERNEL,
        help=f"strobe: morphological close kernel size (0=off; default {DEFAULT_MORPH_KERNEL}).",
    )
    p.add_argument(
        "--layout",
        choices=("concat", "in_place", "spread_h", "spread_v", "adjoin"),
        default=DEFAULT_LAYOUT,
        help="strobe: concat=hard-paste horizontal strip (default); spread_* / adjoin use "
             "motion masks + blending; in_place=center stack.",
    )
    p.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP_FRAC,
        metavar="FRAC",
        dest="overlap_frac",
        help="concat/adjoin: fraction of frame WIDTH that overlaps the previous panel "
             f"(concat: hard overwrite; default {DEFAULT_OVERLAP_FRAC}). Use 0 for no overlap.",
    )
    p.add_argument(
        "--mask-crop-top",
        type=int,
        default=0,
        metavar="PX",
        help="Zero top PX rows of motion mask (HUD). Default 0; --neat sets a preset.",
    )
    p.add_argument(
        "--mask-crop-bottom",
        type=int,
        default=0,
        metavar="PX",
        help="Zero bottom PX rows (timeline). Default 0; --neat sets a preset.",
    )
    p.add_argument(
        "--anchor",
        choices=("full", "body"),
        default="full",
        help="Centroid/largest blob: full mask or largest connected region (body). "
             "'--neat' forces body.",
    )
    p.add_argument(
        "--matte",
        choices=("gaussian", "smooth"),
        default="gaussian",
        dest="matte_mode",
        help="Alpha falloff at edges: gaussian blur or smooth (distance transform). "
             "'--neat' uses smooth.",
    )
    p.add_argument(
        "--neat",
        action="store_true",
        help="Preset: HUD/timeline crop, body anchor, smooth matte, stronger feather.",
    )
    p.add_argument(
        "--spread-pixels",
        type=int,
        default=DEFAULT_SPREAD_PX,
        help=f"strobe spread_* : spacing between successive pose centres (default: {DEFAULT_SPREAD_PX}).",
    )
    p.add_argument(
        "--spread-margin",
        type=int,
        default=DEFAULT_SPREAD_MARGIN,
        help=f"strobe spread_* : outer margin in px (default: {DEFAULT_SPREAD_MARGIN}).",
    )
    p.add_argument(
        "--feather-sigma",
        type=float,
        default=DEFAULT_FEATHER_SIGMA,
        help="strobe spread_* : Gaussian sigma for soft mask edges (0=hard; "
             f"default: {DEFAULT_FEATHER_SIGMA}).",
    )
    p.add_argument(
        "--edge-tone",
        type=float,
        default=DEFAULT_EDGE_TONE,
        help="strobe spread_* : darkening of semi-transparent fringe (0=no tint; "
             f"default: {DEFAULT_EDGE_TONE}).",
    )
    p.add_argument(
        "--no-pose-captions",
        action="store_true",
        help="strobe spread_* : do not draw per-pose step/reward captions.",
    )
    p.add_argument(
        "--caption-position",
        choices=("below", "above"),
        default="below",
        help="Where to place pose captions relative to centroid (default: below).",
    )
    p.add_argument("--out", default=None,
                   help="Output PNG path (single input only). "
                        f"Defaults to <input_dir>/{OUTPUT_FILENAME}.")
    p.add_argument("--stack", action="store_true",
                   help="Stack multiple-input composites vertically into comparison.png.")
    p.add_argument("--no-labels", action="store_true",
                   help="Suppress time labels at the bottom.")
    p.add_argument("--bg", nargs=3, type=int, default=[255, 255, 255],
                   metavar=("R", "G", "B"),
                   help="blend with flat init only (no --ref): unused for strobe/max "
                        f"(default: 255 255 255).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if getattr(args, "neat", False) and getattr(args, "layout", "") != "concat":
        args.mask_crop_top = DEFAULT_MASK_CROP_TOP
        args.mask_crop_bottom = DEFAULT_MASK_CROP_BOTTOM
        args.anchor = "body"
        args.matte_mode = DEFAULT_MATTE
        args.feather_sigma = max(args.feather_sigma, 14.0)
        args.edge_tone = min(args.edge_tone, 0.34)
        args.spread_margin = max(args.spread_margin, 52)

    global BACKGROUND_COLOR
    BACKGROUND_COLOR = tuple(args.bg)

    composites = []
    labels = []

    for inp in args.inputs:
        try:
            jobs = resolve_input_jobs(inp)
        except FileNotFoundError as e:
            print(f"[skip] {e}", file=sys.stderr)
            continue

        for mp4_path, out_dir in jobs:
            if len(args.inputs) == 1 and args.out and len(jobs) == 1:
                out_path = Path(args.out)
            else:
                out_path = out_dir / OUTPUT_FILENAME

            process_one(
                mp4_path=mp4_path,
                out_path=out_path,
                n_frames=args.n_frames,
                alpha_start=args.alpha_start,
                alpha_end=args.alpha_end,
                start_frac=args.start_frac,
                end_frac=args.end_frac,
                add_labels=not args.no_labels,
                mode=args.mode,
                diff_thresh=args.diff_thresh,
                blur_sigma=args.blur_sigma,
                morph_kernel=args.morph_kernel,
                ref_frac=args.ref_frac,
                layout=args.layout,
                spread_px=args.spread_pixels,
                spread_margin=args.spread_margin,
                feather_sigma=args.feather_sigma,
                edge_tone=args.edge_tone,
                no_pose_captions=args.no_pose_captions,
                caption_position=args.caption_position,
                mask_crop_top=args.mask_crop_top,
                mask_crop_bottom=args.mask_crop_bottom,
                anchor=args.anchor,
                matte_mode=args.matte_mode,
                overlap_frac=args.overlap_frac,
            )

            if args.stack:
                img = cv2.imread(str(out_path))
                if img is not None:
                    composites.append(img)
                    labels.append(mp4_path.stem)

    if args.stack and composites:
        stacked = vstack_with_labels(composites, labels)
        stack_path = Path("comparison.png")
        cv2.imwrite(str(stack_path), stacked)
        print(f"\nComparison image saved -> {stack_path.resolve()}")


if __name__ == "__main__":
    main()
