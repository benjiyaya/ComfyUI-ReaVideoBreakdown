"""Shot boundary detection + a light measured camera-motion estimate.

Pure numpy/OpenCV, CPU-only. This is the part that is NOT native to ComfyUI:
splitting a frame batch into shots and measuring global camera movement. The
semantic reading of each shot is done by the VLM (see vlm.py), not here.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

THUMB_W = 96
HIST_BINS = (16, 16, 8)


@dataclass
class Shot:
    index: int
    start_frame: int
    end_frame: int            # exclusive
    transition_in: str = "cut"
    boundary_score: float = 0.0

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self, fps: float) -> dict:
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start": round(self.start_frame / fps, 3),
            "end": round(self.end_frame / fps, 3),
            "duration": round(self.duration_frames / fps, 3),
            "transition_in": self.transition_in,
            "boundary_score": round(self.boundary_score, 3),
        }


def _u8(frame: np.ndarray) -> np.ndarray:
    return (np.clip(frame, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _small(frame: np.ndarray, w: int) -> np.ndarray:
    h, ww = frame.shape[:2]
    if ww <= w:
        return frame
    return cv2.resize(frame, (w, max(1, int(round(h * w / ww)))), interpolation=cv2.INTER_AREA)


def _tracks(frames: list[np.ndarray]) -> dict:
    n = len(frames)
    hist_dist = np.zeros(n, np.float32)
    pix_diff = np.zeros(n, np.float32)
    luma = np.zeros(n, np.float32)
    prev_hist = prev_gray = None
    for i, f in enumerate(frames):
        u8 = _small(_u8(f), THUMB_W)
        gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        luma[i] = float(gray.mean()) / 255.0
        if prev_hist is not None:
            hist_dist[i] = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            pix_diff[i] = float(np.abs(gray - prev_gray).mean()) / 255.0
        prev_hist, prev_gray = hist, gray
    return {"hist_dist": hist_dist, "pix_diff": pix_diff, "luma": luma}


def _thresholds(dist: np.ndarray) -> tuple[float, float]:
    d = np.asarray(dist[1:], np.float64)
    if d.size < 3:
        return 0.15, 0.28
    t_hi = max(0.28, float(d.mean() + 3.0 * d.std() + 0.02))
    return max(0.15, 0.55 * t_hi), t_hi


def detect_shots(frames: list[np.ndarray], fps: float = 30.0,
                 sensitivity: float = 0.5, min_shot_s: float = 0.24) -> list[Shot]:
    n = len(frames)
    if n == 0:
        return []
    t = _tracks(frames)
    dist, pix, luma = t["hist_dist"], t["pix_diff"], t["luma"]
    thr_lo, thr_hi = _thresholds(dist)
    k = 1.35 - 0.7 * float(np.clip(sensitivity, 0.0, 1.0))
    thr_lo, thr_hi = thr_lo * k, thr_hi * k
    score = 0.75 * dist + 0.25 * np.clip(pix * 4.0, 0.0, 1.0)

    cuts: dict[int, tuple[str, float]] = {}
    for i in range(1, n):
        if score[i] < thr_hi:
            continue
        if i > 1 and score[i] < score[i - 1]:
            continue
        if i + 1 < n and score[i] < score[i + 1]:
            continue
        prev = cuts.get(i)
        if prev is None or prev[1] < score[i]:
            cuts[i] = ("cut", float(score[i]))

    sm_luma = np.convolve(luma, np.ones(5) / 5.0, mode="same")
    for i, (typ, s) in list(cuts.items()):
        lo0, hi0 = max(1, i - 3), min(n - 1, i + 4)
        if sm_luma[lo0:hi0].min() < 0.07:
            cuts[i] = ("fade", s)
        elif sm_luma[lo0:hi0].max() > 0.92 and dist[i] < thr_hi * 1.6:
            cuts[i] = ("flash", s)

    plateau = np.convolve(score, np.ones(7) / 7.0, mode="same")
    in_block, block_start = False, 0
    for i in range(1, n - 1):
        if plateau[i] > thr_lo and not in_block:
            in_block, block_start = True, i
        elif plateau[i] <= thr_lo and in_block:
            in_block = False
            if (i - block_start) >= 4 and not any(block_start <= c < i for c in cuts):
                cuts[(block_start + i) // 2] = ("dissolve", float(plateau[i]))
    if in_block and not any(block_start <= c < n for c in cuts):
        cuts[min(n - 1, (block_start + n) // 2)] = ("dissolve", float(plateau[-1]))

    shots: list[Shot] = []
    start = 0
    for b in sorted(cuts):
        if b <= start:
            continue
        typ, sc = cuts[b]
        shots.append(Shot(len(shots), start, b, typ, sc))
        start = b
    if start < n:
        shots.append(Shot(len(shots), start, n, "start" if not shots else "cut", 0.0))

    min_len = max(2, int(round(min_shot_s * fps)))
    merged: list[Shot] = []
    for sh in shots:
        if merged and sh.duration_frames < min_len:
            merged[-1].end_frame = sh.end_frame
            if sh.transition_in == "fade":
                merged[-1].transition_in = "fade"
        else:
            merged.append(sh)
    if not merged:
        merged = [Shot(0, 0, n, "start", 0.0)]
    for i, sh in enumerate(merged):
        sh.index = i
    return merged


def sample_indices(start: int, end: int, cap: int) -> list[int]:
    n = end - start
    if n <= 0:
        return []
    if n <= cap:
        return list(range(start, end))
    idx = np.linspace(start, end - 1, cap).round().astype(int)
    out, seen = [], set()
    for v in idx.tolist():
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def camera_motion(frames: list[np.ndarray], shot: Shot, fps: float, max_pairs: int = 12) -> dict:
    """Measured global camera move: signed pan (px/s), zoom (%/s), and a label.
    Cheap optical-flow + RANSAC similarity fit on a few consecutive pairs."""
    idx = sample_indices(shot.start_frame, min(shot.end_frame, len(frames)), max_pairs + 1)
    if len(idx) < 2:
        return {"camera": "static", "pan_px_per_s": 0.0, "zoom_pct_per_s": 0.0}
    grays = [cv2.cvtColor(_small(_u8(frames[i]), 240), cv2.COLOR_RGB2GRAY) for i in idx]
    dxs, dys, zs = [], [], []
    for j in range(1, len(grays)):
        p0 = cv2.goodFeaturesToTrack(grays[j - 1], 300, 0.01, 8)
        if p0 is None or len(p0) < 8:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(grays[j - 1], grays[j], p0, None, winSize=(15, 15), maxLevel=2)
        if p1 is None:
            continue
        good = st.ravel() == 1
        a, b = p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)
        if len(a) < 8:
            continue
        m, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if m is None or inl is None or inl.sum() < 3:
            continue
        dxs.append(float(m[0, 2])); dys.append(float(m[1, 2]))
        zs.append(float(np.log(max(1e-3, np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)))))
    if not dxs:
        return {"camera": "static", "pan_px_per_s": 0.0, "zoom_pct_per_s": 0.0}
    dt = 1.0 / max(1e-6, fps)
    pan = float(np.hypot(np.mean(dxs), np.mean(dys))) / dt
    zoom = float(np.mean(zs)) * 100.0 / dt
    jitter = float(np.std(np.hypot(dxs, dys)))
    h = grays[0].shape[0]
    moving = pan / max(h, 1) * 2.0 + abs(zoom) / 3.0
    if moving > 0.35:
        lab = "handheld" if jitter > max(0.6, 0.9 * pan * dt) else "smooth_move"
    elif moving > 0.05:
        lab = "smooth_move"
    else:
        lab = "static"
    return {"camera": lab, "pan_px_per_s": round(pan, 1), "zoom_pct_per_s": round(zoom, 2)}
