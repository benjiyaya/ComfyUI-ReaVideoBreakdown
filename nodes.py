"""ComfyUI custom node: rea-video-style shot-by-shot reverse engineering.

Vision understanding uses the NATIVE ComfyUI VLM text encoder you load with
CLIPLoader (Qwen-VL / JoyImage / Gemma / ...). It calls the same path as the core
"Generate Text" node: clip.tokenize(videos=[frames]) -> clip.generate -> clip.decode.
There is NO clip_vision input and no external API.

The node adds only what ComfyUI does not do natively: split the video into shots,
measure global camera motion, and turn each shot into a structured record + a
ready-to-use text-to-video prompt.
"""
from __future__ import annotations

import traceback

import numpy as np
import torch

from .rea_clip.shots import detect_shots, sample_indices, camera_motion
from .rea_clip.vlm import read_shot
from .rea_clip.report import build_breakdown, render_video_md, render_prompts, render_summary

DEFAULT_FPS = 25.0


def _frames_to_numpy(images) -> list[np.ndarray]:
    if hasattr(images, "detach"):
        images = images.detach().cpu().float().numpy()
    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return [np.clip(arr[i], 0.0, 1.0) for i in range(arr.shape[0])]


def _get_fps(fps) -> float:
    try:
        f = float(fps)
    except Exception:
        f = 0.0
    return f if f > 0 else DEFAULT_FPS


def _keyframes_tensor(frames: list[np.ndarray], shots) -> torch.Tensor:
    picks = []
    for sh in shots:
        i = sh.start_frame
        if 0 <= i < len(frames):
            picks.append(frames[i])
    if not picks:
        return torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    return torch.from_numpy(np.stack(picks, 0).astype(np.float32))


class ReaVideoBreakdown:
    """Reverse-engineer a video shot by shot with your native VLM CLIP: how each
    shot was made, what is in it, camera work, and a ready-to-use video prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "every video frame as an IMAGE batch (LoadVideo -> GetVideoComponents, VHS, etc.)"}),
                "fps": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 1000.0, "step": 0.001,
                                   "tooltip": "frame rate of the batch; 0 = assume 25"}),
                "clip": ("CLIP", {"tooltip": "a VLM text encoder loaded via CLIPLoader (qwen_image / joyimage / gemma / ...). This IS the vision model; no clip_vision needed."}),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "cut detector sensitivity (higher = more shots)"}),
                "min_shot_seconds": ("FLOAT", {"default": 0.24, "min": 0.05, "max": 5.0, "step": 0.01}),
                "frames_per_shot": ("INT", {"default": 8, "min": 1, "max": 32, "tooltip": "frames sampled per shot and shown to the VLM (as a video batch)"}),
                "max_length": ("INT", {"default": 1024, "min": 128, "max": 8192, "step": 64, "tooltip": "VLM max generated tokens per shot"}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "lower = more deterministic JSON"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "language": (["en", "zh"], {"default": "en"}),
                "max_side": ("INT", {"default": 512, "min": 224, "max": 1024, "step": 32, "tooltip": "downscale frames to this max side before the VLM (bounds VRAM)"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "IMAGE", "INT")
    RETURN_NAMES = ("video_md", "breakdown_json", "prompts", "summary", "keyframes", "shot_count")
    FUNCTION = "analyze"
    CATEGORY = "RE-VIDEO/breakdown"
    DESCRIPTION = ("Shot-by-shot video reverse engineering (rea-video style) using the native ComfyUI VLM text encoder. "
                   "Outputs a VIDEO.md report, structured JSON, and one ready-to-use text-to-video prompt per shot. No clip_vision, no external API.")

    def analyze(self, images, fps, clip, sensitivity=0.5, min_shot_seconds=0.24, frames_per_shot=8,
                max_length=1024, temperature=0.4, seed=0, language="en", max_side=512):
        # defensive defaults (a stale/misaligned saved graph can pass None)
        sensitivity = 0.5 if sensitivity is None else float(sensitivity)
        min_shot_seconds = 0.24 if min_shot_seconds is None else float(min_shot_seconds)
        frames_per_shot = 8 if frames_per_shot is None else max(1, min(32, int(frames_per_shot)))
        max_length = 1024 if max_length is None else int(max_length)
        temperature = 0.4 if temperature is None else float(temperature)
        seed = 0 if seed is None else int(seed)
        max_side = 512 if max_side is None else int(max_side)
        language = "en" if language not in ("en", "zh") else language

        frames = _frames_to_numpy(images)
        if not frames:
            return ("(no frames)", "{}", "", "no frames", _keyframes_tensor([], []), 0)
        fps = _get_fps(fps)
        h, w = frames[0].shape[:2]
        meta = {"duration": round(len(frames) / fps, 3), "fps": fps, "width": int(w), "height": int(h),
                "frames": len(frames)}

        shots = detect_shots(frames, fps, sensitivity, min_shot_seconds)
        records, errors = [], []
        for sh in shots:
            d = sh.to_dict(fps)
            cam = camera_motion(frames, sh, fps)
            idx = sample_indices(sh.start_frame, sh.end_frame, frames_per_shot)
            shot_frames = [frames[i] for i in idx]
            rec = dict(d)
            rec.update(cam)
            try:
                v = read_shot(clip, shot_frames, d["duration"], cam, lang=language,
                              max_length=max_length, temperature=temperature, seed=seed,
                              max_frames=frames_per_shot, max_side=max_side)
                rec.update(v)
            except Exception as ex:
                traceback.print_exc()
                errors.append(f"shot {sh.index}: {ex}")
                rec.update({"description": "", "video_prompt": "", "made_of": "unknown",
                            "_error": str(ex)})
            records.append(rec)

        breakdown = build_breakdown(meta, records)
        if errors:
            breakdown["errors"] = errors
            breakdown["reader_warning"] = (
                "VLM reading failed for %d/%d shots. Load a VISION text encoder via CLIPLoader "
                "(type qwen_image / joyimage / gemma / qwen3vl ...), not a plain SD1.5 CLIP. First error: %s"
                % (len(errors), len(shots), errors[0]))
        video_md = render_video_md(breakdown, lang=language)
        if errors:
            video_md = "> WARNING: " + breakdown["reader_warning"] + "\n\n" + video_md
        prompts = render_prompts(breakdown)
        summary = render_summary(breakdown)
        if errors:
            summary += f" — VLM FAILED on {len(errors)}/{len(shots)} shots (load a VLM text encoder)"
        print(f"[ReaVideo] {summary}")
        import json
        return (video_md, json.dumps(breakdown, ensure_ascii=False), prompts, summary,
                _keyframes_tensor(frames, shots), len(shots))


NODE_CLASS_MAPPINGS = {"ReaVideoBreakdown": ReaVideoBreakdown}
NODE_DISPLAY_NAME_MAPPINGS = {"ReaVideoBreakdown": "Video Breakdown (Rea · native VLM)"}
