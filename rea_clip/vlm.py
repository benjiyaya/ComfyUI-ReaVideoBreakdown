"""Per-shot vision understanding via the NATIVE ComfyUI VLM text encoder.

This reuses the exact calling path of ComfyUI's core "Generate Text" node
(comfy_extras/nodes_textgen.py :: TextGenerateV2):

    tokens = clip.tokenize(prompt, videos=[frames], system_prompt=..., ...)
    ids    = clip.generate(tokens, max_length=..., temperature=..., ...)
    text   = clip.decode(ids)

The loaded CLIP *is* the vision-language model (Qwen-VL / JoyImage / Gemma / ...
via CLIPLoader). No separate clip_vision tower, no external API, no JoyCaption.
Each shot's frames are passed as a video batch so the model sees motion, and it
returns a structured reverse-engineering record + a ready-to-use video prompt.
"""
from __future__ import annotations

import json
import re

# The production-method label set used by rea-video.
LABELS = ["live_action", "ai_video", "avatar", "cg_3d", "code_render", "screen_capture",
          "still_motion", "static_hold", "anim_2d", "stop_motion", "film_footage"]

SYSTEM_EN = """You are a video reverse-engineering analyst. You are shown the frames of ONE shot from a video, in chronological order. Describe ONLY what is actually visible and moving; do not invent people, objects, text or actions that are not there.

Return STRICT JSON (no markdown fences, no commentary) with exactly these keys:
{
  "description": "2-4 sentences: subject(s) and appearance, setting, lighting, colour, style, and what happens across the shot",
  "made_of": "one of: live_action, ai_video, avatar, cg_3d, code_render, screen_capture, still_motion, static_hold, anim_2d, stop_motion, film_footage",
  "made_of_reason": "one short sentence of visual evidence for that judgement",
  "shot_size": "one of: extreme_close_up, close_up, medium, wide, extreme_wide, top_down, pov",
  "camera_move": "what the camera does: static / pan left / pan right / tilt / push in / pull out / track / handheld / orbit, with speed",
  "subject_motion": "what moves in the frame and how",
  "lighting": "short lighting description",
  "style": "short visual style / grade description",
  "video_prompt": "a single ready-to-use text-to-video prompt: one continuous paragraph, present-progressive verbs, chronological, objective and observable, weaving in shot size + camera move + viewpoint + lighting + style. No headings, no timestamps, no 'the scene opens'. This prompt must let a video model recreate THIS shot."
}
Judge made_of from visual evidence: real camera footage = live_action; smooth dreamlike / melted details = ai_video; fixed talking presenter = avatar; shaded 3D surfaces = cg_3d; flat vector / text / UI motion graphics = code_render; app or desktop interface = screen_capture; a single photo with pan/zoom = still_motion; a frozen card = static_hold. If unsure, pick the closest and lower your confidence in made_of_reason."""

SYSTEM_ZH = """你是视频逆向拆解分析师。你会看到某个视频【一个镜头】的按时间顺序排列的画面帧。只描述真实可见、真实在运动的内容；不要虚构不存在的人物、物体、文字或动作。

只返回严格 JSON（不要 markdown 代码块、不要多余说明），键如下：
{
  "description": "2-4 句：主体及外观、场景、灯光、色彩、风格，以及镜头中发生的变化",
  "made_of": "取值之一: live_action, ai_video, avatar, cg_3d, code_render, screen_capture, still_motion, static_hold, anim_2d, stop_motion, film_footage",
  "made_of_reason": "一句视觉证据说明该判断",
  "shot_size": "取值之一: extreme_close_up, close_up, medium, wide, extreme_wide, top_down, pov",
  "camera_move": "相机运动: 固定 / 左摇 / 右摇 / 俯仰 / 推近 / 拉远 / 跟移 / 手持 / 环绕，含速度",
  "subject_motion": "画面中什么在动、如何动",
  "lighting": "简短灯光描述",
  "style": "简短视觉风格/调色描述",
  "video_prompt": "一段可直接使用的文生视频提示词：单段连续文字、现在进行时、按时间顺序、客观可观察，自然融入景别+运镜+视角+灯光+风格。不要标题、不要时间戳、不要'画面开始'。该提示词须能让视频模型复刻此镜头。"
}
依据视觉证据判断 made_of：真实相机拍摄=live_action；平滑梦幻/细节融化=ai_video；固定口播主播=avatar；有光影的 3D 表面=cg_3d；扁平矢量/文字/UI 动效=code_render；应用或桌面界面=screen_capture；单张照片推拉=still_motion；定格卡片=static_hold。不确定时选最接近者并在 made_of_reason 中降低把握。"""

USER_TMPL = ("Shot duration: {duration:.2f}s. Measured global camera: {camera} "
             "(pan {pan:.0f}px/s, zoom {zoom:+.1f}%/s). Analyse these frames and return the JSON.")


def _strip_think(text: str) -> str:
    _, sep, body = text.partition("</think>")
    return (body if sep else text).strip()


def parse_record(text: str) -> dict:
    """Pull the JSON object out of a VLM reply, tolerating fences / prose."""
    t = _strip_think(text)
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # last resort: keep the raw text as the prompt so nothing is lost
    return {"description": t, "video_prompt": t, "made_of": "unknown", "_raw": True}


def _to_torch(frames_np):
    import torch
    import numpy as np
    arr = np.ascontiguousarray(np.stack(frames_np, 0).astype(np.float32))
    return torch.from_numpy(arr)  # [N,H,W,3] NHWC, 0..1


def read_shot(clip, frames_np, duration: float, camera: dict, lang: str = "en",
              max_length: int = 1024, temperature: float = 0.4, seed: int = 0,
              max_frames: int = 16, max_side: int = 512) -> dict:
    """Run the native VLM generate path on one shot's frames. Returns a record dict.
    frames_np: list of HWC RGB float32 0..1 arrays (already sampled for the shot)."""
    import cv2
    import numpy as np

    # bound vision cost: subsample frames + downscale
    if len(frames_np) > max_frames:
        idx = np.linspace(0, len(frames_np) - 1, max_frames).round().astype(int)
        frames_np = [frames_np[i] for i in idx]
    sized = []
    for f in frames_np:
        h, w = f.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            f = cv2.resize(f, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        sized.append(f)
    batch = _to_torch(sized)

    system = SYSTEM_ZH if lang == "zh" else SYSTEM_EN
    user = USER_TMPL.format(duration=duration, camera=camera.get("camera", "static"),
                            pan=camera.get("pan_px_per_s", 0.0), zoom=camera.get("zoom_pct_per_s", 0.0))

    # Mirror TextGenerateV2 exactly. Prefer video (motion); fall back to stills.
    try:
        tokens = clip.tokenize(user, videos=[batch], system_prompt=system,
                               skip_template=False, min_length=1, thinking=False)
    except TypeError:
        imgs = [batch[i:i + 1] for i in range(batch.shape[0])]
        tokens = clip.tokenize(user, images=imgs, system_prompt=system,
                               skip_template=False, min_length=1, thinking=False)

    ids = clip.generate(tokens, do_sample=True, max_length=max_length,
                        temperature=temperature, top_k=64, top_p=0.95, min_p=0.0,
                        repetition_penalty=1.05, presence_penalty=0.0, seed=seed, mtp=True)
    text = clip.decode(ids)
    rec = parse_record(text if isinstance(text, str) else str(text))
    rec.setdefault("video_prompt", rec.get("description", ""))
    rec.setdefault("made_of", "unknown")
    if rec.get("made_of") not in LABELS:
        rec["made_of"] = "unknown"
    return rec
