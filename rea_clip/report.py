"""Render the breakdown: a VIDEO.md-style report, structured JSON, and a
plain prompts string (one ready-to-use video prompt per shot) for downstream
text-to-video nodes. Every shot carries a real VLM description + prompt, not
placeholder filler.
"""
from __future__ import annotations

from .vlm import LABELS

LABEL_EN = {
    "live_action": "live action", "ai_video": "AI-generated video", "avatar": "AI avatar / talking head",
    "cg_3d": "3D render", "code_render": "code / template motion graphics", "screen_capture": "screen recording / screenshot",
    "still_motion": "still image + pan/zoom", "static_hold": "static hold", "anim_2d": "2D animation",
    "stop_motion": "stop motion", "film_footage": "film / TV footage", "unknown": "uncertain",
}
LABEL_ZH = {
    "live_action": "实拍", "ai_video": "AI 生成视频", "avatar": "数字人口播", "cg_3d": "3D 渲染",
    "code_render": "代码/模板动效", "screen_capture": "录屏/截图", "still_motion": "静图+推拉",
    "static_hold": "静帧停留", "anim_2d": "2D 动画", "stop_motion": "定格动画", "film_footage": "影视片段",
    "unknown": "不确定",
}
CAM_EN = {"static": "static camera", "smooth_move": "smooth camera move", "handheld": "handheld camera"}
CAM_ZH = {"static": "固定机位", "smooth_move": "平稳运镜", "handheld": "手持"}
TRANS_EN = {"cut": "cut", "fade": "fade", "dissolve": "dissolve", "flash": "flash", "start": "start"}


def tc(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    return f"{m:02d}:{sec - 60 * m:05.2f}"


def build_breakdown(meta: dict, records: list[dict]) -> dict:
    recs = sorted(records, key=lambda r: r["index"])
    comp: dict[str, float] = {}
    total = sum(max(0.0, r["end"] - r["start"]) for r in recs) or 1.0
    for r in recs:
        g = r.get("made_of", "unknown")
        comp[g] = comp.get(g, 0.0) + max(0.0, r["end"] - r["start"])
    comp = {k: round(v / total, 3) for k, v in sorted(comp.items(), key=lambda kv: -kv[1])}
    return {
        "format": "videomd-like",
        "subformat": "rea-video-comfyui-vlm",
        "reader": "native ComfyUI VLM text encoder (clip.tokenize -> generate -> decode); no clip_vision",
        "video": meta,
        "card": {"shots": len(recs), "duration": meta.get("duration", 0.0), "composition": comp},
        "shots": recs,
    }


def render_prompts(b: dict) -> str:
    """One ready-to-use video prompt per shot, clearly delimited for downstream nodes."""
    blocks = []
    for r in b["shots"]:
        blocks.append(
            f"[Shot {r['index']:02d} | {tc(r['start'])}-{tc(r['end'])} | {r['duration']:.2f}s | {r.get('made_of', 'unknown')}]\n"
            f"{(r.get('video_prompt') or r.get('description') or '').strip()}"
        )
    return "\n\n".join(blocks)


def render_summary(b: dict) -> str:
    card = b["card"]
    comp = ", ".join(f"{LABEL_EN.get(k, k)} {round(v * 100)}%" for k, v in card["composition"].items())
    return f"{card['shots']} shots, {b['video'].get('duration', 0):.1f}s @ {b['video'].get('fps', 0):.2f}fps; {comp}"


def render_video_md(b: dict, lang: str = "en") -> str:
    names = LABEL_ZH if lang == "zh" else LABEL_EN
    cam = CAM_ZH if lang == "zh" else CAM_EN
    meta, card = b["video"], b["card"]
    L = ["---", f"format: {b['format']}", f"subformat: {b['subformat']}", f"reader: {b['reader']}",
         f"duration: {meta.get('duration', 0):.2f}", f"fps: {meta.get('fps', 0):.3f}",
         f"size: {meta.get('width', 0)}x{meta.get('height', 0)}", f"shots: {card['shots']}", "---", ""]
    L.append("## 一句话结论" if lang == "zh" else "## Verdict")
    comp = " + ".join(f"{names.get(k, k)} {round(v * 100)}%" for k, v in card["composition"].items())
    L.append((f"{card['shots']} 个镜头 · " if lang == "zh" else f"{card['shots']} shots · ") + comp)
    L.append("")
    L.append("## 分镜时间线" if lang == "zh" else "## Timeline")
    L.append("| # | time | in | made | camera (measured) | description |")
    L.append("|---|------|----|------|-------------------|-------------|")
    for r in b["shots"]:
        desc = (r.get("description") or "").replace("|", "/").replace("\n", " ")
        if len(desc) > 90:
            desc = desc[:90] + "…"
        row = [f"{r['index']:02d}", f"{tc(r['start'])}-{tc(r['end'])}",
               TRANS_EN.get(r.get("transition_in", "cut"), r.get("transition_in", "")),
               names.get(r.get("made_of", "unknown"), r.get("made_of", "")),
               cam.get(r.get("camera", "static"), r.get("camera", "")), desc]
        L.append("| " + " | ".join(str(x) for x in row) + " |")
    L.append("")
    L.append("## 逐镜细节与复刻提示词" if lang == "zh" else "## Shot details & remake prompts")
    for r in b["shots"]:
        L.append(f"### #{r['index']:02d} {tc(r['start'])}-{tc(r['end'])} · {names.get(r.get('made_of', 'unknown'), '')}")
        if r.get("description"):
            L.append(r["description"])
        meta_bits = []
        if r.get("shot_size"):
            meta_bits.append(f"shot size: {r['shot_size']}")
        if r.get("camera_move"):
            meta_bits.append(f"camera: {r['camera_move']}")
        if r.get("subject_motion"):
            meta_bits.append(f"motion: {r['subject_motion']}")
        if r.get("lighting"):
            meta_bits.append(f"lighting: {r['lighting']}")
        if r.get("style"):
            meta_bits.append(f"style: {r['style']}")
        if meta_bits:
            L.append("- " + " · ".join(meta_bits))
        L.append(f"- measured camera~: {cam.get(r.get('camera', 'static'), '')} "
                 f"(pan {r.get('pan_px_per_s', 0):.0f}px/s, zoom {r.get('zoom_pct_per_s', 0):+.1f}%/s)")
        if r.get("made_of_reason"):
            L.append(f"- made_of evidence~: {r['made_of_reason']}")
        if r.get("video_prompt"):
            L.append(f"- **{'复刻提示词' if lang == 'zh' else 'video prompt'}:** {r['video_prompt']}")
        L.append("")
    L.append("_Generated by ComfyUI-ReaVideoBreakdown: shot detection + measured camera (OpenCV) and per-shot reading by the native ComfyUI VLM text encoder. `~` = tool-measured; descriptions/prompts come from the VLM seeing the actual frames._")
    return "\n".join(L)
