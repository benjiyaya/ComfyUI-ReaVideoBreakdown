"""MiniMax H3 Ref2VA prompt rewrite (visual references only, v1: no <Audio N>).

Consumes ReaVideoBreakdown's breakdown_json, a free-text user instruction, and
reference images; uses the native VLM (clip.tokenize -> generate -> decode) to
emit the official six-section H3 full-reference prompt, ready for the native
MiniMaxH3ReferenceToVideo node's `prompt` input.

Format ground truth: MiniMax-AI/MiniMax-H3 skills/h3-prompt-writing
(references/ref-en.txt). Audio is intentionally NOT managed here: the user wires
audio into the native node, whose audio_vae latent conditions the model regardless
of prompt tags.
"""
from __future__ import annotations

import json
import re

SECTIONS = ["subject_definitions", "summary", "retention_analysis",
            "detailed_description", "overall_soundscape", "non_diegetic_music"]

H3_SYSTEM = """You rewrite a shot-by-shot video breakdown into a MiniMax H3 full-reference (Ref2VA) video-generation prompt. You are shown reference images (image_1 = <Picture 1>, image_2 = <Picture 2>, ...). Apply the user's instruction to the source breakdown.

Output EXACTLY six sections, in this order, each starting with its literal header on its own line. No markdown fences, no extra commentary:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

Label rules:
- <Subject N> = reusable visible content (person, scene, costume, style) abstracted from the references; define it from its source image, e.g. "<Subject 1> is the woman in <Picture 1>, with ...".
- <Picture N> = reference image N, used as a frame/anchor or cited inside a subject definition.
- Do NOT use <Audio N> or <Video N>. Audio is handled outside this prompt.
- Keep every label's meaning identical across all sections.

Section rules:
- subject_definitions: one line per tracked subject/picture; state what the label denotes and the main features to follow.
- summary: one short paragraph beginning with a task-type prefix like [reference generation] (add " + keyframe completion" only if an image is a concrete frame anchor). Use only already-defined labels.
- retention_analysis: one line per label with a fixed marker: fully_preserved / partially_preserved / attribute_transfer / weak_reference, and which [Shot N] it appears in. When the user swaps a source character for a reference character, use attribute_transfer.
- detailed_description: one or two English style sentences BEFORE [Shot 1]. [Shot 1] has no timestamp; later shots use "[Shot N] At MM:SS.mmm, ...". Preserve the source shot order, cut times and camera movement unless the user overrides. Insert <Subject N>/<Picture N> at first appearance and where their role applies. Give vocal sources stable (S1),(S2) IDs; write dialogue as <d>[Language] ...</d>. Concrete visual/audio detail, not plot summary. The total timeline MUST fit the requested duration.
- overall_soundscape: ambience and physical sounds across the video.
- non_diegetic_music: audience-only background music (instrumentation, tempo, development) or "N/A".

If the user asks a character to speak, write it as that character's own voice: <Subject N> (Sx) says <d>[Language] ...</d>. Do not reference any audio asset.
Write sections in English; preserve dialogue/lyrics/visible text in their original language. Prefer concrete details over words like "cinematic"/"beautiful"."""


def parse_breakdown(text: str) -> dict:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("breakdown_json must be a JSON object")
    shots = obj.get("shots") or obj.get("timeline") or []
    if not isinstance(shots, list):
        raise ValueError("breakdown_json has no shots list")
    return obj


def source_summary(b: dict) -> tuple[str, float]:
    shots = b.get("shots") or b.get("timeline") or []
    lines = []
    for s in shots:
        i = s.get("index", len(lines))
        lines.append(
            f"[Shot {i + 1}] {s.get('start', 0):.3f}-{s.get('end', 0):.3f}s "
            f"(dur {s.get('duration', 0):.2f}s) camera={s.get('camera', 'static')}; "
            f"{(s.get('description') or s.get('video_prompt') or '').strip()}"
        )
    dur = float(b.get("video", {}).get("duration", 0.0)) or (shots[-1].get("end", 0.0) if shots else 0.0)
    return "\n".join(lines), dur


def build_user(source: str, n_images: int, instruction: str, target: float) -> str:
    manifest = "\n".join(f"image_{i} = <Picture {i}>" for i in range(1, n_images + 1)) or "(no reference images)"
    return (f"Reference manifest:\n{manifest}\n\n"
            f"Source breakdown (preserve shot order/cuts/camera unless overridden):\n{source}\n\n"
            f"Requested total duration: {target:.2f}s.\n\n"
            f"User instruction (apply these changes):\n{instruction.strip() or '(none)'}\n\n"
            f"Write the six-section H3 Ref2VA prompt now.")


def parse_sections(text: str) -> dict:
    t = text.partition("</think>")[2] if "</think>" in text else text
    t = re.sub(r"```[a-zA-Z]*\n?", "", t).strip()
    pos = []
    for name in SECTIONS:
        m = re.search(rf"^{name}\s*:", t, flags=re.MULTILINE | re.IGNORECASE)
        pos.append((name, m.start() if m else -1, m.end() if m else -1))
    out = {}
    present = [p for p in pos if p[1] >= 0]
    present.sort(key=lambda p: p[1])
    for j, (name, start, end) in enumerate(present):
        stop = present[j + 1][1] if j + 1 < len(present) else len(t)
        out[name] = t[end:stop].strip()
    for name in SECTIONS:
        out.setdefault(name, "")
    return out


def _first_frames(ref_images, max_side: int):
    import cv2
    import numpy as np
    frames = []
    for batch in ref_images:
        if batch is None:
            continue
        arr = batch.detach().cpu().float().numpy() if hasattr(batch, "detach") else np.asarray(batch, np.float32)
        if arr.ndim == 3:
            arr = arr[None, ...]
        f = np.clip(arr[0], 0, 1)
        h, w = f.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            f = cv2.resize(f, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        frames.append(f)
    return frames


def rewrite(clip, breakdown_json: str, instruction: str, ref_images, target_seconds: float = 0.0,
            max_length: int = 2048, temperature: float = 0.4, seed: int = 0, max_side: int = 768) -> tuple[str, str, str]:
    import numpy as np
    import torch
    b = parse_breakdown(breakdown_json)
    source, src_dur = source_summary(b)
    target = float(target_seconds) if target_seconds and target_seconds > 0 else src_dur
    frames = _first_frames(ref_images, max_side)
    n_images = len(frames)
    user = build_user(source, n_images, instruction, target)

    images = [torch.from_numpy(np.ascontiguousarray(f[None, ...])) for f in frames]
    tokens = clip.tokenize(user, images=images, system_prompt=H3_SYSTEM,
                           skip_template=False, min_length=1, thinking=False)
    ids = clip.generate(tokens, do_sample=True, max_length=max_length, temperature=temperature,
                        top_k=64, top_p=0.95, min_p=0.0, repetition_penalty=1.05,
                        presence_penalty=0.0, seed=seed, mtp=True)
    text = clip.decode(ids)
    text = text if isinstance(text, str) else str(text)
    sections = parse_sections(text)
    prompt = "\n\n".join(f"{name}:\n{sections[name]}" for name in SECTIONS).strip()

    label_map = "\n".join(f"image_{i} -> <Picture {i}> (wire to native ref_image_{i})" for i in range(1, n_images + 1))
    label_map += "\n(audio: wire directly to native ref_audio_*; not managed here)"
    pj = {"sections": sections, "label_map": {f"image_{i}": f"<Picture {i}>" for i in range(1, n_images + 1)},
          "target_seconds": round(target, 3), "source_shots": b.get("shots") or b.get("timeline") or []}
    return prompt, json.dumps(pj, ensure_ascii=False), label_map
