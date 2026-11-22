"""Offline test: no torch / ComfyUI / GPU required.

Injects a minimal fake `torch`, loads the package via importlib, builds a
synthetic multi-shot video, and drives the FULL node with a fake VLM CLIP whose
tokenize/generate/decode mirror the native TextGenerateV2 contract. Verifies:
shot detection, per-shot VLM records, distinct usable prompts, JSON validity,
robust reply parsing, and example-workflow widget order (AST-checked).

    python tests/test_offline.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import types

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------- fake torch ---
class FT:
    def __init__(self, a):
        self._a = np.asarray(a)

    @property
    def shape(self):
        return self._a.shape

    def __getitem__(self, k):
        return FT(self._a[k])


torch_mod = types.ModuleType("torch")
torch_mod.Tensor = FT
torch_mod.float32 = np.float32
torch_mod.from_numpy = lambda a: FT(a)
torch_mod.zeros = lambda shape, dtype=None: FT(np.zeros(shape, dtype=dtype or np.float32))
sys.modules["torch"] = torch_mod

# ------------------------------------------------------- load the package ---
spec = importlib.util.spec_from_file_location(
    "rea_pkg", os.path.join(PKG, "__init__.py"), submodule_search_locations=[PKG])
pkg = importlib.util.module_from_spec(spec)
sys.modules["rea_pkg"] = pkg
spec.loader.exec_module(pkg)
ReaVideoBreakdown = pkg.NODE_CLASS_MAPPINGS["ReaVideoBreakdown"]
from rea_pkg.rea_clip.vlm import parse_record, LABELS  # noqa: E402


# ------------------------------------------------------------- fake VLM ---
class FakeClip:
    """Mirrors clip.tokenize(videos=...) -> generate -> decode (TextGenerateV2)."""

    def __init__(self):
        self.calls = 0
        self.saw_video = []
        self.saw_system = []

    def tokenize(self, prompt, videos=None, images=None, system_prompt=None, **kw):
        self.saw_video.append(None if videos is None else int(videos[0].shape[0]))
        self.saw_system.append(bool(system_prompt))
        return {"clip_l": [("tok", prompt)]}

    def generate(self, tokens, **kw):
        self.calls += 1
        return [1, 2, 3]

    def decode(self, ids):
        i = self.calls
        made = ["live_action", "ai_video", "code_render"][(i - 1) % 3]
        return json.dumps({
            "description": f"Shot {i}: a distinct scene with subject {i}.",
            "made_of": made, "made_of_reason": "visual evidence",
            "shot_size": "medium", "camera_move": "slow push in",
            "subject_motion": f"subject {i} moves", "lighting": "soft key",
            "style": "cinematic", "video_prompt": f"Prompt {i}: medium shot of subject {i}, slow push in, soft key light, cinematic.",
        })


# --------------------------------------------------------- synthetic video ---
def make_video():
    H, W = 120, 180
    rng = np.random.default_rng(0)
    frames = []
    tints = [(0.9, 0.3, 0.2), (0.2, 0.5, 0.9), (0.3, 0.8, 0.3)]
    for s, tint in enumerate(tints):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        base = np.empty((H, W, 3), np.float32)
        base[..., 0] = tint[0] * (0.4 + 0.6 * yy / H)
        base[..., 1] = tint[1] * (0.4 + 0.6 * xx / W)
        base[..., 2] = tint[2] * (0.5 + 0.5 * (yy + xx) / (H + W))
        for i in range(30):
            f = base.copy()
            cx = 40 + (i * 3 + s * 20) % 100
            f[max(0, 50):70, cx:cx + 20] = (0.95, 0.95, 0.9)
            f += rng.standard_normal((H, W, 1), dtype=np.float32) * (2.0 / 255.0)
            frames.append(np.clip(f, 0, 1))
    return np.stack(frames, 0).astype(np.float32)


def check_widgets():
    src = open(os.path.join(PKG, "nodes.py")).read()
    tree = ast.parse(src)
    specs = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ReaVideoBreakdown":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "INPUT_TYPES":
                    ret = fn.body[-1].value
                    d = {ast.literal_eval(k): v for k, v in zip(ret.keys, ret.values)}
                    specs = []
                    for section in ("required", "optional"):
                        sec = d.get(section)
                        if sec is None:
                            continue
                        for k, v in zip(sec.keys, sec.values):
                            name = ast.literal_eval(k)
                            t = v.elts[0]
                            if isinstance(t, ast.Constant) and t.value in ("FLOAT", "INT", "BOOLEAN", "STRING"):
                                opts = ast.literal_eval(v.elts[1]) if len(v.elts) > 1 else {}
                                specs.append((name, t.value, opts))
                            elif isinstance(t, ast.List):
                                specs.append((name, "combo", {"values": ast.literal_eval(t)}))
    wf = json.load(open(os.path.join(PKG, "example_workflows", "rea_breakdown.json")))
    n5 = [n for n in wf["nodes"] if n["type"] == "ReaVideoBreakdown"][0]
    # ComfyUI serializes a "control_after_generate" combo ("fixed"/"randomize"/…)
    # right after the seed widget; drop it so the real widgets align 1:1.
    control = {"fixed", "increment", "decrement", "randomize"}
    vals = [v for v in n5["widgets_values"]
            if not (isinstance(v, str) and v.lower() in control)]
    assert len(vals) == len(specs), f"widget count {len(vals)} != {len(specs)}"
    for (name, typ, opts), v in zip(specs, vals):
        if typ == "FLOAT":
            ok = isinstance(v, (int, float)) and opts.get("min", -1e18) <= v <= opts.get("max", 1e18)
        elif typ == "INT":
            ok = isinstance(v, int) and opts.get("min", -1e18) <= v <= opts.get("max", 1e18)
        elif typ == "BOOLEAN":
            ok = isinstance(v, bool)
        elif typ == "combo":
            ok = v in opts["values"]
        else:
            ok = True
        assert ok, f"widget {name}={v!r} invalid for {typ} {opts}"
    print(f"  widget order OK ({len(specs)} widgets): {[s[0] for s in specs]}")


def main():
    print("== parse_record robustness ==")
    assert parse_record('{"made_of":"ai_video","video_prompt":"x"}')["made_of"] == "ai_video"
    assert parse_record('```json\n{"made_of":"cg_3d","video_prompt":"y"}\n```')["made_of"] == "cg_3d"
    assert parse_record('Sure! Here is the result:\n{"made_of":"live_action","video_prompt":"z"}\nHope that helps.')["made_of"] == "live_action"
    assert parse_record("totally not json")["video_prompt"] == "totally not json"
    print("  OK")

    print("== example workflow widget order (AST) ==")
    check_widgets()

    print("== full node with fake VLM ==")
    video = make_video()
    clip = FakeClip()
    node = ReaVideoBreakdown()
    video_md, bj, prompts, summary, keyframes, shot_count = node.analyze(
        images=video, fps=30.0, clip=clip, frames_per_shot=6, max_length=512, language="en")
    print("  summary:", summary)
    assert shot_count == 3, f"expected 3 shots, got {shot_count}"
    assert clip.calls == 3, f"VLM called {clip.calls} times"
    assert all(n and n > 0 for n in clip.saw_video), "VLM must receive video frames"
    assert all(clip.saw_system), "system prompt must be passed"
    b = json.loads(bj)
    assert len(b["shots"]) == 3
    made = [s["made_of"] for s in b["shots"]]
    assert all(m in LABELS for m in made), made
    vp = [s["video_prompt"] for s in b["shots"]]
    assert all(p and len(p) > 10 for p in vp), "prompts must be non-empty"
    assert len(set(vp)) == 3, "prompts must differ per shot (no filler)"
    assert prompts.count("[Shot") == 3
    assert "## Verdict" in video_md and "## Timeline" in video_md and "video prompt" in video_md
    assert keyframes.shape[0] == 3
    print("  made_of:", made)
    print("  prompts distinct:", len(set(vp)) == 3)
    print("  OK")

    print("== zh report ==")
    _, _, _, _, _, _ = node.analyze(images=video, fps=30.0, clip=FakeClip(), language="zh")
    print("  OK")

    print("\nALL OFFLINE TESTS PASSED")


if __name__ == "__main__":
    main()
