"""V3 node: VideoPrompt Mod With References.

Uses the same io.Schema / io.Autogrow style as the native MiniMaxH3ReferenceToVideo
node so reference images are dynamic. Registered into NODE_CLASS_MAPPINGS by
nodes.py (ComfyUI handles V3 classes in that dict). Only needs the native VLM CLIP.
"""
from __future__ import annotations

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .rea_clip import h3


class VideoPromptModWithReferences(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoPromptModWithReferences",
            display_name="VideoPrompt Mod With References",
            category="RE-VIDEO/breakdown",
            description=("Rewrite a ReaVideoBreakdown breakdown_json into a MiniMax H3 Ref2VA prompt using your native VLM CLIP "
                         "and reference images, applying a user instruction. Output feeds MiniMaxH3ReferenceToVideo.prompt. "
                         "Wire the SAME images (same order) to that node's ref_image_*. Audio is wired there directly (not here)."),
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("breakdown_json", multiline=True, dynamic_prompts=True,
                                tooltip="breakdown_json output of Video Breakdown (Rea · native VLM)"),
                io.String.Input("user_instruction", multiline=True, dynamic_prompts=True, default="",
                                tooltip="e.g. use image_1 as the woman, replace character-1, make her speak"),
                io.Autogrow.Input("ref_images", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("image", tooltip="Reference image; image_N = <Picture N>"),
                                      prefix="image_", min=0, max=9)),
                io.Float.Input("target_seconds", default=0.0, min=0.0, max=120.0, step=0.1,
                               tooltip="total duration the rewrite must fit; 0 = source duration"),
                io.Int.Input("max_length", default=2048, min=256, max=8192, step=64),
                io.Float.Input("temperature", default=0.4, min=0.0, max=2.0, step=0.05),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("max_side", default=768, min=224, max=1536, step=32),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="prompt_json"),
                io.String.Output(display_name="label_map"),
            ],
        )

    @classmethod
    def execute(cls, clip, breakdown_json, user_instruction, ref_images=None, target_seconds=0.0,
                max_length=2048, temperature=0.4, seed=0, max_side=768) -> io.NodeOutput:
        refs = []
        if ref_images:
            for key in sorted(ref_images, key=lambda k: int(k.rsplit("_", 1)[-1])):
                if ref_images[key] is not None:
                    refs.append(ref_images[key])
        prompt, pj, label_map = h3.rewrite(
            clip, breakdown_json, user_instruction, refs, target_seconds=target_seconds,
            max_length=max_length, temperature=temperature, seed=seed, max_side=max_side)
        return io.NodeOutput(prompt, pj, label_map)


class ReaVideoV3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VideoPromptModWithReferences]
