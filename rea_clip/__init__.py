# rea_clip - shot-by-shot video reverse engineering for ComfyUI.
# Vision understanding uses the native ComfyUI VLM text encoder (clip.tokenize ->
# clip.generate -> clip.decode), the same path as the core "Generate Text" node.
# No clip_vision tower, no external API.
