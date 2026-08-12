"""MiniMax H3 Concat AV Latent.

Replica of ComfyUI-PT_H3ConcatAVLatent (PT_H3ConcatAVLatent) converted to the
plugin's V3 io API. Merges a separate video latent and audio latent into the
joint ``NestedTensor`` latent required by the MiniMax H3 sampler.

- video latent:  [B, 24, T, H/16, W/16]
- audio latent:  [B, 32, 2, T_audio]
"""

from __future__ import annotations

import torch

import comfy.nested_tensor
from comfy_api.latest import ComfyExtension, io


class MiniMaxH3ConcatAVLatent(io.ComfyNode):
    """Merge separate video/audio latents into one H3 joint AV latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ConcatAVLatent",
            display_name="MiniMax H3 Concat AV Latent",
            description=(
                "Merge a separate video latent and audio latent into the joint "
                "NestedTensor latent required by the MiniMax H3 sampler "
                "(video [B,24,T,H/16,W/16] + audio [B,32,2,T_audio])."
            ),
            category="model/conditioning/minimax/unified",
            inputs=[
                io.Latent.Input("video_latent", tooltip="Video latent [B,24,T,H/16,W/16]."),
                io.Latent.Input("audio_latent", tooltip="Audio latent [B,32,2,T_audio]."),
            ],
            outputs=[
                io.Latent.Output("av_latent", tooltip="Joint NestedTensor AV latent for the H3 sampler."),
            ],
        )

    @classmethod
    def execute(cls, video_latent, audio_latent) -> io.NodeOutput:
        if not isinstance(video_latent, dict) or not isinstance(audio_latent, dict):
            raise ValueError("video_latent and audio_latent must be LATENT dicts.")
        video_tensor = video_latent["samples"]
        audio_tensor = audio_latent["samples"]
        if not isinstance(video_tensor, torch.Tensor) or not isinstance(audio_tensor, torch.Tensor):
            raise ValueError("Both video and audio latents must be torch.Tensor.")
        if video_tensor.ndim != 5:
            raise ValueError(f"Video latent expects 5D tensor, got {video_tensor.ndim}D.")
        if audio_tensor.ndim != 4:
            raise ValueError(f"Audio latent expects 4D tensor, got {audio_tensor.ndim}D.")
        if video_tensor.shape[0] != audio_tensor.shape[0]:
            raise ValueError(
                f"Video and audio latents must share the batch size; "
                f"got video {video_tensor.shape[0]} vs audio {audio_tensor.shape[0]}."
            )
        nested = comfy.nested_tensor.NestedTensor([video_tensor, audio_tensor])
        return io.NodeOutput({"samples": nested})


class MiniMaxH3ConcatAVExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3ConcatAVLatent]


async def comfy_entrypoint():
    return MiniMaxH3ConcatAVExtension()


__all__ = [
    "MiniMaxH3ConcatAVLatent",
    "MiniMaxH3ConcatAVExtension",
    "comfy_entrypoint",
]
