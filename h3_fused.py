"""MiniMax H3 Unified + Context IR - fused single node.

One node that:
1. (optional) sends the input prompt + media to the H3-Context-IR API and
   receives an enhanced H3 prompt;
2. always builds local MiniMax H3 conditioning (t2va/i2va/r2va) from the
   (possibly enhanced) prompt and the connected media.

``mode`` uses the original Context-IR generation types:
- t2va: text only
- i2va: text + first/last keyframes (locally mapped to fl2va/i2va/l2va/t2va)
- r2va: text + reference images/videos/audios; first/last keyframes are
  allowed as extra anchors (locally ref2va or hybrid when keyframes exist)
"""

from __future__ import annotations

import json
import math
import time

from comfy_api.latest import ComfyExtension, io

from .h3_context_ir import (
    BASE_URLS,
    ENDPOINTS,
    _build_content_by_mode,
    _require_api_key,
    _task_id_from_response,
    _validate_content,
    create_task,
    query_task,
)
from .h3_unified import build_unified_conditioning


def _round_half_up(value: float) -> int:
    """Match the official formula's round() semantics (JS Math.round)."""
    return math.floor(value + 0.5)


def _orphan_soundtracks(ref_videos, ref_video_audios) -> list[int]:
    """Soundtrack ordinals that have no same-numbered reference video."""
    video_ordinals = set()
    for key, value in (ref_videos or {}).items():
        if value is None:
            continue
        try:
            video_ordinals.add(int(str(key).rsplit("_", 1)[-1]))
        except ValueError:
            pass
    orphan = []
    for key, value in (ref_video_audios or {}).items():
        if value is None:
            continue
        try:
            ordinal = int(str(key).rsplit("_", 1)[-1])
        except ValueError:
            continue
        if ordinal not in video_ordinals:
            orphan.append(ordinal)
    return sorted(orphan)


class MiniMaxH3UnifiedContextIR(io.ComfyNode):
    """Fused local conditioning + optional Context-IR prompt enhancement."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3UnifiedContextIR",
            display_name="MiniMax H3 Unified + Context IR",
            description=(
                "Local H3 conditioning (t2va/i2va/r2va) with an optional first pass "
                "through the H3-Context-IR API to enhance the prompt. Toggle "
                "use_context_ir off for pure local conditioning."
            ),
            category="model/conditioning/minimax/unified",
            inputs=[
                io.Clip.Input("clip", tooltip="Native MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="Required only for reference audio / video soundtracks."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Combo.Input("mode", options=["t2va", "i2va", "r2va"], default="t2va",
                               tooltip="t2va: text only; i2va: first/last keyframes; r2va: reference media (first/last keyframes allowed as extra anchors)."),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Float.Input("duration", default=5.0, min=1, max=15, step=0.01,
                               tooltip="Scene duration in seconds. Frame count is computed automatically via max(5, round(duration*fps)) snapped up to the 17k+5 grid."),
                io.Int.Input("fps", default=24, min=1, max=60,
                             tooltip="Frame rate used to convert duration to frames."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="Local reference image sizing: match (generation pixel area) or max (2048 short edge)."),
                io.Boolean.Input("use_context_ir", default=True,
                                 tooltip="On: enhance the prompt via H3-Context-IR API first. Off: local conditioning only (no API call)."),
                io.Combo.Input("ratio", options=["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                               default="16:9", tooltip="Used by the Context-IR API request (t2va cannot be adaptive)."),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image (downscaled to 2048 short edge if larger, never upscaled)"),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (2-15s)"),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        prefix="ref_audio_", min=0, max=3)),
                io.String.Input("api_key", default="", tooltip="MiniMax API key (or set MINIMAX_API_KEY env var)."),
                io.Combo.Input("base_url", options=["global", "cn"], default="global"),
                io.String.Input("callback_url", default="", tooltip="Optional task completion callback URL."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="Latent"),
                io.String.Output(display_name="enhanced_prompt"),
                io.String.Output(display_name="media_map_json"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, clip, video_vae, audio_vae=None, use_context_ir=True, prompt="", mode="t2va",
                width=1344, height=768, duration=5.0, fps=24, ratio="16:9", ref_image_size="match",
                first_frame=None, last_frame=None, ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None, api_key="", base_url="global",
                callback_url="") -> io.NodeOutput:
        mode = (mode or "t2va").lower()
        if mode not in {"t2va", "i2va", "r2va"}:
            raise ValueError(f"Unknown MiniMax H3 mode: {mode}")

        refs_present = any(
            value is not None
            for value in (
                list((ref_images or {}).values())
                + list((ref_videos or {}).values())
                + list((ref_video_audios or {}).values())
                + list((ref_audios or {}).values())
            )
        )
        kf_present = first_frame is not None or last_frame is not None

        if mode == "t2va" and (kf_present or refs_present):
            raise ValueError("T2VA accepts text only; switch to I2VA (keyframes) or R2VA (references).")
        if mode == "i2va" and refs_present:
            raise ValueError("I2VA accepts first/last frame images only; switch to R2VA for reference media.")
        if mode == "r2va" and not refs_present:
            raise ValueError(
                "R2VA requires at least one reference media input; "
                "first/last frames alone should use I2VA."
            )
        orphan = _orphan_soundtracks(ref_videos, ref_video_audios)
        if orphan:
            raise ValueError(
                "Reference-video soundtrack(s) have no same-numbered video: "
                + ", ".join(map(str, orphan))
            )

        # official frame-count formula: max(5, round(a*fps)) + (5 - (max(5, round(a*fps)) % 17)) % 17
        frame_count = max(5, _round_half_up(duration * fps))
        length = frame_count + ((5 - frame_count) % 17)
        if length > 3600:
            raise ValueError(
                f"Computed frame count {length} exceeds 3600; reduce duration or fps."
            )

        api_report_lines = []
        enhanced_prompt = prompt or ""
        if use_context_ir:
            key = _require_api_key(api_key)
            if mode == "t2va" and ratio == "adaptive":
                raise ValueError("T2VA Context-IR requires a concrete ratio (not adaptive).")
            content = _build_content_by_mode(
                mode,
                prompt or "",
                first_frame=first_frame,
                last_frame=last_frame,
                ref_images=ref_images,
                ref_videos=ref_videos,
                ref_video_audios=ref_video_audios,
                ref_audios=ref_audios,
            )
            _validate_content(content)
            api_duration = int(min(15, max(1, _round_half_up(duration))))
            payload = {
                "model": "MiniMax-H3",
                "content": content,
                "duration": api_duration,
                "ratio": ratio,
            }
            if callback_url.strip():
                payload["callback_url"] = callback_url.strip()
            resp = create_task(BASE_URLS[base_url], key, ENDPOINTS["context_ir"], payload)
            task_id = _task_id_from_response(resp)

            deadline = time.time() + 180
            status = "unknown"
            while True:
                resp = query_task(BASE_URLS[base_url], key, task_id)
                task = resp.get("task", resp)
                status = task.get("status", "unknown")
                if status in ("succeeded", "failed", "cancelled") or time.time() >= deadline:
                    break
                time.sleep(5)
            if status != "succeeded":
                raise RuntimeError(
                    "H3-Context-IR failed: "
                    f"status={status} {json.dumps(resp, ensure_ascii=False)[:500]}"
                )
            enhanced_prompt = task.get("content", {}).get("prompt", "") or ""
            api_report_lines = [f"context_ir=on", f"api_duration={api_duration}s", f"api_ratio={ratio}"]
        else:
            api_report_lines = ["context_ir=off"]

        if mode == "t2va":
            local_task = "t2va"
        elif mode == "i2va":
            # same automatic distinction as the official Image to Video node:
            # no frames -> t2va, first -> i2va, first+last -> fl2va, last -> l2va
            local_task = "auto"
        else:
            # refs only -> ref2va; keyframes + refs -> hybrid (exact keyframe
            # anchors plus reference media)
            local_task = "auto"

        conditioning, latent, _conditioned_prompt, media_map, unified_report = build_unified_conditioning(
            clip,
            video_vae,
            audio_vae,
            enhanced_prompt,
            width,
            height,
            length,
            task_type=local_task,
            ref_image_size=ref_image_size,
            first_frame=first_frame,
            last_frame=last_frame,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
        )
        report = "\n".join(api_report_lines + [f"mode={mode}"] + unified_report.splitlines())
        return io.NodeOutput(conditioning, latent, enhanced_prompt, media_map, report)


class MiniMaxH3UnifiedContextIRExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3UnifiedContextIR]


async def comfy_entrypoint():
    return MiniMaxH3UnifiedContextIRExtension()
