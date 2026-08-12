"""MiniMax H3 Unified to Video - fused Image-to-Video + Reference-to-Video.

Fuses ComfyUI's official ``MiniMaxH3ImageToVideo`` and
``MiniMaxH3ReferenceToVideo`` into a single node:

``mode`` options: auto / t2va / fl2va / ref2va.

- t2va  : text only
- fl2va : keyframe mode - auto maps to i2va (first), l2va (last) or
          fl2va (first + last) by the connected images
- ref2va: reference media (images <=9 / videos <=3 with paired
          soundtracks / audios <=3); first/last keyframes may be connected
          at the same time (internally handled as hybrid conditioning)
- auto  : detects the task from connected inputs

``duration`` (seconds) and ``fps`` are converted to the frame count with the
official formula max(5, round(a*fps)) + (5 - (max(5, round(a*fps)) % 17)) % 17.
Prompt media tags are canonicalized (<Image 1>/Image1 -> <Picture 1>,
<Audio1> -> <Audio 1>); references to media that is not connected are reported
as warnings in the report instead of raising.
"""

from __future__ import annotations

import inspect
import json
import math
import re

import torch
import torchaudio

import node_helpers
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
from comfy_api.latest import ComfyExtension, io


CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 1920 * 1088  # H3 practical canvas cap (2.0 MP)
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40

HYBRID_KEYFRAME_SENTINEL = "unified_keyframe_latent"


def _round_half_up(value: float) -> int:
    """Match JavaScript Math.round() semantics used by the official formula."""
    return math.floor(value + 0.5)


def duration_to_length(duration: float, fps: int) -> int:
    """official formula: max(5, round(a*fps)) + (5 - (max(5, round(a*fps)) % 17)) % 17"""
    frame_count = max(5, _round_half_up(duration * fps))
    return frame_count + ((5 - frame_count) % 17)


# ---------------------------------------------------------------- geometry

def align_frame_count(frame_count: int) -> int:
    """Snap up to MiniMax H3's 17n+5 frame grid."""
    frame_count = max(5, int(frame_count))
    return frame_count + ((5 - frame_count) % 17)


def align_frame_count_down(frame_count: int) -> int:
    frame_count = int(frame_count)
    return frame_count - ((frame_count - 5) % 17)


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length: int) -> tuple[int, int, int]:
    frame_count = align_frame_count(length)
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    return (
        max(CANVAS_MULTIPLE, round(nominal_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nominal_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def resize_image(image: torch.Tensor, width: int, height: int, crop: str = "disabled") -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE [B,H,W,C], got {tuple(image.shape)}")
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def empty_av_latent(width: int, height: int, length: int) -> tuple[dict, int]:
    if width % 32 or height % 32:
        raise ValueError("MiniMax H3 width and height must be divisible by 32")
    frame_count, latent_t, audio_t = temporal_shape(length)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros((1, 24, latent_t, height // 16, width // 16), device=device)
    audio = torch.zeros((1, 32, 2, audio_t), device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


# ---------------------------------------------------------------- media helpers

def validate_audio(audio, name: str = "audio") -> tuple[torch.Tensor, int]:
    if not isinstance(audio, dict):
        raise ValueError(f"{name} must be a connected AUDIO value")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or sample_rate is None:
        raise ValueError(f"{name} is missing waveform or sample_rate")
    if waveform.ndim != 3:
        raise ValueError(f"{name} must use [batch,channels,samples], got {tuple(waveform.shape)}")
    if waveform.shape[0] != 1:
        raise ValueError(f"{name} must have batch size 1 for MiniMax H3")
    return waveform, int(sample_rate)


def encode_audio_once(audio_vae, audio) -> torch.Tensor:
    waveform, sample_rate = validate_audio(audio)
    vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != vae_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)
    latent = audio_vae.encode(waveform[:1].movedim(1, -1))
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        raise ValueError("The audio VAE did not return [B,C,stereo,T] latent data")
    return latent


def _encode_reference_audio(audio_vae, audio: dict) -> tuple[torch.Tensor, int]:
    latent = encode_audio_once(audio_vae, audio)
    return latent, int(latent.shape[-1])


def sorted_autogrow_items(values) -> list[tuple[int, object]]:
    if not values:
        return []

    def sort_key(item):
        key = str(item[0])
        try:
            return int(key.rsplit("_", 1)[-1])
        except ValueError:
            return 10_000

    output = []
    for key, value in sorted(values.items(), key=sort_key):
        if value is None:
            continue
        try:
            ordinal = int(str(key).rsplit("_", 1)[-1])
        except ValueError:
            ordinal = len(output) + 1
        output.append((ordinal, value))
    return output


def sorted_autogrow_values(values) -> list:
    return [value for _, value in sorted_autogrow_items(values)]


def _resize_reference_image(image, width: int, height: int, ref_image_size: str):
    h, w = int(image.shape[1]), int(image.shape[2])
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((width * height) / (w * h)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    target_width = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    target_height = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return resize_image(image[:1], target_width, target_height), target_width, target_height


# ---------------------------------------------------------------- prompt tags

MEDIA_TAG_RE = re.compile(
    r"<\s*(Image|Picture|Video|Audio)\s*(\d+)\s*>|"
    r"(?<![\w<])(Image|Picture|Video|Audio)\s*#?\s*(\d+)\b(?!\s*>)",
    re.IGNORECASE,
)
OFFICIAL_TAG_RE = re.compile(r"<(Picture|Video|Audio)\s+(\d+)>", re.IGNORECASE)


def canonicalize_media_tags(prompt: str) -> str:
    def replacement(match: re.Match) -> str:
        media_type = (match.group(1) or match.group(3)).lower()
        ordinal = int(match.group(2) or match.group(4))
        official_type = "Picture" if media_type in {"image", "picture"} else media_type.title()
        return f"<{official_type} {ordinal}>"

    return MEDIA_TAG_RE.sub(replacement, prompt or "")


def prepare_prompt(prompt: str, counts: dict[str, int]) -> tuple[str, list[str]]:
    normalized = canonicalize_media_tags(prompt)
    warnings: list[str] = []
    limits = {
        "picture": int(counts.get("pictures", 0)),
        "video": int(counts.get("videos", 0)),
        "audio": int(counts.get("audios", 0)),
    }
    for match in OFFICIAL_TAG_RE.finditer(normalized):
        media_type, ordinal = match.group(1).lower(), int(match.group(2))
        if ordinal < 1 or ordinal > limits[media_type]:
            warnings.append(
                f"{match.group(0)} is not connected; available {media_type} count is {limits[media_type]}"
            )
    return normalized, warnings


def media_map_json(pictures: list[str], videos: list[str], audios: list[str]) -> str:
    return json.dumps(
        {
            "pictures": {str(index + 1): label for index, label in enumerate(pictures)},
            "videos": {str(index + 1): label for index, label in enumerate(videos)},
            "audios": {str(index + 1): label for index, label in enumerate(audios)},
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------- task resolution

def resolve_task_type(task_type: str, first_frame, last_frame, has_refs: bool) -> str:
    """Resolve the effective MiniMax H3 task for the selected mode.

    Explicit modes shield (ignore) inputs that do not belong to the mode
    instead of raising, and raise only when a required input is missing:

    - t2va  : first/last keyframes and reference media are all ignored
    - i2va  : only first_frame is used (last_frame and refs ignored)
    - l2va  : only last_frame is used (first_frame and refs ignored)
    - fl2va : only keyframes are used (refs ignored)
    - ref2va: keyframes + reference media are all effective

    - t2va has no media requirement
    - i2va / l2va require their keyframe
    - fl2va requires at least one keyframe (maps to i2va / l2va / fl2va)
    - ref2va requires at least one reference media (keyframes optional)
    """
    first = first_frame is not None
    last = last_frame is not None
    requested = (task_type or "auto").lower()
    if requested == "auto":
        if has_refs and (first or last):
            return "hybrid"
        if has_refs:
            return "ref2va"
        if first and last:
            return "fl2va"
        if first:
            return "i2va"
        if last:
            return "l2va"
        return "t2va"

    if requested == "t2va":
        return "t2va"
    if requested == "fl2va":
        if first and last:
            return "fl2va"
        if first:
            return "i2va"
        if last:
            return "l2va"
        raise ValueError(
            "MiniMax H3 mode fl2va requires first_frame and/or last_frame; "
            "connect at least one keyframe."
        )
    if requested == "i2va":
        if not first:
            raise ValueError("MiniMax H3 mode i2va requires first_frame.")
        return "i2va"
    if requested == "l2va":
        if not last:
            raise ValueError("MiniMax H3 mode l2va requires last_frame.")
        return "l2va"
    if requested == "ref2va":
        if not has_refs:
            raise ValueError(
                "MiniMax H3 mode ref2va requires at least one reference media "
                "(ref_images / ref_videos / ref_video_audios / ref_audios)."
            )
        if first or last:
            return "hybrid"
        return "ref2va"
    if requested == "hybrid":
        if not has_refs:
            raise ValueError(
                "MiniMax H3 mode hybrid requires at least one reference media "
                "(ref_images / ref_videos / ref_video_audios / ref_audios)."
            )
        return "hybrid" if (first or last) else "ref2va"
    raise ValueError(f"Unknown MiniMax H3 mode: {task_type}")


def _mode_media_flags(resolved: str) -> tuple[bool, bool, bool]:
    """(use_first, use_last, use_refs) for a resolved task."""
    if resolved == "t2va":
        return False, False, False
    if resolved == "i2va":
        return True, False, False
    if resolved == "l2va":
        return False, True, False
    if resolved == "fl2va":
        return True, True, False
    return True, True, True  # ref2va / hybrid


def assert_hybrid_layout_contract() -> None:
    """Fail loudly if ComfyUI changes keyframe/ref latent assembly."""
    try:
        from comfy.ldm.minimax.model import PackedLayout
        from comfy.model_base import MiniMaxH3 as MiniMaxH3BaseModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "MiniMax H3 Hybrid path requires ComfyUI's PackedLayout/MiniMaxH3 internals; "
            "this build does not provide them."
        ) from exc

    extra_conds_source = inspect.getsource(MiniMaxH3BaseModel.extra_conds)
    required_contract = 'payload["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r]'
    if required_contract not in extra_conds_source:
        raise RuntimeError(
            "This ComfyUI build changed MiniMax H3 ref/keyframe latent assembly; "
            "the unified Hybrid path is disabled until its ordering can be revalidated."
        )

    keyframe = {"resolved_frame_index": 0, "latent": torch.zeros(1)}
    baseline = PackedLayout(1, 2, 2, 2, 1, keyframes=[keyframe], frame_count=5)
    hybrid = PackedLayout(
        1,
        2,
        2,
        2,
        1,
        keyframes=[keyframe],
        refs=[{"kind": HYBRID_KEYFRAME_SENTINEL, "latent": torch.zeros(1)}],
        frame_count=5,
    )
    if baseline.segments != hybrid.segments or baseline.seq_len != hybrid.seq_len:
        raise RuntimeError(
            "This ComfyUI build changed MiniMax H3 PackedLayout reference handling; "
            "the unified exact-keyframe + reference path is disabled to prevent corrupt conditioning."
        )


# ---------------------------------------------------------------- conditioning build

def build_unified_conditioning(
    clip,
    video_vae,
    audio_vae,
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_type: str = "auto",
    ref_image_size: str = "match",
    first_frame=None,
    last_frame=None,
    ref_images=None,
    ref_videos=None,
    ref_video_audios=None,
    ref_audios=None,
):
    if width % 32 or height % 32:
        raise ValueError("MiniMax H3 width and height must be divisible by 32")
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"Requested canvas has {width * height:,} pixels and exceeds the MiniMax H3 "
            f"cap of {MAX_PIXELS:,} pixels (1920x1088); reduce width/height"
        )

    ref_image_values = sorted_autogrow_values(ref_images)
    ref_video_entries = sorted_autogrow_items(ref_videos)
    ref_video_values = [value for _, value in ref_video_entries]
    ref_audio_values = sorted_autogrow_values(ref_audios)
    ref_video_audio_by_ordinal = dict(sorted_autogrow_items(ref_video_audios))

    has_refs_raw = bool(ref_image_values or ref_video_values or ref_audio_values)
    resolved_task = resolve_task_type(task_type, first_frame, last_frame, has_refs_raw)
    use_first, use_last, use_refs = _mode_media_flags(resolved_task)

    ignored_inputs: list[str] = []
    if not use_first and first_frame is not None:
        ignored_inputs.append("first_frame")
    if not use_last and last_frame is not None:
        ignored_inputs.append("last_frame")
    if not use_refs:
        ignored_inputs.extend(
            [f"ref_image_{index}" for index, value in enumerate(ref_image_values, 1) if value is not None]
            + [f"ref_video_{index}" for index, (_, value) in enumerate(ref_video_entries, 1) if value is not None]
            + [f"ref_audio_{index}" for index, value in enumerate(ref_audio_values, 1) if value is not None]
        )

    if use_refs:
        if len(ref_image_values) > 9 or len(ref_video_values) > 3 or len(ref_audio_values) > 3:
            raise ValueError("MiniMax H3 reference limits are 9 pictures, 3 videos, and 3 standalone audios")
        video_ordinals = {ordinal for ordinal, _ in ref_video_entries}
        orphan_soundtracks = sorted(set(ref_video_audio_by_ordinal) - video_ordinals)
        if orphan_soundtracks:
            raise ValueError(
                "Reference-video soundtrack(s) have no same-numbered video: "
                + ", ".join(map(str, orphan_soundtracks))
            )

    latent, frame_count = empty_av_latent(width, height, length)

    keyframes = []
    keyframe_images = []
    picture_labels: list[str] = []
    if use_first and first_frame is not None:
        image = resize_image(first_frame[:1], width, height, "disabled")
        keyframe_images.append(image)
        picture_labels.append("first_frame (exact frame 0)")
        keyframes.append({"resolved_frame_index": 0, "latent": video_vae.encode(image)})
    if use_last and last_frame is not None:
        image = resize_image(last_frame[:1], width, height, "center")
        keyframe_images.append(image)
        picture_labels.append(f"last_frame (exact frame {frame_count - 1})")
        keyframes.append({"resolved_frame_index": frame_count - 1, "latent": video_vae.encode(image)})

    real_ref_items: list[dict] = []
    real_ref_blocks: list[dict] = []
    video_labels: list[str] = []
    audio_labels: list[str] = []

    if use_refs:
        for index, image in enumerate(ref_image_values, 1):
            resized, ref_width, ref_height = _resize_reference_image(image, width, height, ref_image_size)
            encoded = video_vae.encode(resized)
            real_ref_items.append({"type": "image", "data": resized})
            real_ref_blocks.append(
                {
                    "kind": "image",
                    "latent_h": ref_height // 16,
                    "latent_w": ref_width // 16,
                    "latent": encoded,
                }
            )
            picture_labels.append(f"ref_image_{index}")

        for index, (video_ordinal, frames) in enumerate(ref_video_entries, 1):
            if frames.ndim != 4 or frames.shape[0] < 5:
                raise ValueError(f"ref_video_{index} must contain at least 5 IMAGE frames")
            source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
            canvas_width, canvas_height = adapt_canvas(source_width, source_height)
            if source_width * source_height < canvas_width * canvas_height:
                canvas_width = max(CANVAS_MULTIPLE, round(source_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                canvas_height = max(CANVAS_MULTIPLE, round(source_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = resize_image(frames, canvas_width, canvas_height)
            frames = frames[:frame_count]
            aligned_count = align_frame_count_down(int(frames.shape[0]))
            if aligned_count < 5:
                raise ValueError(f"ref_video_{index} is too short after 17n+5 alignment")
            frames = frames[:aligned_count]
            encoded_video = video_vae.encode(frames)

            soundtrack = ref_video_audio_by_ordinal.get(video_ordinal)
            encoded_soundtrack, soundtrack_t = None, 0
            if soundtrack is not None:
                if audio_vae is None:
                    raise ValueError("audio_vae is required when reference videos have soundtracks")
                encoded_soundtrack, soundtrack_t = _encode_reference_audio(audio_vae, soundtrack)
                # soundtrack gets its own <Audio j> label, emitted before <Video k>
                real_ref_items.append({"type": "audio"})
                audio_labels.append(f"ref_video_audio_{video_ordinal}")
            sample_indices = list(range(0, frames.shape[0], FPS // 2))
            real_ref_items.append(
                {
                    "type": "video",
                    "data": frames[sample_indices],
                    "timestamps": [sample_index / FPS for sample_index in sample_indices],
                }
            )
            real_ref_blocks.append(
                {
                    "kind": "video_audio" if soundtrack_t else "video",
                    "latent_t": int(encoded_video.shape[2]),
                    "latent_h": canvas_height // 16,
                    "latent_w": canvas_width // 16,
                    "ref_audio_t": soundtrack_t,
                    "latent": encoded_video,
                    "audio_latent": encoded_soundtrack,
                }
            )
            video_labels.append(f"ref_video_{video_ordinal}")

        for index, audio in enumerate(ref_audio_values, 1):
            if audio_vae is None:
                raise ValueError("audio_vae is required when reference audio is connected")
            encoded_audio, audio_t = _encode_reference_audio(audio_vae, audio)
            real_ref_items.append({"type": "audio"})
            real_ref_blocks.append({"kind": "audio", "ref_audio_t": audio_t, "audio_latent": encoded_audio})
            audio_labels.append(f"ref_audio_{index}")

    counts = {"pictures": len(picture_labels), "videos": len(video_labels), "audios": len(audio_labels)}
    conditioned_prompt, prompt_warnings = prepare_prompt(
        prompt,
        counts,
    )

    if keyframes and real_ref_blocks:
        assert_hybrid_layout_contract()
        ref_items = [{"type": "image", "data": image} for image in keyframe_images] + real_ref_items
        refs = [
            {"kind": HYBRID_KEYFRAME_SENTINEL, "latent": keyframe["latent"]}
            for keyframe in keyframes
        ] + real_ref_blocks
        tokens = clip.tokenize(conditioned_prompt, minimax_ref_items=ref_items)
    elif real_ref_blocks:
        refs = real_ref_blocks
        tokens = clip.tokenize(conditioned_prompt, minimax_ref_items=real_ref_items)
    else:
        refs = []
        tokens = clip.tokenize(conditioned_prompt, images=keyframe_images)

    conditioning = clip.encode_from_tokens_scheduled(tokens)
    values = {}
    if keyframes:
        values.update({"minimax_keyframes": keyframes, "minimax_frame_count": frame_count})
    if refs:
        values["minimax_refs"] = refs
    if values:
        conditioning = node_helpers.conditioning_set_values(conditioning, values)

    media_map = media_map_json(picture_labels, video_labels, audio_labels)
    report_lines = [
        f"task={resolved_task}",
        f"frames={frame_count} ({frame_count / FPS:.3f}s at 24fps)",
        f"pictures={len(picture_labels)}, videos={len(video_labels)}, audios={len(audio_labels)}",
    ]
    if ignored_inputs:
        report_lines.append("ignored_inputs=" + ",".join(ignored_inputs))
    report_lines.extend(f"warning: {warning}" for warning in prompt_warnings)
    return conditioning, latent, conditioned_prompt, media_map, "\n".join(report_lines)


# ---------------------------------------------------------------- the node

class MiniMaxH3UnifiedToVideo(io.ComfyNode):
    """Unified H3 conditioning: keyframes and/or reference media in one node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3UnifiedToVideo",
            display_name="MiniMax H3 Unified to Video",
            description=(
                "Fusion of MiniMax H3 Image to Video and Reference to Video: "
                "t2va / i2va / fl2va / l2va / ref2va / hybrid with auto task detection."
            ),
            category="model/conditioning/minimax/unified",
            inputs=[
                io.Clip.Input("clip", tooltip="Native MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae", optional=True, tooltip="Required only for reference audio / video soundtracks."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Combo.Input("mode", options=["auto", "t2va", "fl2va", "ref2va"], default="auto",
                               tooltip="auto detects the task from connected inputs. Explicit modes ignore extra inputs and raise only when a required input is missing: t2va ignores all media; fl2va requires at least one keyframe (maps to i2va/l2va/fl2va) and ignores references; ref2va requires at least one reference and uses keyframes + references together."),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Float.Input("duration", default=5.0, min=1, max=15, step=0.01,
                               tooltip="Scene duration in seconds. Frame count is computed automatically via max(5, round(duration*fps)) snapped up to the 17k+5 grid."),
                io.Int.Input("fps", default=24, min=1, max=60,
                             tooltip="Frame rate used to convert duration to frames."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="'match' scales each reference image (down only, aspect-preserving) to the generation pixel area; 'max' uses a 2048px short edge for best identity fidelity but is slower."),
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
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="av_latent"),
                io.String.Output(display_name="conditioned_prompt"),
                io.String.Output(display_name="media_map_json"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, clip, video_vae, audio_vae=None, prompt="", width=1344, height=768,
                duration=5.0, fps=24, mode="auto", ref_image_size="match",
                first_frame=None, last_frame=None, ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None) -> io.NodeOutput:
        if not 1 <= float(duration) <= 15:
            raise ValueError("Duration must be between 1 and 15 seconds.")
        if not 1 <= int(fps) <= 60:
            raise ValueError("FPS must be between 1 and 60.")
        length = duration_to_length(float(duration), int(fps))
        task_type = mode or "auto"
        return io.NodeOutput(*build_unified_conditioning(
            clip,
            video_vae,
            audio_vae,
            prompt,
            width,
            height,
            length,
            task_type=task_type,
            ref_image_size=ref_image_size,
            first_frame=first_frame,
            last_frame=last_frame,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
        ))


class MiniMaxH3UnifiedExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3UnifiedToVideo]


async def comfy_entrypoint():
    return MiniMaxH3UnifiedExtension()
