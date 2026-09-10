"""Faithful replica of the Fantastic MiniMax H3 Media Loader / Reference Splitter.

Port of ComfyUI-Fantastic-MiniMaxH3-PromptBuilder (MIT License,
Copyright (c) 2026 Adudeguyman), converted to the V3 io API so it can be
registered through the plugin's ComfyExtension (mixing legacy
``NODE_CLASS_MAPPINGS`` with a V3 extension would make ComfyUI skip the
extension and hide every other node in this plugin).

The loader is registered under ``MiniMaxH3MediaLoaderFantastic`` so it can
coexist with our native loader; the splitter keeps its original id because our
plugin does not already use it. The frontend ``web/fant_medialoader.js`` is the
original panel (drag-drop, previews, presets, clip budgets).
"""

import json

import torch
import torch.nn.functional as F

import comfy.utils
from comfy_api.latest import ComfyExtension, io

from . import h3_media_io as media_io
from .h3_resolution import RATIOS, _dual_ratio_options, parse_size


PICTURES = 90
VIDEOS = 30
VIDEO_AUDIOS = 30
AUDIOS = 90
STRINGS = 30
# Reference Splitter output port caps (smaller than the loading limits)
SPLIT_PICTURES = 9
SPLIT_VIDEOS = 3
SPLIT_VIDEO_AUDIOS = 3
SPLIT_AUDIOS = 3
SPLIT_STRINGS = 3

H3_REFS = io.Custom("H3_REFS")

# Splitter outputs adapt to whatever is connected (picture slots are IMAGE,
# audio slots AUDIO), so validation can't map a dynamic socket count back to
# static per-slot types.
SLOT_TEMPLATE = io.MatchType.Template(
    "minimax_h3_splitter_slot", allowed_types=[io.Image, io.Audio])


def _media_names():
    return (
        [f"picture_{i}" for i in range(1, PICTURES + 1)]
        + [f"video_{i}" for i in range(1, VIDEOS + 1)]
        + [f"video_audio_{i}" for i in range(1, VIDEO_AUDIOS + 1)]
        + [f"audio_{i}" for i in range(1, AUDIOS + 1)]
    )


def _partition(items):
    """Split items into the native groups, preserving list order.

    A video's split audio goes to the paired group (its <Audio N> is
    emitted just after its <Video N>) or to the standalone group,
    depending on the item's audio_mode.
    """
    pictures, videos, video_audios, audios, standalone_tracks, strings = \
        [], [], [], [], [], []
    for item in items:
        # Items switched off in the loader are kept in the list but never
        # reach the model, so the tag numbering closes up around them.
        if isinstance(item, dict) and item.get("enabled") is False:
            continue
        kind = item.get("kind")
        if kind == "picture":
            pictures.append(item)
        elif kind == "video":
            mode = item.get("audio_mode", "paired")
            has_audio = bool(item.get("has_audio"))
            if has_audio and mode == "standalone":
                # standalone mode sends only the soundtrack — no video slot
                standalone_tracks.append(item)
            else:
                videos.append(item)
                if has_audio and mode == "paired":
                    video_audios.append(item)
                else:
                    video_audios.append(None)
        elif kind == "audio":
            audios.append(item)
        elif kind == "string":
            strings.append(item)
    # standalone video tracks come after the standalone audio items so the
    # <Audio M> ordinals match the splitter's audio_M sockets
    return pictures, videos, video_audios, audios + standalone_tracks, strings


class MiniMaxH3MediaLoaderFantastic(io.ComfyNode):
    """Drag-and-drop / file-picker loader for H3 reference media (Fant replica)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MediaLoaderFantastic",
            display_name="Media Loader",
            description=(
                "Load MiniMax H3 reference media by drag-and-drop or file picker. "
                "Wire 'references' to the MiniMax H3 Reference Splitter (or use the "
                "node's '+ Native-output splitter' button) when you want individual "
                "slots. A video's soundtrack can be split off and paired with it "
                "automatically."
            ),
            category="conditioning/video_models",
            inputs=[
                io.String.Input(
                    "media_state",
                    multiline=False,
                    default="[]",
                    tooltip="JSON list of media items, written by the node's panel.",
                ),
            ],
            outputs=[H3_REFS.Output("references")],
        )

    @classmethod
    def validate_inputs(cls, media_state="[]", **kwargs):
        try:
            items = json.loads(media_state or "[]")
        except Exception:
            return "Media Loader state is corrupt; clear the node and re-add media."
        if not isinstance(items, list):
            return "Media Loader state is corrupt; clear the node and re-add media."
        pics = sum(1 for i in items if i.get("kind") == "picture")
        vids = sum(1 for i in items if i.get("kind") == "video")
        if pics > PICTURES:
            return f"{pics} pictures loaded; H3 accepts {PICTURES}."
        if vids > VIDEOS:
            return f"{vids} videos loaded; H3 accepts {VIDEOS}."
        return True

    @classmethod
    def fingerprint_inputs(cls, media_state="[]"):
        return media_state

    @classmethod
    def execute(cls, media_state="[]") -> io.NodeOutput:
        try:
            items = json.loads(media_state or "[]")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Media Loader state is corrupt; clear the node and re-add media.") from exc
        if not isinstance(items, list):
            raise ValueError("Media Loader state is corrupt; clear the node and re-add media.")

        pictures, videos, video_audios, audios, strings = _partition(items)

        def _trim(item):
            trim = item.get("trim") if isinstance(item, dict) else None
            if not isinstance(trim, dict):
                return None, None

            def num(value):
                try:
                    value = float(value)
                    return value if value > 0 else None
                except (TypeError, ValueError):
                    return None

            return num(trim.get("start")), num(trim.get("end"))

        pic_t = [
            media_io.load_image(i["file"], crop=i.get("crop"))
            for i in pictures[:PICTURES]
        ]
        vid_t = [
            media_io.load_video_frames(
                i["file"],
                start=_trim(i)[0],
                end=_trim(i)[1],
                crop=i.get("crop"),
            )
            for i in videos[:VIDEOS]
        ]
        vaud_t = [
            media_io.extract_audio(i["file"], start=_trim(i)[0], end=_trim(i)[1]) if i else None
            for i in video_audios[:VIDEO_AUDIOS]
        ]
        aud_t = []
        for i in audios[:AUDIOS]:
            if i.get("kind") == "video":
                aud_t.append(media_io.extract_audio(i["file"], start=_trim(i)[0], end=_trim(i)[1]))
            else:
                aud_t.append(media_io.load_audio(i["file"], start=_trim(i)[0], end=_trim(i)[1]))

        bundle = {
            "pictures": pic_t,
            "videos": vid_t,
            "video_audios": vaud_t,
            "audios": aud_t,
            "strings": [s.get("text", "") for s in strings],
            "items": items,
        }
        return io.NodeOutput(bundle)


class MiniMaxH3ReferenceSplitter(io.ComfyNode):
    """Fan a `references` bundle out into fixed 9/3/3/3/3 media slots.

    Empty slots are padded with None; execute also returns the two
    resolution pairs and the duration from its own widgets.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceSplitter",
            display_name="Reference Splitter",
            description=(
                "Split a MiniMax H3 references bundle into individual picture / video / "
                "video_audio / audio / string slots for the H3 reference nodes."
            ),
            category="conditioning/video_models",
            inputs=[
                H3_REFS.Input("references"),
                io.DynamicCombo.Input(
                    "aspect_ratio",
                    options=_dual_ratio_options(),
                    tooltip="Aspect ratio for both size windows; each follows the selected ratio.",
                ),
                io.Float.Input(
                    "duration",
                    default=5.0,
                    min=1.0,
                    max=3600.0,
                    tooltip="Duration of the video to generate (seconds).",
                ),
                io.Int.Input(
                    "short_edge_max",
                    default=0,
                    min=0,
                    max=8192,
                    step=8,
                    tooltip="Short edge max pixels; 0 = no scaling. "
                            "Pictures are downscaled so their short edge fits, then aligned.",
                ),
                io.Int.Input(
                    "align_to",
                    default=16,
                    min=1,
                    max=128,
                    step=1,
                    tooltip="Pixel alignment for scaled dimensions.",
                ),
            ],
            outputs=(
                [io.MatchType.Output(SLOT_TEMPLATE, display_name=f"picture_{i}")
                 for i in range(1, SPLIT_PICTURES + 1)]
                + [io.MatchType.Output(SLOT_TEMPLATE, display_name=f"video_{i}")
                   for i in range(1, SPLIT_VIDEOS + 1)]
                + [io.MatchType.Output(SLOT_TEMPLATE, display_name=f"video_audio_{i}")
                   for i in range(1, SPLIT_VIDEO_AUDIOS + 1)]
                + [io.MatchType.Output(SLOT_TEMPLATE, display_name=f"audio_{i}")
                   for i in range(1, SPLIT_AUDIOS + 1)]
                + [io.String.Output(display_name=f"string_{i}")
                   for i in range(1, SPLIT_STRINGS + 1)]
                + [io.Int.Output("width_1", tooltip="First size: width (multiple of 32)."),
                   io.Int.Output("height_1", tooltip="First size: height (multiple of 32)."),
                   io.Int.Output("width_2", tooltip="Second size: width (multiple of 32)."),
                   io.Int.Output("height_2", tooltip="Second size: height (multiple of 32)."),
                   io.Float.Output("duration", tooltip="Requested video duration (seconds).")]
            ),
        )

    @classmethod
    def execute(cls, references=None, aspect_ratio=None, duration: float = 5.0,
                short_edge_max: int = 0, align_to: int = 16) -> io.NodeOutput:
        bundle = references or {}
        # positional padding like the original Fant splitter: empty slots stay
        # None so downstream nodes can tell unused slots apart
        def _pad(seq, n):
            seq = list(seq or [])[:n]
            return seq + [None] * (n - len(seq))

        pic_t = _pad(bundle.get("pictures"), SPLIT_PICTURES)
        vid_t = _pad(bundle.get("videos"), SPLIT_VIDEOS)
        vaud_t = _pad(bundle.get("video_audios"), SPLIT_VIDEO_AUDIOS)
        aud_t = _pad(bundle.get("audios"), SPLIT_AUDIOS)
        str_t = _pad([s if isinstance(s, str) else "" for s in bundle.get("strings") or []],
                     SPLIT_STRINGS)

        # global image scaling: downscale pictures so short edge <= short_edge_max,
        # both dimensions aligned to align_to; only downscales (never upscales)
        if short_edge_max > 0:
            scaled = []
            for img in pic_t:
                if img is None:
                    scaled.append(None)
                    continue
                _, h, w, _ = img.shape
                short = min(h, w)
                if short <= short_edge_max:
                    scaled.append(img)
                    continue
                scale = short_edge_max / short
                new_h = max(align_to, round(h * scale / align_to) * align_to)
                new_w = max(align_to, round(w * scale / align_to) * align_to)
                t = img[..., :3].permute(0, 3, 1, 2).float()
                t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
                scaled.append(t.permute(0, 2, 3, 1).contiguous().to(img.dtype))
            pic_t = scaled

        ratio_label = (aspect_ratio or {}).get("aspect_ratio") or ""
        ratio = ratio_label.split(" ")[0]
        if ratio not in RATIOS:
            raise ValueError(f"Unknown MiniMax H3 aspect ratio: {ratio_label!r}")
        width_1, height_1, _ = parse_size((aspect_ratio or {}).get("size1"), ratio)
        width_2, height_2, _ = parse_size((aspect_ratio or {}).get("size2"), ratio)
        return io.NodeOutput(
            *(pic_t + vid_t + vaud_t + aud_t + str_t
              + [width_1, height_1, width_2, height_2, float(duration)]))


class MiniMaxH3FantNodesExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3MediaLoaderFantastic, MiniMaxH3ReferenceSplitter]


async def comfy_entrypoint():
    return MiniMaxH3FantNodesExtension()


__all__ = [
    "MiniMaxH3MediaLoaderFantastic",
    "MiniMaxH3ReferenceSplitter",
    "MiniMaxH3FantNodesExtension",
    "comfy_entrypoint",
]
