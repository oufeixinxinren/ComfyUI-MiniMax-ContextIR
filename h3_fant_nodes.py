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

from comfy_api.latest import ComfyExtension, io

from . import h3_media_io as media_io


PICTURES = 9
VIDEOS = 3
VIDEO_AUDIOS = 3
AUDIOS = 3

H3_REFS = io.Custom("H3_REFS")


def _media_names():
    return (
        [f"picture_{i}" for i in range(1, PICTURES + 1)]
        + [f"video_{i}" for i in range(1, VIDEOS + 1)]
        + [f"video_audio_{i}" for i in range(1, VIDEO_AUDIOS + 1)]
        + [f"audio_{i}" for i in range(1, AUDIOS + 1)]
    )


def _partition(items):
    """Split items into the four native groups, preserving list order.

    A video's split audio goes to the paired group (its <Audio N> is
    emitted just before its <Video N>) or to the standalone group,
    depending on the item's audio_mode.
    """
    pictures, videos, video_audios, audios = [], [], [], []
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
            videos.append(item)
            if has_audio and mode == "paired":
                video_audios.append(item)
            else:
                video_audios.append(None)
            if has_audio and mode == "standalone":
                audios.append(item)
        elif kind == "audio":
            audios.append(item)
    return pictures, videos, video_audios, audios


def _pad(seq, n):
    return list(seq or []) + [None] * (n - len(seq or []))


class MiniMaxH3MediaLoaderFantastic(io.ComfyNode):
    """Drag-and-drop / file-picker loader for H3 reference media (Fant replica)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MediaLoaderFantastic",
            display_name="MiniMax H3 Media Loader (Fant)",
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

        pictures, videos, video_audios, audios = _partition(items)

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
            "items": items,
        }
        return io.NodeOutput(bundle)


class MiniMaxH3ReferenceSplitter(io.ComfyNode):
    """Fan a `references` bundle out into individual slots."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceSplitter",
            display_name="MiniMax H3 Reference Splitter",
            description=(
                "Split a MiniMax H3 references bundle into individual picture / video / "
                "video_audio / audio slots for the H3 reference nodes."
            ),
            category="conditioning/video_models",
            inputs=[H3_REFS.Input("references")],
            outputs=(
                [io.Image.Output(display_name=f"picture_{i}") for i in range(1, PICTURES + 1)]
                + [io.Image.Output(display_name=f"video_{i}") for i in range(1, VIDEOS + 1)]
                + [io.Audio.Output(display_name=f"video_audio_{i}") for i in range(1, VIDEO_AUDIOS + 1)]
                + [io.Audio.Output(display_name=f"audio_{i}") for i in range(1, AUDIOS + 1)]
            ),
        )

    @classmethod
    def execute(cls, references=None) -> io.NodeOutput:
        bundle = references or {}
        return io.NodeOutput(
            *(tuple(_pad(bundle.get("pictures"), PICTURES))
              + tuple(_pad(bundle.get("videos"), VIDEOS))
              + tuple(_pad(bundle.get("video_audios"), VIDEO_AUDIOS))
              + tuple(_pad(bundle.get("audios"), AUDIOS)))
        )


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
