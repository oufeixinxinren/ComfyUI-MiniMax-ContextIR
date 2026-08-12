from comfy_api.latest import ComfyExtension

from .h3_audio_lock import MiniMaxH3AudioLock
from .h3_chat import MiniMaxH3MultimodalChat
from .h3_concat_av import MiniMaxH3ConcatAVLatent
from .h3_context_ir import MiniMaxH3ContextIR
from .h3_fant_nodes import MiniMaxH3MediaLoaderFantastic, MiniMaxH3ReferenceSplitter
from .h3_resolution import MiniMaxH3ResolutionSelector
from .h3_unified import MiniMaxH3UnifiedToVideo


WEB_DIRECTORY = "./web"

# Registers the media-loader upload / probe routes when running inside ComfyUI.
try:
    from . import h3_media_routes  # noqa: F401
except Exception as exc:  # pragma: no cover
    print(f"[MiniMaxH3] media loader routes unavailable: {exc}")


class MiniMaxH3Extension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3UnifiedToVideo,
            MiniMaxH3ConcatAVLatent,
            MiniMaxH3ContextIR,
            MiniMaxH3MultimodalChat,
            MiniMaxH3AudioLock,
            MiniMaxH3MediaLoaderFantastic,
            MiniMaxH3ReferenceSplitter,
            MiniMaxH3ResolutionSelector,
        ]


async def comfy_entrypoint():
    return MiniMaxH3Extension()


__all__ = [
    "MiniMaxH3UnifiedToVideo",
    "MiniMaxH3ConcatAVLatent",
    "MiniMaxH3ContextIR",
    "MiniMaxH3MultimodalChat",
    "MiniMaxH3AudioLock",
    "MiniMaxH3MediaLoaderFantastic",
    "MiniMaxH3ReferenceSplitter",
    "MiniMaxH3ResolutionSelector",
    "MiniMaxH3Extension",
    "comfy_entrypoint",
    "WEB_DIRECTORY",
]
