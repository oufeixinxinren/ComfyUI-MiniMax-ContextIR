"""HTTP routes backing the MiniMax H3 Media Loader (upload + probe)."""

import os
import re
import time
import json

try:
    from server import PromptServer
    from aiohttp import web
except Exception:  # pragma: no cover - only outside ComfyUI
    PromptServer = None
    web = None

try:
    import folder_paths
except Exception:  # pragma: no cover
    folder_paths = None

from . import h3_media_io as media_io


SUBFOLDER = "minimax_h3"

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


def kind_for(name: str) -> str | None:
    ext = os.path.splitext(name or "")[1].lower()
    if ext in IMAGE_EXT:
        return "picture"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    return None


def _safe(name: str) -> str:
    name = os.path.basename(name or "")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "upload"
    return name[:120]


def _target_dir() -> str:
    base = folder_paths.get_input_directory() if folder_paths else "input"
    path = os.path.join(base, SUBFOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def _unique(directory: str, name: str) -> str:
    stem, ext = os.path.splitext(name)
    candidate = name
    if os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{stem}_{int(time.time() * 1000) % 100000}{ext}"
    return candidate


def _preset_dir() -> str:
    base = None
    if folder_paths is not None:
        for getter in ("get_user_directory", "get_output_directory"):
            fn = getattr(folder_paths, getter, None)
            if callable(fn):
                try:
                    base = fn()
                    break
                except Exception:  # noqa: BLE001
                    continue
    if not base:
        base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "minimax_h3_presets")
    os.makedirs(path, exist_ok=True)
    return path


def _preset_path(name: str):
    safe = re.sub(r"[^A-Za-z0-9 ._-]+", "_", str(name or "")).strip(" ._-")
    if not safe:
        return None, None
    safe = safe[:80]
    return safe, os.path.join(_preset_dir(), safe + ".json")


def _resolve_annotated(annotated: str) -> str:
    if folder_paths is not None:
        try:
            return folder_paths.get_annotated_filepath(annotated)
        except Exception:  # noqa: BLE001
            pass
    return annotated


def _probe(path: str) -> dict:
    """Duration / audio / dimensions. Never raises."""
    info = {"duration": None, "has_audio": False, "width": None, "height": None}
    try:
        import av

        with av.open(path) as container:
            if container.duration:
                info["duration"] = round(container.duration / 1_000_000, 3)
            info["has_audio"] = bool(container.streams.audio)
            if container.streams.video:
                stream = container.streams.video[0]
                info["width"] = getattr(stream, "width", None)
                info["height"] = getattr(stream, "height", None)
    except Exception:  # noqa: BLE001
        pass
    return info


if PromptServer is not None and web is not None and getattr(PromptServer, "instance", None) is not None:

    routes = PromptServer.instance.routes

    @routes.post("/minimax_h3/upload")
    async def upload(request):
        """Accept one media file, store it under input/minimax_h3, return metadata."""
        try:
            reader = await request.multipart()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "expected multipart form data"}, status=400)
        field = await reader.next()
        while field is not None and field.name != "file":
            field = await reader.next()
        if field is None:
            return web.json_response({"error": "no file field in request"}, status=400)

        original = field.filename or "upload"
        kind = kind_for(original)
        if kind is None:
            return web.json_response(
                {"error": f"unsupported file type: {os.path.splitext(original)[1]}"},
                status=400,
            )

        directory = _target_dir()
        name = _unique(directory, _safe(original))
        path = os.path.join(directory, name)
        size = 0
        try:
            with open(path, "wb") as handle:
                while True:
                    chunk = await field.read_chunk()
                    if not chunk:
                        break
                    size += len(chunk)
                    handle.write(chunk)
        except Exception as exc:  # noqa: BLE001
            if os.path.exists(path):
                os.remove(path)
            return web.json_response({"error": f"write failed: {exc}"}, status=500)

        info = _probe(path) if kind in ("video", "audio") else {}
        return web.json_response(
            {
                "file": f"{SUBFOLDER}/{name} [input]",
                "name": name,
                "original": original,
                "kind": kind,
                "size": size,
                "duration": info.get("duration"),
                "has_audio": bool(info.get("has_audio")),
                "width": info.get("width"),
                "height": info.get("height"),
            }
        )

    @routes.get("/minimax_h3/capabilities")
    async def capabilities(request):
        caps = media_io.backends()
        caps["video"] = media_io.can_decode_video()
        return web.json_response(caps)

    @routes.post("/minimax_h3/probe")
    async def probe_route(request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "expected JSON body"}, status=400)
        target = body.get("file")
        if not target:
            return web.json_response({"error": "missing 'file'"}, status=400)
        return web.json_response(media_io.probe(target))

    @routes.get("/minimax_h3/presets")
    async def list_presets(request):
        names = []
        try:
            for fn in os.listdir(_preset_dir()):
                if fn.endswith(".json"):
                    names.append(fn[:-5])
        except Exception:  # noqa: BLE001
            pass
        return web.json_response({"presets": sorted(names, key=str.lower)})

    @routes.post("/minimax_h3/presets/save")
    async def save_preset(request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "expected JSON body"}, status=400)
        name, path = _preset_path(body.get("name"))
        if not path:
            return web.json_response({"error": "give the preset a name"}, status=400)
        items = body.get("items")
        if not isinstance(items, list):
            return web.json_response({"error": "items must be a list"}, status=400)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "items": items}, handle, indent=1)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"save failed: {exc}"}, status=500)
        return web.json_response({"name": name, "count": len(items)})

    @routes.post("/minimax_h3/presets/load")
    async def load_preset(request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "expected JSON body"}, status=400)
        name, path = _preset_path(body.get("name"))
        if not path or not os.path.exists(path):
            return web.json_response({"error": "preset not found"}, status=404)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"unreadable preset: {exc}"}, status=500)
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return web.json_response({"error": "preset has no item list"}, status=500)
        kept, missing = [], []
        for item in items:
            target = item.get("file") if isinstance(item, dict) else None
            if target and os.path.exists(_resolve_annotated(target)):
                kept.append(item)
            elif target:
                missing.append(item.get("name") or target)
        return web.json_response({"name": name, "items": kept, "missing": missing})

    @routes.post("/minimax_h3/presets/delete")
    async def delete_preset(request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "expected JSON body"}, status=400)
        name, path = _preset_path(body.get("name"))
        if not path or not os.path.exists(path):
            return web.json_response({"error": "preset not found"}, status=404)
        try:
            os.remove(path)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"delete failed: {exc}"}, status=500)
        return web.json_response({"deleted": name})
