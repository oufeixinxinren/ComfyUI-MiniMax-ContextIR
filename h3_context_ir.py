"""MiniMax H3 Context-IR node (io schema, autogrow media inputs).

Modes:
  - t2va: text only
  - i2va: text + first_frame / last_frame
  - r2va: text + reference images (<=9) / videos (<=3, with paired soundtracks) / audio (<=3)

Output: the enhanced H3 prompt as a string.
"""

from __future__ import annotations

import base64
import io as _io
import json
import mimetypes
import os
import tempfile
import time
import wave as wave_module

import torch

from comfy_api.latest import ComfyExtension, io

from .minimax_api import BASE_URLS, ENDPOINTS, create_task, query_task

_MAX_VIDEO_DATA_URL_BYTES = 20 * 1024 * 1024  # keep Base64 payload well under the 64 MB request body limit


# ---------------------------------------------------------------- media helpers

def _image_to_data_url(image: torch.Tensor) -> str:
    from torchvision.transforms import ToPILImage

    img = image.detach().cpu().float()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
        img = img.permute(2, 0, 1)  # HWC -> CHW for torchvision
    pil = ToPILImage()(img)
    buf = _io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _waveform_to_wav_data_url(waveform: torch.Tensor, sample_rate: int) -> str:
    wf = waveform.detach().cpu().float()
    if wf.ndim == 2:
        wf = wf.unsqueeze(0)
    wf = wf[0]  # [channels, samples]
    if wf.ndim != 2:
        raise ValueError("Unsupported waveform shape for WAV encoding.")
    channels, _samples = wf.shape
    pcm = (wf.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)
    if channels > 1:
        pcm = pcm.transpose(0, 1).contiguous()  # interleave [L, C] for the wave module
    buf = _io.BytesIO()
    with wave_module.open(buf, "wb") as wav:
        wav.setnchannels(int(channels))
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.numpy().tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _audio_to_data_url(audio: dict) -> str:
    path = audio.get("path") or ""
    if path and os.path.isfile(path):
        mime = mimetypes.guess_type(path)[0] or "audio/wav"
        with open(path, "rb") as fh:
            payload = fh.read()
        return "data:" + mime + ";base64," + base64.b64encode(payload).decode("ascii")
    waveform = audio.get("waveform")
    if waveform is None:
        raise ValueError("AUDIO input has neither a source file path nor a waveform.")
    return _waveform_to_wav_data_url(waveform, int(audio.get("sample_rate", 44100)))


def _frames_to_mp4_bytes(frames: torch.Tensor, fps: int = 24) -> bytes:
    """Encode an IMAGE batch of frames to an mp4 in memory.

    Tries PyAV, OpenCV, then imageio-ffmpeg; raises if none is available.
    """
    imgs = frames.detach().cpu().float()
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
    n, h, w, c = imgs.shape
    if n < 5:
        raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps).")
    if c != 3:
        raise ValueError("Reference video frames must be RGB (3 channels).")
    rgb = (imgs.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()

    try:
        import av

        out = _io.BytesIO()
        with av.open(out, "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            for frame in rgb:
                vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(vf):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return out.getvalue()
    except Exception:  # noqa: BLE001 - fall back to the next encoder
        pass

    try:
        import cv2

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        try:
            for frame in rgb:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        with open(tmp.name, "rb") as fh:
            data = fh.read()
        os.unlink(tmp.name)
        return data
    except Exception:  # noqa: BLE001 - fall back to the next encoder
        pass

    try:
        import imageio.v2 as imageio

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        imageio.mimsave(tmp.name, rgb, fps=fps, codec="libx264")
        with open(tmp.name, "rb") as fh:
            data = fh.read()
        os.unlink(tmp.name)
        return data
    except Exception:  # noqa: BLE001 - all encoders failed
        pass

    raise ValueError(
        "No video encoder found (av / cv2 / imageio-ffmpeg). Reference videos cannot be sent."
    )


def _validate_content(content: list[dict]) -> None:
    roles = [item.get("role") for item in content if item.get("type") != "text"]
    has_keyframe = any(r in ("first_frame", "last_frame") for r in roles)
    has_reference = any((r or "").startswith("reference_") for r in roles)
    if has_keyframe and has_reference:
        raise ValueError(
            "MiniMax H3 API: i2va/fl2va (first_frame/last_frame) and r2va "
            "(reference_*) are mutually exclusive and cannot be mixed."
        )
    if sum(1 for r in roles if r == "reference_image") > 9:
        raise ValueError("MiniMax H3 API: too many reference images (max 9).")
    if sum(1 for r in roles if r == "reference_video") > 3:
        raise ValueError("MiniMax H3 API: too many reference videos (max 3).")
    if sum(1 for r in roles if r == "reference_audio") > 3:
        raise ValueError("MiniMax H3 API: too many reference audios (max 3).")
    if sum(1 for r in roles if r == "first_frame") > 1:
        raise ValueError("MiniMax H3 API: at most one first_frame image.")
    if sum(1 for r in roles if r == "last_frame") > 1:
        raise ValueError("MiniMax H3 API: at most one last_frame image.")


# ---------------------------------------------------------------- API helpers

def _require_api_key(widget_key: str) -> str:
    key = (widget_key or "").strip()
    if not key:
        raise ValueError(
            "MiniMax API key missing: set the MINIMAX_API_KEY environment variable "
            "or pass api_key into the node."
        )
    return key


def _task_id_from_response(resp: dict) -> str:
    task_id = resp.get("task_id")
    if not task_id and isinstance(resp.get("task"), dict):
        task_id = resp["task"].get("id")
    return str(task_id or "")


# ---------------------------------------------------------------- content build

def _build_content_by_mode(
    mode: str,
    text: str,
    first_frame=None,
    last_frame=None,
    ref_images: dict | None = None,
    ref_videos: dict | None = None,
    ref_video_audios: dict | None = None,
    ref_audios: dict | None = None,
) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": text}]
    ref_present = any(
        v is not None
        for v in (
            list((ref_images or {}).values())
            + list((ref_videos or {}).values())
            + list((ref_video_audios or {}).values())
            + list((ref_audios or {}).values())
        )
    )
    kf_present = first_frame is not None or last_frame is not None
    if ref_images and len(ref_images) > 9:
        raise ValueError("MiniMax H3 API: too many reference images (max 9).")
    if ref_videos and len(ref_videos) > 3:
        raise ValueError("MiniMax H3 API: too many reference videos (max 3).")
    if ref_video_audios and len(ref_video_audios) > 3:
        raise ValueError("MiniMax H3 API: too many reference video audios (max 3).")
    if ref_audios and len(ref_audios) > 3:
        raise ValueError("MiniMax H3 API: too many reference audios (max 3).")

    if mode == "t2va":
        if ref_present or kf_present:
            raise ValueError(
                "t2va mode accepts text only; switch to i2va (keyframes) or r2va (references)."
            )
    elif mode == "i2va":
        if ref_present:
            raise ValueError(
                "i2va mode accepts first/last frame images only; switch to r2va for reference media."
            )
        if first_frame is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_url(first_frame)},
                    "role": "first_frame",
                }
            )
        if last_frame is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_url(last_frame)},
                    "role": "last_frame",
                }
            )
    elif mode == "r2va":
        if kf_present:
            raise ValueError(
                "r2va mode does not accept first/last frames; use i2va for keyframes."
            )
        for i in range(9):
            img = (ref_images or {}).get(f"ref_image_{i}")
            if img is not None:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(img)},
                        "role": "reference_image",
                    }
                )
        for i in range(3):
            frames = (ref_videos or {}).get(f"ref_video_{i}")
            if frames is not None:
                video_bytes = _frames_to_mp4_bytes(frames)
                if len(video_bytes) > _MAX_VIDEO_DATA_URL_BYTES:
                    raise ValueError(
                        f"Encoded reference video {i} exceeds the 20 MB data-URL budget; "
                        "shorten the clip or use reference images/audio only."
                    )
                content.append(
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "data:video/mp4;base64,"
                            + base64.b64encode(video_bytes).decode("ascii")
                        },
                        "role": "reference_video",
                    }
                )
            audio = (ref_video_audios or {}).get(f"ref_video_audio_{i}")
            if audio is not None:
                content.append(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": _audio_to_data_url(audio)},
                        "role": "reference_audio",
                    }
                )
        for i in range(3):
            audio = (ref_audios or {}).get(f"ref_audio_{i}")
            if audio is not None:
                content.append(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": _audio_to_data_url(audio)},
                        "role": "reference_audio",
                    }
                )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return content


# ---------------------------------------------------------------- the node

class MiniMaxH3ContextIR(io.ComfyNode):
    """H3-Context-IR: multimodal context -> enhanced video prompt string."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ContextIR",
            display_name="MiniMax H3 Context IR",
            category="MiniMax ContextIR",
            description=(
                "H3-Context-IR: turn a plain prompt (+ optional media) into the "
                "structured H3 prompt. Modes: t2va / i2va / r2va."
            ),
            inputs=[
                io.Combo.Input(
                    "mode", options=["t2va", "i2va", "r2va"], default="t2va"
                ),
                io.String.Input("text", multiline=True, default=""),
                io.Int.Input("duration", default=5, min=4, max=15),
                io.Combo.Input(
                    "ratio",
                    options=["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                    default="16:9",
                ),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", optional=True),
                        prefix="ref_image_",
                        min=4,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", optional=True),
                        prefix="ref_video_",
                        min=1,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio", optional=True),
                        prefix="ref_video_audio_",
                        min=1,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", optional=True),
                        prefix="ref_audio_",
                        min=1,
                        max=3,
                    ),
                ),
                io.String.Input("api_key", default=""),
                io.Combo.Input("base_url", options=["global", "cn"], default="global"),
                io.String.Input("callback_url", default=""),
            ],
            outputs=[io.String.Output(display_name="enhanced_prompt")],
        )

    @classmethod
    def execute(
        cls,
        mode,
        text,
        duration,
        ratio,
        first_frame=None,
        last_frame=None,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
        api_key="",
        base_url="global",
        callback_url="",
    ) -> io.NodeOutput:
        key = _require_api_key(api_key)
        content = _build_content_by_mode(
            mode,
            text,
            first_frame=first_frame,
            last_frame=last_frame,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
        )
        _validate_content(content)
        if mode == "t2va" and ratio == "adaptive":
            raise ValueError("t2va (text-only) Context-IR requires a concrete ratio.")

        payload: dict = {
            "model": "MiniMax-H3",
            "content": content,
            "duration": int(duration),
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
        enhanced = task.get("content", {}).get("prompt", "") or ""
        return io.NodeOutput(enhanced)


class MiniMaxH3ContextIRExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3ContextIR]


async def comfy_entrypoint():
    return MiniMaxH3ContextIRExtension()
