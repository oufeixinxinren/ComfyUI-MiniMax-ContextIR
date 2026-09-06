"""MiniMax H3 Audio Lock - keep a supplied audio track exactly in the output.

H3 treats ``ref_audio`` as a *reference*: the model invents a new audio track
that follows the timbre / rhythm / content, and no prompt can force a
bit-identical copy. To get the exact reference audio in the result (e.g. lip
sync to a song or a dialogue track) the source audio must be encoded into the
initial AV latent and held there with a zero denoise mask while only the video
is generated.

This node implements that "audio lock":

- lock  : replace the audio part of the latent with the encoded source and set
          its denoise mask to 0 -> the sampler never touches the audio;
          only the video is generated (best for exact lip sync / MV).
- remix : replace the audio part and denoise it by ``strength`` -> the model
          keeps the beat / phoneme structure but may alter the sound.

The original source AUDIO is passed through as a second output so the final
save step can mux the exact waveform (VAE encode/decode is lossy, so prefer
the passthrough audio for the finished file).
"""

from __future__ import annotations

import torch
import torchaudio

import comfy.nested_tensor
from comfy_api.latest import ComfyExtension, io


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


def nested_av_parts(av_latent: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise ValueError("Expected a MiniMax H3 joint AV LATENT")
    samples = av_latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise ValueError("Expected a nested MiniMax H3 joint video/audio latent")
    parts = tuple(samples.unbind())
    if len(parts) != 2:
        raise ValueError(f"Expected exactly two AV latent parts, got {len(parts)}")
    video, audio = parts
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            "Unexpected MiniMax H3 AV latent layout: "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("MiniMax H3 currently supports batch size 1 only")
    return video, audio


def fit_audio_latent(encoded_audio: torch.Tensor, template_audio: torch.Tensor) -> torch.Tensor:
    if encoded_audio.ndim != 4 or template_audio.ndim != 4:
        raise ValueError("MiniMax H3 audio latents must use [B,C,stereo,T]")
    if encoded_audio.shape[1:-1] != template_audio.shape[1:-1]:
        raise ValueError(
            "Audio VAE latent layout mismatch: "
            f"got {tuple(encoded_audio.shape)}, target {tuple(template_audio.shape)}"
        )
    if encoded_audio.shape[0] != template_audio.shape[0]:
        if encoded_audio.shape[0] == 1:
            encoded_audio = encoded_audio.expand(template_audio.shape[0], -1, -1, -1)
        else:
            raise ValueError("Audio latent batch cannot be matched to the AV latent")
    target_t = template_audio.shape[-1]
    if encoded_audio.shape[-1] > target_t:
        encoded_audio = encoded_audio[..., :target_t]
    elif encoded_audio.shape[-1] < target_t:
        padding = encoded_audio.new_zeros((*encoded_audio.shape[:-1], target_t - encoded_audio.shape[-1]))
        encoded_audio = torch.cat((encoded_audio, padding), dim=-1)
    return encoded_audio.to(device=template_audio.device, dtype=template_audio.dtype)


def split_noise_masks(av_latent: dict, video: torch.Tensor, audio: torch.Tensor):
    masks = av_latent.get("noise_mask")
    if masks is None:
        return None, None
    if getattr(masks, "is_nested", False):
        parts = tuple(masks.unbind())
        if len(parts) == 2:
            return parts
    if isinstance(masks, torch.Tensor):
        # A legacy video-only mask must never be silently discarded.
        return masks, None
    raise ValueError("Unsupported AV noise_mask layout")


def lock_audio(av_latent: dict, audio: dict, audio_vae, mode: str = "lock", strength: float = 0.35):
    """Return (locked_latent, source_audio_passthrough, report)."""
    mode = (mode or "lock").lower()
    if mode not in {"lock", "remix"}:
        raise ValueError(f"Audio lock mode must be lock or remix, got {mode!r}")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    if audio_vae is None:
        raise ValueError("audio_vae is required to lock or remix a reference audio track")

    video, template_audio = nested_av_parts(av_latent)
    encoded = encode_audio_once(audio_vae, audio)
    fitted = fit_audio_latent(encoded, template_audio)

    video_mask, _ = split_noise_masks(av_latent, video, template_audio)
    if video_mask is None:
        video_mask = torch.ones_like(video)
    denoise = 0.0 if mode == "lock" else float(strength)
    audio_mask = torch.full_like(fitted, denoise)

    output = dict(av_latent)
    output["samples"] = comfy.nested_tensor.NestedTensor((video, fitted))
    output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

    duration_s = round(int(template_audio.shape[-1]) / 40.0, 3)
    report = "\n".join(
        [
            f"audio_mode={mode}",
            f"denoise_strength={'0.0 (audio kept exactly)' if mode == 'lock' else str(float(strength))}",
            f"audio_latent_t={int(fitted.shape[-1])} (≈{duration_s}s at 40fps)",
            f"video_mask={'all 1.0' if torch.all(video_mask == 1.0).item() else 'preserved from input'}",
        ]
    )
    return output, audio, report


class MiniMaxH3AudioLock(io.ComfyNode):
    """Lock or remix a source audio track inside an H3 AV latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioLock",
            display_name="MiniMax H3 Audio Lock",
            description=(
                "Replace the audio part of an H3 AV latent with the connected source audio. "
                "lock keeps the source audio untouched (denoise mask 0) so only the video is "
                "generated - the result audio is exactly the reference. remix denoises the "
                "source by strength to keep its structure while letting the model change the "
                "sound. The original AUDIO is passed through for an exact final mux."
            ),
            category="model/conditioning/minimax/unified",
            inputs=[
                io.Latent.Input("av_latent", tooltip="Joint H3 AV latent from a MiniMax H3 conditioning node."),
                io.Audio.Input("audio", tooltip="Source audio to lock into the latent (e.g. the same file connected to ref_audio)."),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE (same one used by the conditioning node)."),
                io.Combo.Input("mode", options=["lock", "remix"], default="lock",
                               tooltip="lock: audio is kept exactly (only video is generated). remix: audio is redrawn by strength."),
                io.Float.Input("strength", default=0.35, min=0.0, max=1.0, step=0.01,
                               tooltip="Denoise strength for remix mode; ignored in lock mode."),
            ],
            outputs=[
                io.Latent.Output(display_name="av_latent"),
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, av_latent, audio, audio_vae, mode="lock", strength=0.35) -> io.NodeOutput:
        return io.NodeOutput(*lock_audio(av_latent, audio, audio_vae, mode=mode, strength=strength))


class MiniMaxH3AudioLockExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3AudioLock]


async def comfy_entrypoint():
    return MiniMaxH3AudioLockExtension()
