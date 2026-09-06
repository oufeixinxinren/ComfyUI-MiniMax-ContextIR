"""
H3 Tiled Sampler (Fixed & Reference Cropping - 支持视频+音频联合 latent & R2V 参考条件同步裁剪)
作者：基于 yichengup/ComfyUI-YCNodes-MiniMax-H3 修复

复刻到 ComfyUI-MiniMax-ContextIR（V3 io API 版）。
原始实现基于 yichengup/ComfyUI-YCNodes-MiniMax-H3 的修复版，依据
ComfyUI-MiniMax-H3-Block-based-sampler（MIT License, Copyright (c) 2026 yc）复刻。

修复记录：
  1. 修复原版分块采样丢弃 audio 分量导致 IndexError 问题。
  2. 补全 Reference to Video (R2V) 空间同步裁切：在空间分块采样时，同步对 cond 中的 
     minimax_refs 进行 H/W 轴裁切，解决 R2V 接入时每个 Tile 独立渲染整张参考图导致的拼接重复问题。

原理：
  H3 是视频+音频联合 DiT。latent 为 NestedTensor(video [B,24,T,H/16,W/16],
  audio [B,32,2,T40])。采样器 guider.sample() 内部已完整支持 NestedTensor。
  分块采样把完整的 NestedTensor(video, audio) 传给 guider.sample()，
  只在 video 的空间维度 (H/W) 切片，audio 完整透传（不参与空间分块）。
"""

import torch
import torch.nn.functional as F
import comfy.sample
import comfy.utils
import comfy.samplers
import comfy.model_management
from comfy.nested_tensor import NestedTensor
from comfy.k_diffusion.sampling import to_d
from comfy.utils import model_trange
import latent_preview

from comfy_api.latest import io


# H3 视频 VAE 训练约束
H3_VIDEO_FRAMES = 17       # 输入视频帧数硬约束
H3_LATENT_CHANNELS = 24    # H3 video latent 通道数
H3_LATENT_TIME = 5         # 17 帧 → 5 latent frame (vae_ratio_t=4 + token_drop=3)


# ─────────────────────────────────────────────────────────────────────────────
# H3 视频/音频提取与重建
# ─────────────────────────────────────────────────────────────────────────────

def _h3_extract(samples, debug=False):
    type_name = type(samples).__name__

    # --- NestedTensor ---
    if hasattr(samples, "is_nested") and samples.is_nested:
        try:
            parts = list(samples.unbind())
            video = None
            audio = None
            for p in parts:
                if isinstance(p, torch.Tensor):
                    if video is None:
                        video = p
                    else:
                        audio = p
            if video is not None:
                if debug:
                    print(f"  · [H3 extract] NestedTensor video={tuple(video.shape)} "
                          f"audio={tuple(audio.shape) if audio is not None else None}")
                return video, audio, {"type": "nested_tensor"}
        except Exception as e:
            if debug:
                print(f"  · [H3 extract] NestedTensor unbind failed: {e}")

    # --- 普通 tensor ---
    if isinstance(samples, torch.Tensor):
        if debug:
            print(f"  · [H3 extract] plain tensor {tuple(samples.shape)}")
        return samples, None, {"type": "tensor"}

    # --- tuple / list ---
    if isinstance(samples, (tuple, list)):
        video = None
        audio = None
        for item in samples:
            if isinstance(item, torch.Tensor):
                if video is None:
                    video = item
                else:
                    audio = item
        if video is not None:
            fmt = "tuple" if isinstance(samples, tuple) else "list"
            if debug:
                print(f"  · [H3 extract] {fmt} video={tuple(video.shape)} "
                      f"audio={tuple(audio.shape) if audio is not None else None}")
            return video, audio, {"type": fmt}
        raise TypeError(
            f"H3 extract: {type_name} 中未找到 video tensor. "
            f"items: {[type(it).__name__ for it in samples]}"
        )

    pub_attrs = [a for a in dir(samples) if not a.startswith("_")][:25]
    raise TypeError(
        f"H3 extract: 不支持的格式 '{type_name}'. "
        f"期望 5D tensor / NestedTensor / (tensor, tensor) tuple. "
        f"可用属性: {pub_attrs}"
    )


def _h3_reconstruct(video, audio, format_info, debug=False):
    fmt = format_info.get("type", "tensor")
    if fmt == "nested_tensor":
        parts = [video] + ([audio] if audio is not None else [])
        return NestedTensor(parts)
    if fmt == "tensor":
        return video
    if fmt == "tuple":
        return (video, audio) if audio is not None else (video,)
    if fmt == "list":
        return [video, audio] if audio is not None else [video]
    return (video, audio) if audio is not None else video


def _h3_make_nested(video, audio):
    if audio is not None:
        return NestedTensor([video, audio])
    return video


# ─────────────────────────────────────────────────────────────────────────────
# H3 帧数调整
# ─────────────────────────────────────────────────────────────────────────────

def _adjust_frame_count(latent_5d, target_frames, mode, debug=False):
    B, C, T, H, W = latent_5d.shape
    target_T = round((target_frames - 3) / 4) + 1

    if T >= target_T:
        return latent_5d

    if mode == "error":
        raise ValueError(
            f"H3: latent 时间维 T={T}, 期望至少 T={target_T} "
            f"(对应 {target_frames} 帧). 当前 mode=error, 请调整输入或换模式."
        )

    if mode == "replicate_last":
        pad_n = target_T - T
        last = latent_5d[:, :, -1:, :, :].expand(-1, -1, pad_n, -1, -1)
        out = torch.cat([latent_5d, last], dim=2)
        if debug:
            print(f"  · [frame] replicate_last: T {T} -> {target_T} (+{pad_n})")
    elif mode == "zero":
        pad_n = target_T - T
        zeros = torch.zeros(
            B, C, pad_n, H, W,
            dtype=latent_5d.dtype, device=latent_5d.device
        )
        out = torch.cat([latent_5d, zeros], dim=2)
        if debug:
            print(f"  · [frame] zero: T {T} -> {target_T} (+{pad_n})")
    else:
        raise ValueError(f"H3: 未知 pad 模式 '{mode}'")

    return out.contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# 分块数学与 R2V 参考映射
# ─────────────────────────────────────────────────────────────────────────────

def _compute_tile_starts(total, n_tiles, overlap):
    if n_tiles <= 1:
        return [0], total

    stride = int(round(total / n_tiles))
    tile_size = stride + 2 * overlap

    starts = []
    for i in range(n_tiles):
        start = i * stride - overlap
        start = max(0, start)
        starts.append(start)

    dedup = []
    for s in starts:
        if not dedup or s > dedup[-1]:
            dedup.append(s)
    starts = dedup

    return starts, tile_size


def _make_window_1d(length, ov_left, ov_right, dtype, device):
    w = torch.ones(length, dtype=dtype, device=device)
    if ov_left > 0:
        n = min(ov_left, length // 2 + 1)
        if n > 0:
            t = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)[:-1]
            fade = 0.5 - 0.5 * torch.cos(t * 3.14159265)
            w[:n] = torch.minimum(w[:n], fade)
    if ov_right > 0:
        n = min(ov_right, length // 2 + 1)
        if n > 0:
            t = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)[:-1]
            fade = 0.5 - 0.5 * torch.cos((1 - t) * 3.14159265)
            w[-n:] = torch.minimum(w[-n:], fade)
    return w


def _crop_minimax_refs_in_extra_args(extra_args, tile_axis, ax_start, ax_end):
    """
    R2V 核心修复：对 extra_args 条件中的 minimax_refs 进行空间同步裁切，
    确保 DiT 在当前 Tile 采样时只接收该区域对应的 R2V 参考特征。
    """
    if not extra_args:
        return extra_args

    new_extra = extra_args.copy()
    if "cond" in new_extra and isinstance(new_extra["cond"], dict):
        cond_dict = new_extra["cond"].copy()
        if "minimax_refs" in cond_dict:
            refs = cond_dict["minimax_refs"]
            tiled_refs = []
            for ref in refs:
                if isinstance(ref, torch.Tensor) and ref.ndim >= 4:
                    if tile_axis == "H":
                        tiled_ref = ref[..., ax_start:ax_end, :].contiguous()
                    else:
                        tiled_ref = ref[..., ax_start:ax_end].contiguous()
                    tiled_refs.append(tiled_ref)
                else:
                    tiled_refs.append(ref)
            cond_dict["minimax_refs"] = tiled_refs
        new_extra["cond"] = cond_dict

    return new_extra


# ─────────────────────────────────────────────────────────────────────────────
# 主节点
# ─────────────────────────────────────────────────────────────────────────────

class H3TiledSampler(io.ComfyNode):
    @classmethod
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TiledSampler",
            display_name="MiniMax H3 Tiled Sampler",
            category="MiniMax ContextIR",
            description="用于 H3 高清分块采样，支持音画联合 Latent 并且已修复 R2V (Reference to Video) 分块拼接重复现象。",
            inputs=[
                io.Noise.Input("noise", tooltip="随机噪声。"),
                io.Guider.Input("guider", tooltip="引导器：决定使用哪个 H3 模型和提示词/参考条件。"),
                io.Sampler.Input("sampler", tooltip="采样算法，例如 Euler。"),
                io.Sigmas.Input("sigmas", tooltip="噪声调度/采样步。"),
                io.Latent.Input("latent_image", tooltip="待采样的 H3 音画联合 latent。"),
                io.Boolean.Input("bypass_tiling", default=False, tooltip="不分块。"),
                io.Combo.Input("tile_axis", options=["auto", "H", "W"], default="auto", tooltip="切块方向。"),
                io.Int.Input("n_tiles", default=2, min=1, max=8, step=1, tooltip="分块数。"),
                io.Int.Input("tile_overlap", default=8, min=0, max=32, step=1, tooltip="重叠宽度。"),
                io.Int.Input("max_size_for_no_tile", default=24, min=8, max=256, step=1, tooltip="自动不分块阈值。"),
                io.Int.Input("target_frames", default=17, min=1, max=512, step=1, tooltip="最少保留帧数。"),
                io.Combo.Input("frame_padding_mode", options=["replicate_last", "zero", "error"], default="replicate_last", tooltip="帧数不足时的填充方式。"),
                io.Boolean.Input("refine_seams", default=True, tooltip="旧路径接缝二次精修。"),
                io.Int.Input("refine_steps", default=8, min=1, max=25, step=1),
                io.Boolean.Input("debug", default=False, tooltip="调试日志。"),
            ],
            outputs=[
                io.Latent.Output(display_name="分块采样结果"),
                io.Latent.Output(display_name="去噪预测结果"),
            ],
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image,
                bypass_tiling=False, tile_axis="auto", n_tiles=2, tile_overlap=8,
                max_size_for_no_tile=24, target_frames=17,
                frame_padding_mode="replicate_last",
                refine_seams=True, refine_steps=8, debug=False):

        latent = latent_image.copy()
        raw_samples = latent["samples"]

        if debug:
            print(f"→ [H3] TiledSampler(Fixed): input type={type(raw_samples).__name__} bypass={bypass_tiling}")

        video_tensor, audio_tensor, fmt_info = _h3_extract(raw_samples, debug)

        if video_tensor.dim() != 5:
            raise ValueError(f"H3: video latent 必须是 5D [B,C,T,H,W], 实际 {video_tensor.dim()}D")

        video_tensor = _adjust_frame_count(video_tensor, target_frames, frame_padding_mode, debug)
        B, C, F, H, W = video_tensor.shape

        if bypass_tiling:
            return io.NodeOutput(*cls._single_pass(noise, guider, sampler, sigmas, latent, video_tensor, audio_tensor, fmt_info, debug))

        if tile_axis == "auto":
            tile_axis = "H" if H >= W else "W"
        axis_size = H if tile_axis == "H" else W

        if axis_size <= max_size_for_no_tile or n_tiles <= 1:
            return cls._single_pass(noise, guider, sampler, sigmas, latent, video_tensor, audio_tensor, fmt_info, debug)

        starts, tile_size = _compute_tile_starts(axis_size, n_tiles, tile_overlap)
        device = comfy.model_management.get_torch_device()
        dtype = video_tensor.dtype

        video_tensor = video_tensor.to(device=device)
        if audio_tensor is not None:
            audio_tensor = audio_tensor.to(device=device)

        output = torch.zeros_like(video_tensor, dtype=torch.float32, device=device)
        weights_shape = (1, 1, 1, H if tile_axis == "H" else 1, W if tile_axis == "W" else 1)
        weights = torch.zeros(weights_shape, dtype=torch.float32, device=device)

        denoised_output = torch.zeros_like(output)
        denoised_present = False
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        H3TiledSampler._clean_minimax_layout(guider, debug)
        keyframe_contexts = H3TiledSampler._prepare_minimax_keyframes(guider, H, W, debug)

        full_nested = _h3_make_nested(video_tensor, audio_tensor)
        full_noise = noise.generate_noise({"samples": full_nested})
        if hasattr(full_noise, "is_nested") and full_noise.is_nested:
            full_video_noise = full_noise.unbind()[0]
            full_audio_noise = full_noise.unbind()[1] if len(full_noise.unbind()) > 1 else None
        else:
            full_video_noise = full_noise
            full_audio_noise = None

        if cls._can_use_synchronized_euler(sampler, keyframe_contexts):
            if debug:
                print("  · [一致性优化] 启用 Euler 逐步同步分块 (含 R2V 空间条件适配)")
            return io.NodeOutput(*cls._sample_synchronized_euler(
                latent, noise, guider, sampler, sigmas,
                video_tensor, audio_tensor, fmt_info,
                full_noise, starts, tile_size, tile_axis,
                keyframe_contexts, debug,
            ))

        for tile_idx, ax_start in enumerate(starts):
            if tile_axis == "H":
                ax_end = min(ax_start + tile_size, H)
                tile_latent = video_tensor[:, :, :, ax_start:ax_end, :].contiguous()
                tile_video_noise = full_video_noise[:, :, :, ax_start:ax_end, :].contiguous()
            else:
                ax_end = min(ax_start + tile_size, W)
                tile_latent = video_tensor[:, :, :, :, ax_start:ax_end].contiguous()
                tile_video_noise = full_video_noise[:, :, :, :, ax_start:ax_end].contiguous()

            actual_size = tile_latent.shape[3 if tile_axis == "H" else 4]

            tile_nested = _h3_make_nested(tile_latent, audio_tensor)
            tile_noise = _h3_make_nested(tile_video_noise, full_audio_noise)

            x0_output = {}
            callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

            H3TiledSampler._apply_minimax_keyframe_region(keyframe_contexts, tile_axis, ax_start, ax_end, debug)
            # R2V 条件裁切支持
            H3TiledSampler._apply_minimax_refs_region(guider, tile_axis, ax_start, ax_end, debug)
            H3TiledSampler._clean_minimax_layout(guider, debug)

            try:
                tile_samples = guider.sample(
                    tile_noise, tile_nested, sampler, sigmas,
                    denoise_mask=None, callback=callback,
                    disable_pbar=disable_pbar, seed=noise.seed,
                )
            finally:
                H3TiledSampler._restore_minimax_keyframes(keyframe_contexts)

            if hasattr(tile_samples, "is_nested") and tile_samples.is_nested:
                tile_samples_video = tile_samples.unbind()[0]
            else:
                tile_samples_video = tile_samples

            tile_samples_video = tile_samples_video.to(device=device)

            has_prev = tile_idx > 0
            has_next = tile_idx < len(starts) - 1
            ov_left = 0
            ov_right = 0
            if has_prev:
                prev_end = starts[tile_idx - 1] + tile_size
                ov_left = max(0, min(prev_end, ax_end) - ax_start)
            if has_next:
                next_start = starts[tile_idx + 1]
                ov_right = max(0, ax_end - max(ax_start, next_start))
            ov_left = min(ov_left, actual_size)
            ov_right = min(ov_right, actual_size)

            window_1d = _make_window_1d(actual_size, ov_left, ov_right, torch.float32, device)
            if tile_axis == "H":
                window = window_1d.view(1, 1, 1, -1, 1)
                output[:, :, :, ax_start:ax_end, :] += tile_samples_video.float() * window
                weights[:, :, :, ax_start:ax_end, :] += window
            else:
                window = window_1d.view(1, 1, 1, 1, -1)
                output[:, :, :, :, ax_start:ax_end] += tile_samples_video.float() * window
                weights[:, :, :, :, ax_start:ax_end] += window

            del tile_samples, tile_samples_video, tile_latent, tile_nested, tile_noise, window, window_1d
            if device.type == "cuda":
                torch.cuda.empty_cache()

        output = output / weights.clamp(min=1e-8)
        del weights

        if refine_seams and len(starts) > 1:
            output = cls._refine_seams(
                output, full_video_noise, audio_tensor, full_audio_noise,
                starts, tile_size, tile_overlap, tile_axis, noise, guider, sampler, sigmas,
                refine_steps, device, dtype, keyframe_contexts, debug
            )

        intermediate_device = comfy.model_management.intermediate_device()
        output = output.to(dtype=dtype, device=intermediate_device)
        denoised_output_final = output

        reconstructed = _h3_reconstruct(output, audio_tensor, fmt_info, debug)
        denoised_reconstructed = _h3_reconstruct(denoised_output_final, audio_tensor, fmt_info, debug)

        out_dict = latent.copy()
        out_dict["samples"] = reconstructed
        out_denoised_dict = latent.copy()
        out_denoised_dict["samples"] = denoised_reconstructed

        return (out_dict, out_denoised_dict)

    @staticmethod
    def _can_use_synchronized_euler(sampler, keyframe_contexts):
        sampler_function = getattr(sampler, "sampler_function", None)
        if getattr(sampler_function, "__name__", "") != "sample_euler":
            return False
        return not any(not ctx.get("disable_hard_injection") for ctx in keyframe_contexts)

    @staticmethod
    def _sample_synchronized_euler(
        latent_dict, noise, guider, sampler, sigmas,
        video_tensor, audio_tensor, fmt_info,
        full_noise, starts, tile_size, tile_axis,
        keyframe_contexts, debug=False,
    ):
        full_nested = _h3_make_nested(video_tensor, audio_tensor)
        full_shapes = [tuple(video_tensor.shape)]
        if audio_tensor is not None:
            full_shapes.append(tuple(audio_tensor.shape))

        axis_dim = 3 if tile_axis == "H" else 4
        axis_total = video_tensor.shape[axis_dim]

        regions = []
        for tile_idx, ax_start in enumerate(starts):
            ax_end = min(ax_start + tile_size, axis_total)
            actual_size = ax_end - ax_start
            prev_end = starts[tile_idx - 1] + tile_size if tile_idx > 0 else ax_start
            next_start = starts[tile_idx + 1] if tile_idx < len(starts) - 1 else ax_end
            ov_left = min(actual_size, max(0, min(prev_end, ax_end) - ax_start))
            ov_right = min(actual_size, max(0, ax_end - max(ax_start, next_start)))
            window_1d = _make_window_1d(actual_size, ov_left, ov_right, torch.float32, video_tensor.device)
            window = window_1d.view(1, 1, 1, -1, 1) if tile_axis == "H" else window_1d.view(1, 1, 1, 1, -1)
            regions.append((ax_start, ax_end, window))

        weight_shape = (1, 1, 1, axis_total, 1) if tile_axis == "H" else (1, 1, 1, 1, axis_total)
        weights = torch.zeros(weight_shape, dtype=torch.float32, device=video_tensor.device)
        for ax_start, ax_end, window in regions:
            if tile_axis == "H":
                weights[:, :, :, ax_start:ax_end, :] += window
            else:
                weights[:, :, :, :, ax_start:ax_end] += window
        weights = weights.clamp(min=1e-8)

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        H3TiledSampler._apply_minimax_keyframe_region(keyframe_contexts, tile_axis, 0, axis_total, debug)
        H3TiledSampler._clean_minimax_layout(guider, debug)

        original_extra = dict(getattr(sampler, "extra_options", {}) or {})
        original_inpaint = dict(getattr(sampler, "inpaint_options", {}) or {})

        @torch.no_grad()
        def synchronized_euler(
            model, x, step_sigmas, extra_args=None, callback=None, disable=None,
            s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
            **_unused,
        ):
            extra_args = {} if extra_args is None else extra_args
            s_in = x.new_ones([x.shape[0]])

            prepared_model = model.inner_model.inner_model
            saved_model_shapes = getattr(prepared_model, "latent_shapes", None)
            saved_conds = {}

            def set_tile_shapes(tile_shapes):
                prepared_model.latent_shapes = tile_shapes
                for cond_group in getattr(model.inner_model, "conds", {}).values():
                    if cond_group is None:
                        continue
                    for cond in cond_group:
                        model_conds = cond.get("model_conds", {}) if isinstance(cond, dict) else {}
                        shape_cond = model_conds.get("latent_shapes")
                        if shape_cond is not None and hasattr(shape_cond, "cond"):
                            saved_conds.setdefault(id(shape_cond), (shape_cond, shape_cond.cond))
                            shape_cond.cond = tile_shapes

            try:
                for step_idx in model_trange(len(step_sigmas) - 1, disable=disable):
                    gamma = 0.0
                    if s_churn > 0 and s_tmin <= step_sigmas[step_idx] <= s_tmax:
                        gamma = min(s_churn / (len(step_sigmas) - 1), 2 ** 0.5 - 1)
                    sigma_hat = step_sigmas[step_idx] * (gamma + 1)
                    if gamma > 0:
                        eps = torch.randn_like(x) * s_noise
                        x = x + eps * (sigma_hat ** 2 - step_sigmas[step_idx] ** 2) ** 0.5

                    streams = comfy.utils.unpack_latents(x, full_shapes)
                    video_x = streams[0]
                    audio_x = streams[1] if len(streams) > 1 else None
                    video_denoised = torch.zeros_like(video_x, dtype=torch.float32)
                    audio_denoised = torch.zeros_like(audio_x, dtype=torch.float32) if audio_x is not None else None

                    for ax_start, ax_end, window in regions:
                        if tile_axis == "H":
                            video_tile = video_x[:, :, :, ax_start:ax_end, :].contiguous()
                        else:
                            video_tile = video_x[:, :, :, :, ax_start:ax_end].contiguous()

                        tile_streams = [video_tile]
                        if audio_x is not None:
                            tile_streams.append(audio_x)
                        tile_x, tile_shapes = comfy.utils.pack_latents(tile_streams)
                        set_tile_shapes(tile_shapes)

                        # R2V 条件裁切支持
                        tile_extra_args = _crop_minimax_refs_in_extra_args(
                            extra_args, tile_axis, ax_start, ax_end
                        )

                        tile_pred = model(tile_x, sigma_hat * s_in, **tile_extra_args)
                        pred_streams = comfy.utils.unpack_latents(tile_pred, tile_shapes)
                        pred_video = pred_streams[0].float()

                        if tile_axis == "H":
                            video_denoised[:, :, :, ax_start:ax_end, :] += pred_video * window
                        else:
                            video_denoised[:, :, :, :, ax_start:ax_end] += pred_video * window
                        if audio_denoised is not None:
                            audio_denoised += pred_streams[1].float()

                    video_denoised /= weights
                    merged_streams = [video_denoised.to(dtype=video_x.dtype)]
                    if audio_denoised is not None:
                        audio_denoised /= float(len(regions))
                        merged_streams.append(audio_denoised.to(dtype=audio_x.dtype))
                    denoised, _ = comfy.utils.pack_latents(merged_streams)

                    prepared_model.latent_shapes = full_shapes
                    if callback is not None:
                        callback({
                            "x": x, "i": step_idx,
                            "sigma": step_sigmas[step_idx],
                            "sigma_hat": sigma_hat,
                            "denoised": denoised,
                        })
                    d = to_d(x, sigma_hat, denoised)
                    x = x + d * (step_sigmas[step_idx + 1] - sigma_hat)
                return x
            finally:
                prepared_model.latent_shapes = saved_model_shapes
                for cond_obj, old_value in saved_conds.values():
                    cond_obj.cond = old_value

        sync_sampler = comfy.samplers.KSAMPLER(
            synchronized_euler,
            extra_options=original_extra,
            inpaint_options=original_inpaint,
        )

        try:
            samples = guider.sample(
                full_noise, full_nested, sync_sampler, sigmas,
                denoise_mask=None, callback=callback,
                disable_pbar=disable_pbar, seed=noise.seed,
            )
        finally:
            H3TiledSampler._restore_minimax_keyframes(keyframe_contexts)

        sampled_video, _sampled_audio, _ = _h3_extract(samples, debug)
        intermediate_device = comfy.model_management.intermediate_device()
        sampled_video = sampled_video.to(device=intermediate_device, dtype=video_tensor.dtype)

        final_audio = audio_tensor.to(intermediate_device) if audio_tensor is not None else None
        reconstructed = _h3_reconstruct(sampled_video, final_audio, fmt_info, debug)

        out = latent_dict.copy()
        out["samples"] = reconstructed
        out_denoised = latent_dict.copy()
        out_denoised["samples"] = reconstructed

        del weights, regions
        if video_tensor.device.type == "cuda":
            torch.cuda.empty_cache()
        return io.NodeOutput(out, out_denoised)

    @staticmethod
    def _apply_minimax_refs_region(guider, tile_axis, start, end, debug=False):
        """兼容非 Euler 模式下，对 guider 原始条件中的 minimax_refs 进行动态切片"""
        if hasattr(guider, "original_conds"):
            for cond_key, cond_list in guider.original_conds.items():
                for cond in cond_list:
                    if isinstance(cond, dict) and "minimax_refs" in cond:
                        refs = cond["minimax_refs"]
                        tiled_refs = []
                        for ref in refs:
                            if isinstance(ref, torch.Tensor) and ref.ndim >= 4:
                                if tile_axis == "H":
                                    tiled_refs.append(ref[..., start:end, :].contiguous())
                                else:
                                    tiled_refs.append(ref[..., :, start:end].contiguous())
                            else:
                                tiled_refs.append(ref)
                        cond["minimax_refs"] = tiled_refs

    @staticmethod
    def _prepare_minimax_keyframes(guider, full_h, full_w, debug=False):
        contexts = []
        if not hasattr(guider, "original_conds"):
            return contexts

        for cond_key, cond_list in guider.original_conds.items():
            for cond in cond_list:
                if not isinstance(cond, dict):
                    continue
                original = cond.get("minimax_keyframes")
                if not original:
                    continue

                prepared = []
                mismatched = []
                for kf in original:
                    if not isinstance(kf, dict):
                        continue
                    item = kf.copy()
                    latent = item.get("latent")
                    if not isinstance(latent, torch.Tensor) or latent.dim() != 5:
                        continue

                    old_hw = tuple(latent.shape[-2:])
                    if old_hw != (full_h, full_w):
                        mismatched.append(old_hw)
                    prepared.append(item)

                disable_hard_injection = bool(mismatched)
                contexts.append({
                    "cond": cond,
                    "original": original,
                    "prepared": prepared,
                    "disable_hard_injection": disable_hard_injection,
                })
        return contexts

    @staticmethod
    def _apply_minimax_keyframe_region(contexts, tile_axis, start, end, debug=False):
        for ctx in contexts:
            if ctx.get("disable_hard_injection"):
                ctx["cond"].pop("minimax_keyframes", None)
                continue

            tiled = []
            for kf in ctx["prepared"]:
                item = kf.copy()
                latent = item["latent"]
                if tile_axis == "H":
                    region = latent[:, :, :, start:end, :].contiguous()
                else:
                    region = latent[:, :, :, :, start:end].contiguous()

                pad_h = (-region.shape[-2]) % 2
                pad_w = (-region.shape[-1]) % 2
                if pad_h or pad_w:
                    region = F.pad(region, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
                item["latent"] = region
                tiled.append(item)
            ctx["cond"]["minimax_keyframes"] = tiled

    @staticmethod
    def _restore_minimax_keyframes(contexts):
        for ctx in contexts:
            ctx["cond"]["minimax_keyframes"] = ctx["original"]

    @staticmethod
    def _refine_seams(output, full_video_noise, audio_tensor, full_audio_noise,
                      starts, tile_size, tile_overlap, tile_axis, noise, guider, sampler, sigmas,
                      refine_steps, device, dtype, keyframe_contexts=None, debug=False):
        if refine_steps <= 0 or sigmas.shape[-1] <= 1:
            return output

        refine_sigmas = sigmas[-(refine_steps + 1):].clone()
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        axis_dim = 3 if tile_axis == "H" else 4
        axis_total = output.shape[axis_dim]

        for seam_idx in range(len(starts) - 1):
            seam_center = starts[seam_idx + 1]
            half = max(8, tile_overlap)
            band_start = max(0, seam_center - half)
            band_end = min(axis_total, seam_center + half)

            if band_end - band_start < 2:
                continue

            if tile_axis == "H":
                band_latent = output[:, :, :, band_start:band_end, :].contiguous()
                band_noise = full_video_noise[:, :, :, band_start:band_end, :].contiguous()
            else:
                band_latent = output[:, :, :, :, band_start:band_end].contiguous()
                band_noise = full_video_noise[:, :, :, :, band_start:band_end].contiguous()

            band_nested = _h3_make_nested(band_latent, audio_tensor)
            band_noise_nested = _h3_make_nested(band_noise, full_audio_noise)

            x0_output = {}
            callback = latent_preview.prepare_callback(guider.model_patcher, refine_sigmas.shape[-1] - 1, x0_output)
            H3TiledSampler._apply_minimax_keyframe_region(keyframe_contexts or [], tile_axis, band_start, band_end, debug)
            H3TiledSampler._clean_minimax_layout(guider, debug)
            try:
                band_samples = guider.sample(
                    band_noise_nested, band_nested, sampler, refine_sigmas,
                    denoise_mask=None, callback=callback,
                    disable_pbar=disable_pbar, seed=noise.seed,
                )
            finally:
                H3TiledSampler._restore_minimax_keyframes(keyframe_contexts or [])

            if hasattr(band_samples, "is_nested") and band_samples.is_nested:
                band_samples_video = band_samples.unbind()[0]
            else:
                band_samples_video = band_samples
            band_samples_video = band_samples_video.to(device=device)

            if tile_axis == "H":
                output[:, :, :, band_start:band_end, :] = band_samples_video.float()
            else:
                output[:, :, :, :, band_start:band_end] = band_samples_video.float()

            del band_latent, band_noise, band_nested, band_noise_nested, band_samples, band_samples_video
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return output

    @staticmethod
    def _clean_minimax_layout(guider, debug=False):
        if hasattr(guider, 'model_patcher') and hasattr(guider.model_patcher, 'model'):
            model = guider.model_patcher.model
            if hasattr(model, 'diffusion_model'):
                if hasattr(model, '_cached_extra_conds'):
                    cached = model._cached_extra_conds
                    if isinstance(cached, dict):
                        for k, v in cached.items():
                            if hasattr(v, 'cond') and isinstance(v.cond, dict):
                                if 'layout' in v.cond:
                                    del v.cond['layout']
                                if 'cond_video_latents' in v.cond:
                                    del v.cond['cond_video_latents']

    @staticmethod
    def _single_pass(noise, guider, sampler, sigmas, latent_dict,
                     video_tensor, audio_tensor, fmt_info, debug=False):
        H3TiledSampler._clean_minimax_layout(guider, debug)
        keyframe_contexts = H3TiledSampler._prepare_minimax_keyframes(
            guider, video_tensor.shape[-2], video_tensor.shape[-1], debug
        )

        latent_for_sample = _h3_reconstruct(video_tensor, audio_tensor, fmt_info, debug)
        latent_dict["samples"] = latent_for_sample

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        H3TiledSampler._apply_minimax_keyframe_region(keyframe_contexts, "H", 0, video_tensor.shape[-2], debug)
        H3TiledSampler._clean_minimax_layout(guider, debug)
        try:
            samples = guider.sample(
                noise.generate_noise(latent_dict),
                latent_for_sample,
                sampler,
                sigmas,
                denoise_mask=None,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=noise.seed,
            )
        finally:
            H3TiledSampler._restore_minimax_keyframes(keyframe_contexts)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent_dict.copy()
        out["samples"] = samples
        return (out, out)


NODE_CLASS_MAPPINGS = {
    "H3TiledSampler": H3TiledSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TiledSampler": "H3 高清分块采样器（音画/关键帧/R2V修复版）",
}
