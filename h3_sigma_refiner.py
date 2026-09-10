"""MiniMax H3 Sigma Refiner - 低噪细节精修器。

对低 Sigma 区间进行局部加步：保留原始调度的高噪头部不动，从阈值点起
把尾部重采样成更长、更平滑的曲线，让模型在细节收尾阶段多走几步。

基于 ComfyUI-YCNodes-MiniMax-H3（MIT License, Copyright (c) 2026 yc）复刻。
"""

import math

import torch

from comfy_api.latest import io


class H3SigmaRefiner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SigmaRefiner",
            display_name="MiniMax H3 Sigma Refiner",
            description=(
                "对低噪点区间进行局部微雕加步，消除高速运动边缘的马赛克与像素紊乱。"
            ),
            category="MiniMax ContextIR",
            inputs=[
                io.Sigmas.Input("sigmas", tooltip="输入的原始噪声序列。"),
                io.Int.Input(
                    "extra_steps", default=1, min=0, max=15, step=1,
                    tooltip="在低噪点区间额外增加的细节平滑步数。",
                ),
                io.Float.Input(
                    "start_at_sigma", default=0.7, min=0.0, max=20.0, step=0.01,
                    tooltip="启动细节加步的 Sigma 阈值。H3 推荐设在 2.0～3.5 之间。",
                ),
                io.Float.Input(
                    "end_at_sigma", default=0.0, min=0.0, max=5.0, step=0.01,
                    tooltip="结束细化的 Sigma 边界（默认 0.0）。",
                ),
                io.Combo.Input(
                    "spacing",
                    options=["cosine", "linear", "exponential"],
                    default="cosine",
                    tooltip="插值分布曲线。cosine（余弦）在趋近于 0 时分布更密，消噪效果最丝滑。",
                ),
            ],
            outputs=[
                io.Sigmas.Output(display_name="sigmas"),
            ],
        )

    @classmethod
    def execute(cls, sigmas, extra_steps: int = 1, start_at_sigma: float = 0.7,
                end_at_sigma: float = 0.0, spacing: str = "cosine") -> io.NodeOutput:
        if extra_steps <= 0:
            return io.NodeOutput(sigmas)

        sigmas_cpu = sigmas.detach().cpu()

        idx = -1
        for i, s in enumerate(sigmas_cpu):
            if s <= start_at_sigma:
                idx = i
                break

        if idx == -1 or idx >= len(sigmas_cpu) - 1:
            return io.NodeOutput(sigmas)

        unmodified_head = sigmas_cpu[:idx]
        A = sigmas_cpu[idx].item()
        B = max(end_at_sigma, sigmas_cpu[-1].item())

        original_tail_len = len(sigmas_cpu) - idx
        new_tail_len = original_tail_len + extra_steps

        t = torch.linspace(0.0, 1.0, steps=new_tail_len)

        if spacing == "cosine":
            factor = (1.0 - torch.cos(t * math.pi)) / 2.0
        elif spacing == "exponential":
            alpha = 3.0
            factor = (torch.exp(t * alpha) - 1.0) / (math.exp(alpha) - 1.0)
        else:
            factor = t

        new_tail = A + (B - A) * factor

        if sigmas_cpu[-1].item() == 0.0 and B > 0.0:
            new_tail = torch.cat([new_tail, torch.tensor([0.0])])

        new_sigmas = torch.cat([unmodified_head, new_tail])

        return io.NodeOutput(
            new_sigmas.to(device=sigmas.device, dtype=sigmas.dtype))
