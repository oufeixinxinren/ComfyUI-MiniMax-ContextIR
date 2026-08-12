"""MiniMax H3 Resolution Selector.

Native ``DynamicCombo`` version: selecting an aspect ratio swaps the ``size``
dropdown to that ratio's own ``WxH`` preset list.

The presets follow the official/community reference style: the official 16:9
canvas table is the base, other ratios are converted from the same pixel area,
and every value is 32-aligned. Ratios are therefore *approximately* correct
(exactly like the official size-settings reference), not mathematically exact.

``16:9`` is deliberately listed below ``9:16``.
"""

from __future__ import annotations

import math

from comfy_api.latest import ComfyExtension, io


# Official MiniMax H3 16:9 canvas presets (width x height, both multiples of 32).
OFFICIAL_16_9 = {
    "608×352": (608, 352),
    "736×416": (736, 416),
    "864×480": (864, 480),
    "960×544": (960, 544),
    "1056×608": (1056, 608),
    "1152×640": (1152, 640),
    "1216×672": (1216, 672),
    "1280×736": (1280, 736),
    "1344×768": (1344, 768),
    "1376×768": (1376, 768),
    "1504×832": (1504, 832),
    "1664×928": (1664, 928),
    "1824×1024": (1824, 1024),
    "1920×1088": (1920, 1088),
}

RATIOS = {
    "1:1": 1.0,
    "2:3": 2.0 / 3.0,
    "3:2": 3.0 / 2.0,
    "3:4": 3.0 / 4.0,
    "4:3": 4.0 / 3.0,
    "9:16": 9.0 / 16.0,
    "16:9": 16.0 / 9.0,
    "21:9": 21.0 / 9.0,
}

# Same labels as the official Resolution Selector. 16:9 is deliberately below
# 9:16 (requested ordering).
RATIO_LABELS = [
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
]

CANVAS_MULTIPLE = 32
MIN_EDGE = 256
MAX_EDGE = 5760

# Extra 32-aligned presets close to common resolutions that the area-based
# conversion does not produce. 736x1280 is the 720p-class 9:16 portrait value
# (720 itself is not a multiple of 32; 736/1280 ≈ 0.575, ~2.2% from 9:16).
EXTRA_SIZE_BY_RATIO = {
    "9:16": ["736×1280"],
}


def _round_to_multiple(value: float, multiple: int = CANVAS_MULTIPLE) -> int:
    return int(math.floor(value / multiple + 0.5) * multiple)


def canvas_for(ratio: str, size: str) -> tuple[int, int]:
    """Reference-style (width, height) for a ratio + 16:9 base preset."""
    if ratio not in RATIOS:
        raise ValueError(f"Unknown MiniMax H3 ratio: {ratio}")
    if size not in OFFICIAL_16_9:
        raise ValueError(f"Unknown MiniMax H3 size preset: {size}")
    if ratio == "16:9":
        return OFFICIAL_16_9[size]

    width16, height16 = OFFICIAL_16_9[size]
    area = width16 * height16
    aspect = RATIOS[ratio]
    if aspect >= 1.0:
        height = _round_to_multiple(math.sqrt(area / aspect))
        width = _round_to_multiple(aspect * height)
    else:
        width = _round_to_multiple(math.sqrt(area * aspect))
        height = _round_to_multiple(width / aspect)
    width = min(max(width, MIN_EDGE), MAX_EDGE)
    height = min(max(height, MIN_EDGE), MAX_EDGE)
    return width, height


def _size_options_for(ratio: str) -> list[str]:
    """One WxH list per ratio, derived from the official 16:9 presets."""
    seen: list[str] = []
    for base in OFFICIAL_16_9:
        width, height = canvas_for(ratio, base)
        label = f"{width}×{height}"
        if label not in seen:
            seen.append(label)
    for extra in EXTRA_SIZE_BY_RATIO.get(ratio, []):
        if extra not in seen:
            seen.append(extra)
    return seen


def _default_size_for(ratio: str) -> str:
    width, height = canvas_for(ratio, "1344×768")
    return f"{width}×{height}"


SIZE_OPTIONS_BY_RATIO = {ratio: _size_options_for(ratio) for ratio in RATIOS}
DEFAULT_SIZE_BY_RATIO = {ratio: _default_size_for(ratio) for ratio in RATIOS}


def parse_size(value: str, ratio: str) -> tuple[int, int, str]:
    """Validate a WxH label against the selected ratio's own preset list."""
    text = (value or "").strip()
    try:
        width_text, height_text = text.split("×", 1)
        width, height = int(width_text), int(height_text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid WxH label: {value!r}") from exc
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError(f"{text} is not aligned to {CANVAS_MULTIPLE}px")
    if text not in SIZE_OPTIONS_BY_RATIO.get(ratio, []):
        raise ValueError(
            f"Size {text!r} does not match the selected ratio {ratio!r}; "
            "pick a size from the dropdown for this ratio."
        )
    return width, height, text


def _ratio_options() -> list[io.DynamicCombo.Option]:
    options = []
    for label in RATIO_LABELS:
        ratio = label.split(" ")[0]
        options.append(
            io.DynamicCombo.Option(
                label,
                [
                    io.Combo.Input(
                        "size",
                        options=SIZE_OPTIONS_BY_RATIO[ratio],
                        default=DEFAULT_SIZE_BY_RATIO[ratio],
                        tooltip=(
                            f"Official reference-style {ratio} width×height presets "
                            "(32-aligned; options follow the selected ratio)."
                        ),
                    ),
                ],
            )
        )
    return options


class MiniMaxH3ResolutionSelector(io.ComfyNode):
    """Calculate H3 width/height from an aspect ratio and its matching WxH preset."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ResolutionSelector",
            display_name="MiniMax H3 Resolution Selector",
            description=(
                "Aspect ratio + official reference-style WxH preset -> H3 width/height. "
                "The size dropdown only shows presets for the selected ratio "
                "(native DynamicCombo, no JS)."
            ),
            category="model/conditioning/minimax",
            inputs=[
                io.DynamicCombo.Input(
                    "aspect_ratio",
                    options=_ratio_options(),
                    tooltip="Aspect ratio for the output dimensions; each ratio has its own size list.",
                ),
            ],
            outputs=[
                io.Int.Output("width", tooltip="Calculated width in pixels (multiple of 32)."),
                io.Int.Output("height", tooltip="Calculated height in pixels (multiple of 32)."),
                io.String.Output("size", tooltip='WxH label, e.g. "1344×768".'),
            ],
        )

    @classmethod
    def execute(cls, aspect_ratio: dict) -> io.NodeOutput:
        if not isinstance(aspect_ratio, dict):
            raise ValueError(f"Invalid aspect_ratio payload: {aspect_ratio!r}")
        ratio_label = aspect_ratio.get("aspect_ratio") or ""
        size = aspect_ratio.get("size")
        ratio = ratio_label.split(" ")[0]
        if ratio not in RATIOS:
            raise ValueError(f"Unknown MiniMax H3 aspect ratio: {ratio_label!r}")
        width, height, label = parse_size(size, ratio)
        return io.NodeOutput(width, height, label)


class MiniMaxH3ResolutionExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3ResolutionSelector]


async def comfy_entrypoint():
    return MiniMaxH3ResolutionExtension()


__all__ = [
    "MiniMaxH3ResolutionSelector",
    "MiniMaxH3ResolutionExtension",
    "comfy_entrypoint",
]
