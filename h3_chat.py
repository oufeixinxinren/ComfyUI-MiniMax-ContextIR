"""MiniMax H3 Multimodal Chat - skill loader + external multimodal LLM chat node.

Features:
- loads the bundled MiniMax official skills from this plugin's skills/ directory
- calls any OpenAI-compatible multimodal chat completions API
- H3 media input ports (first/last frame, reference images/videos/audio)
- click a connected media chip or type @-mention in the message window to reference media,
  e.g. @ref_image_0 / @ref_video_0 / @ref_audio_0
- duration (1-15 s) is injected into the system prompt so shots, actions and
  timestamps stay within the target video length
- prompt_only toggle: assemble the prompt text without calling the API
- outputs reply text and the assembled prompt text
"""

from __future__ import annotations

import json
import re

from comfy_api.latest import ComfyExtension, io

from .h3_chat_api import (
    build_label_map,
    chat_completion,
    collect_media_inputs,
    discover_skills,
    extract_prompt_from_reply,
    load_skill_reference,
    load_skill_text,
    media_to_api_parts,
    referenced_tokens,
    render_prompt_text,
    resolve_skill,
    select_skill_with_api,
)


AUTO = "auto"
DEFAULT_SYSTEM = "你是一个有帮助的AI助手。"
PROMPT_MODES = ["auto", "T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"]
SKILL_STATE_TAG = re.compile(r"<mmx_skill_state>\s*(\{.*?\})\s*</mmx_skill_state>", re.DOTALL)
SKILL_EXEC_PROTOCOL = """
你正在通过 ComfyUI 的 Skill 执行器工作。严格遵循下方当前 Skill，并遵守以下交互协议：
1. 只完成当前 Skill 能在文本对话中完成的工作。Skill 提到画布、媒体生成、联网工具或
   Hub agent 时，不得声称已经执行；应输出对应方案、提示词或说明当前需要连接的 ComfyUI 节点。
2. 信息不足或到达确认门时，先提问并等待用户。每次只推进当前阶段，不得替用户确认。
3. 回复正文之后必须追加一个状态标记，且标记必须是回复的最后内容：
<mmx_skill_state>{"stage":"当前阶段","options":["选项1","选项2"],"load_references":[],"final":false}</mmx_skill_state>
4. 需要用户选择时，options 提供 2 到 6 个可直接作为用户回复的完整选项；开放问题可以使用空数组。
5. Skill 要求读取 reference 时，如果该文件尚未出现在"已加载 references"，必须先把相对路径
   写入 load_references。执行器会加载文件并让你重新回答，不要猜测文件内容。
6. 只有已经交付当前 Skill 要求的最终文本产物时才设置 final=true。最终产物必须完整写在状态标记之前。
7. 使用简体中文交流和输出；协议字段、H3 固定字段、标签以及用户要求原样保留的内容除外。
""".strip()


def _skill_options() -> list[str]:
    return [AUTO] + [skill["label"] for skill in discover_skills()]


def _default_meta() -> dict:
    return {"skill": "", "skill_loaded": False, "loaded_references": []}


def _parse_history(raw: str) -> tuple[list[dict], dict]:
    if not raw or not raw.strip():
        return [], _default_meta()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Chat history JSON is invalid: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("Chat history must be a JSON list of messages.")
    history = []
    meta = _default_meta()
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "_meta":
            meta = {
                "skill": str(item.get("skill") or ""),
                "skill_loaded": bool(item.get("skill_loaded")),
                "loaded_references": [
                    str(ref) for ref in (item.get("loaded_references") or []) if isinstance(ref, str)
                ],
            }
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        media = item.get("media") or []
        if not isinstance(media, list):
            media = []
        history.append(
            {
                "role": role,
                "content": content,
                "media": [str(token) for token in media if isinstance(token, str)],
            }
        )
    return history, meta


def _serialize_history(history: list[dict], meta: dict) -> str:
    payload = list(history)
    payload.append(
        {
            "role": "_meta",
            "skill": str(meta.get("skill") or ""),
            "skill_loaded": bool(meta.get("skill_loaded")),
            "loaded_references": list(meta.get("loaded_references") or []),
        }
    )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _trim_history(history: list[dict], max_rounds: int) -> list[dict]:
    max_messages = max(1, int(max_rounds)) * 2
    trimmed = history[-max_messages:]
    while trimmed and trimmed[0]["role"] == "assistant":
        trimmed.pop(0)
    return trimmed


def _history_to_api_messages(history: list[dict], media: dict, label_map: dict[str, str]) -> list[dict]:
    messages = []
    for item in history:
        if item["role"] != "user":
            messages.append({"role": "assistant", "content": item["content"]})
            continue
        tokens = [token for token in item.get("media", []) if token in media]
        if tokens:
            content = [{"type": "text", "text": item["content"]}]
            for token in tokens:
                value = media[token]
                if isinstance(value, dict):
                    from .h3_chat_api import _audio_to_data_url

                    audio_url = _audio_to_data_url(value)
                    content.append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_url.split(",", 1)[1], "format": "wav"},
                        }
                    )
                else:
                    from .h3_chat_api import _image_to_data_url

                    frames = value
                    if frames.ndim == 3:
                        frames = frames.unsqueeze(0)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_to_data_url(frames[0].unsqueeze(0))},
                        }
                    )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": item["content"]})
    return messages


def _default_flow_state() -> dict:
    return {"stage": "未开始", "options": [], "load_references": [], "final": False}


def _parse_skill_state(reply: str) -> tuple[str, dict]:
    matches = list(SKILL_STATE_TAG.finditer(reply or ""))
    if not matches:
        return (reply or "").strip(), _default_flow_state()
    match = matches[-1]
    try:
        state = json.loads(match.group(1))
    except json.JSONDecodeError:
        return SKILL_STATE_TAG.sub("", reply).strip(), _default_flow_state()
    if not isinstance(state, dict):
        state = {}
    text = (reply[: match.start()] + reply[match.end() :]).strip()
    return text, {
        "stage": str(state.get("stage") or "未开始")[:40],
        "options": [str(item)[:240] for item in (state.get("options") or [])[:6] if str(item).strip()],
        "load_references": [
            str(item) for item in (state.get("load_references") or []) if isinstance(item, str)
        ],
        "final": bool(state.get("final")),
    }


def _media_summary(media: dict) -> str:
    labels = {
        "first_frame": "首帧",
        "last_frame": "尾帧",
        "ref_image_": "参考图",
        "ref_video_audio_": "参考视频音轨",
        "ref_video_": "参考视频",
        "ref_audio_": "参考音频",
    }
    parts = []
    for prefix, label in labels.items():
        def _ordinal(token: str) -> int:
            tail = token.rsplit("_", 1)[-1]
            return int(tail) if tail.isdigit() else 0

        tokens = sorted([t for t in media if t.startswith(prefix)], key=_ordinal)
        if tokens:
            parts.append(f"{label} {len(tokens)} 个")
    return "、".join(parts) if parts else "无"


def _mode_instruction(prompt_mode: str, media_summary: str) -> str:
    mode = (prompt_mode or "auto").upper()
    if mode == "AUTO":
        return (
            "提示词生成模式：自动。\n"
            f"已连接媒体：{media_summary}。\n"
            "请根据用户需求和已连接媒体，自动判断应生成的 MiniMax H3 提示词类型：\n"
            "- T2VA：纯文生视频；\n"
            "- I2VA：首帧图生视频（使用 first_frame）；\n"
            "- L2VA：尾帧生视频（使用 last_frame）；\n"
            "- FL2VA：首尾帧生视频（同时使用 first_frame 与 last_frame）；\n"
            "- REF2VA：仅参考媒体生视频（ref_image/ref_video/ref_audio）；\n"
            "注意：ref_video_audio 只是同编号 ref_video 的音轨，单独连接不算参考媒体；\n"
            "输出与该类型匹配的 H3 提示词结构，并用 <Picture i> / <Video k> / <Audio j> 标签引用可用媒体。"
        )
    descriptions = {
        "T2VA": "纯文生视频（不使用任何媒体标签）",
        "I2VA": "首帧图生视频（使用 <Picture 1> 指代首帧）",
        "FL2VA": "首尾帧生视频（<Picture 1> 首帧、<Picture 2> 尾帧）",
        "L2VA": "尾帧生视频（使用 <Picture 1> 指代尾帧）",
        "REF2VA": "参考媒体生视频（<Picture i> / <Video k> / <Audio j>；可包含首/尾关键帧作为 <Picture N> 锚点，summary 使用 keyframe completion）",
    }
    return (
        f"提示词生成模式：{mode}（{descriptions.get(mode, '')}）。\n"
        f"已连接媒体：{media_summary}。\n"
        "请输出与该模式匹配的 MiniMax H3 提示词结构。"
    )


def _duration_instruction(duration: float) -> str:
    seconds = f"{duration:g}"
    return (
        f"目标视频时长：{seconds} 秒（允许范围 1–15 秒）。\n"
        "所有分镜、动作、台词、音效与音乐切点的时间戳必须落在该时长内；"
        "时长越短动作越简洁（1–2 秒只安排一个主要动作或单一镜头），"
        "时长越长可容纳更多分镜与复杂编排。时间戳（如 At 00:02.500）不得超过该时长。"
    )


def _apply_prompt_mode(prompt_mode: str, media: dict) -> tuple[str, dict]:
    """Apply an explicit prompt mode by shielding (ignoring) foreign inputs.

    Returns ``(effective_mode, effective_media)``. Inputs that do not belong to
    the selected mode are dropped instead of raising, so connected media that
    is not used simply has no effect. Explicit modes raise only when a required
    input is missing:

    - T2VA  : all media ignored (no media requirement)
    - I2VA  : only first_frame is used (requires first_frame)
    - L2VA  : only last_frame is used (requires last_frame)
    - FL2VA : only first/last keyframes are used, refs ignored
              (requires at least one keyframe)
    - REF2VA: keyframes + references are all used (requires at least one
              reference media)
    """
    mode = (prompt_mode or "auto").upper()
    if mode == "AUTO":
        return mode, media
    has_first = "first_frame" in media
    has_last = "last_frame" in media
    has_refs = _has_reference_media(media)

    def keep(*tokens: str) -> dict:
        return {key: value for key, value in media.items() if key in tokens}

    if mode == "T2VA":
        return mode, {}
    if mode == "I2VA":
        if not has_first:
            raise ValueError("Prompt mode I2VA requires first_frame.")
        return mode, keep("first_frame")
    if mode == "L2VA":
        if not has_last:
            raise ValueError("Prompt mode L2VA requires last_frame.")
        return mode, keep("last_frame")
    if mode == "FL2VA":
        if not (has_first or has_last):
            raise ValueError("Prompt mode FL2VA requires first_frame and/or last_frame.")
        return mode, keep("first_frame", "last_frame")
    if mode == "REF2VA":
        if not has_refs:
            raise ValueError(
                "Prompt mode REF2VA requires at least one reference media "
                "(ref_images / ref_videos / ref_video_audios / ref_audios)."
            )
        return mode, media
    raise ValueError(f"Unknown MiniMax H3 prompt mode: {prompt_mode}")


def _has_reference_media(media: dict) -> bool:
    """True when a real reference media is connected.

    Matches the official node semantics: ref_video_audio_N is only a reference
    when the same-numbered ref_video_N is also connected; a standalone
    soundtrack alone does not count as a reference.
    """
    for token in media:
        if token.startswith("ref_image_") or token.startswith("ref_audio_"):
            return True
        if token.startswith("ref_video_audio_"):
            ordinal = token[len("ref_video_audio_"):]
            if f"ref_video_{ordinal}" in media:
                return True
            continue
        if token.startswith("ref_video_"):
            return True
    return False


def _strip_tokens(text: str, tokens: list[str]) -> str:
    """Remove @-mentions for tokens that were shielded by the selected mode."""
    for token in tokens:
        text = re.sub(rf"@{re.escape(token)}\b", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _infer_mode_from_media(media: dict) -> str:
    """Deterministic auto-mode mapping, matching the Unified node's auto logic."""
    has_first = "first_frame" in media
    has_last = "last_frame" in media
    has_refs = _has_reference_media(media)
    if has_refs:
        return "REF2VA"
    if has_first and has_last:
        return "FL2VA"
    if has_first:
        return "I2VA"
    if has_last:
        return "L2VA"
    return "T2VA"


def _build_system_text(skill: dict | None, loaded_references: list[str] | None = None,
                       prompt_mode: str = "auto", media: dict | None = None,
                       duration: float = 5.0) -> str:
    loaded_references = loaded_references or []
    parts = []
    if not parts:
        parts.append(DEFAULT_SYSTEM)
    parts.append(_mode_instruction(prompt_mode, _media_summary(media or {})))
    if (prompt_mode or "auto").upper() == "AUTO":
        parts.append(
            "自动模式规则（必须遵守）：已连接参考媒体（ref_image / ref_video / ref_audio；"
            "ref_video_audio 需与同编号 ref_video 同时连接才算）→ 使用 REF2VA 结构；"
            "仅连接 first_frame+last_frame → FL2VA；仅 first_frame → I2VA；"
            "仅 last_frame → L2VA；无媒体 → T2VA。直接输出最终 H3 提示词正文，"
            "不要解释，不要输出模式名称，不要输出与提示词无关的内容。"
        )
    parts.append(_duration_instruction(float(duration)))
    if skill is not None:
        parts.append(
            f"当前 Skill：{skill['name']} ({skill['id']})\n"
            f"可用 references：{', '.join(skill['references']) or '无'}\n"
            f"已加载 references：{', '.join(loaded_references) or '无'}\n"
            f"===== {skill['skill_file']} =====\n{load_skill_text(skill)}"
        )
        for relative_path in loaded_references:
            parts.append(f"===== reference: {relative_path} =====\n{load_skill_reference(skill, relative_path)}")
        parts.append(SKILL_EXEC_PROTOCOL)
    return "\n\n".join(parts)


class MiniMaxH3MultimodalChat(io.ComfyNode):
    """Multi-turn multimodal chat with skill loading and an external LLM API."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MultimodalChat",
            display_name="MiniMax H3 Multimodal Chat",
            description=(
                "Multi-turn chat with bundled MiniMax official skill loading and any OpenAI-compatible "
                "multimodal API. Click a connected media chip in the message window (or type "
                "@tokens such as @first_frame / @last_frame / @ref_image_N / @ref_video_N / "
                "@ref_video_audio_N / @ref_audio_N) to reference connected H3 media."
            ),
            category="MiniMax ContextIR",
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                io.String.Input("prompt", multiline=True, default="",
                                tooltip="User message; type @ to reference connected media."),
                io.String.Input("input_string", multiline=True, default="",
                                tooltip="External text input; when connected, overrides the chat window text."),
                io.String.Input("chat_history", multiline=True, default="[]",
                                tooltip="JSON chat history, maintained by the node."),
                io.String.Input("conversations", multiline=True, default="{}",
                                tooltip="Conversation store managed by the chat UI (hidden)."),
                io.String.Input("request_id", default="",
                                tooltip="Frontend request id; set automatically by the chat UI."),
                io.Combo.Input("skill", options=_skill_options(), default=AUTO,
                               tooltip="auto asks the LLM to pick a skill on the first message."),
                io.Boolean.Input("auto_load_references", default=True,
                                 tooltip="On the first skill load, inject all local reference files into the system prompt so the remote API can read them."),
                io.Combo.Input("prompt_mode", options=PROMPT_MODES, default="auto",
                               tooltip="auto lets the LLM decide the H3 prompt type from connected ports and the request; explicit modes are validated against the connected ports."),
                io.Float.Input("duration", default=5.0, min=1, max=15, step=0.01,
                               tooltip="Target video length in seconds (1-15); injected into the system prompt so shots, actions and timestamps stay within the duration."),
                io.String.Input("api_base", default="https://api.openai.com/v1",
                                tooltip="OpenAI-compatible API base URL."),
                io.String.Input("api_key", default="", tooltip="API key for the external LLM."),
                io.String.Input("model", default="gpt-4o", tooltip="Multimodal model name."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.01),
                io.Int.Input("seed", default=0, min=-1, max=2147483647, step=1,
                             control_after_generate=io.ControlAfterGenerate.fixed,
                             tooltip="Fixed seed (>=0) makes repeated runs reproducible (same prompt). -1 = no seed sent (random)."),
                io.Boolean.Input("enable_thinking", default=False,
                                 tooltip="Request reasoning from the model (passes reasoning_effort; supported by provider/model only)."),
                io.Combo.Input("reasoning_effort", options=["none", "low", "medium", "high"], default="medium",
                               tooltip="Used only when enable_thinking is on."),
                io.Boolean.Input("output_thinking", default=False,
                                 tooltip="Keep <think> reasoning blocks in the reply when the model emits them."),
                io.Int.Input("max_tokens", default=2048, min=20, max=128000, step=1,
                             tooltip="Max completion tokens; clamped to 128000 before sending (provider/model may support less)."),
                io.Int.Input("max_history_rounds", default=1000, min=1, max=10000, step=1,
                             tooltip="Client-side history trim; the real limit is the model's context window."),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image"),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video frame batch"),
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
                io.String.Output(display_name="reply"),
                io.String.Output(display_name="prompt_text"),
                io.String.Output(display_name="chat_history"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt="",
        input_string="",
        chat_history="[]",
        conversations="{}",
        request_id="",
        skill=AUTO,
        prompt_mode="auto",
        api_base="https://api.openai.com/v1",
        api_key="",
        model="gpt-4o",
        temperature=0.7,
        seed=0,
        enable_thinking=False,
        reasoning_effort="medium",
        output_thinking=False,
        max_tokens=1024,
        max_history_rounds=100,
        auto_load_references=True,
        duration=5.0,
        first_frame=None,
        last_frame=None,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ) -> io.NodeOutput:
        del conversations
        del request_id
        duration = float(duration)
        if not (1 <= duration <= 15):
            raise ValueError("Duration must be between 1 and 15 seconds.")
        seed_value = int(seed) if int(seed) >= 0 else None
        user_text = (input_string or "").strip() or (prompt or "").strip()
        history, meta = _parse_history(chat_history)
        history = _trim_history(history, int(max_history_rounds))
        media = collect_media_inputs(
            first_frame, last_frame, ref_images, ref_videos, ref_video_audios, ref_audios
        )
        full_media = media
        effective_mode, media = _apply_prompt_mode(prompt_mode, media)
        effective_skill = skill
        label_map = build_label_map(media)
        tokens = referenced_tokens(user_text)
        shielded_tokens = [token for token in tokens if token in full_media and token not in media]
        if shielded_tokens:
            user_text = _strip_tokens(user_text, shielded_tokens)
            tokens = referenced_tokens(user_text)
        for token in tokens:
            if token not in media:
                raise ValueError(
                    f"{token} is not connected; connect it to the node or remove it from the message."
                )
        prompt_text = render_prompt_text(user_text, label_map)
        input_source = "input_string" if (input_string or "").strip() else ("prompt" if (prompt or "").strip() else "none")
        report_lines = [f"input_source={input_source}", f"media={len(media)}",
                        f"tokens={len(tokens)}", f"seed={seed_value}", f"mode={effective_mode}"]
        if len(full_media) > len(media):
            report_lines.append(f"ignored_media={len(full_media) - len(media)}")

        if not user_text:
            last_assistant = next(
                (item for item in reversed(history) if item["role"] == "assistant"), None
            )
            if last_assistant is None:
                raise ValueError("请先发送一条消息；空输入会复用上一条回复与提示词。")
            reply = last_assistant.get("content", "")
            prompt_text = last_assistant.get("prompt") or extract_prompt_from_reply(reply)
            history_json = _serialize_history(history, meta)
            report_lines.append("no_input=reuse_last")
            report = "\n".join(report_lines)
            ui = {
                "chat_history": [history_json],
                "reply": [reply],
                "prompt_text": [prompt_text],
                "report": [report],
            }
            return io.NodeOutput(reply, prompt_text, history_json, report, ui=ui)

        skill_to_use = None
        if effective_skill != AUTO and effective_skill:
            skill_to_use = resolve_skill(effective_skill)
            if meta.get("skill") != skill_to_use["id"]:
                if meta.get("skill"):
                    # switching skills restarts the conversation
                    history = []
                    report_lines.append("conversation_restarted=True")
                meta["skill"] = skill_to_use["id"]
                meta["skill_loaded"] = False
                meta["loaded_references"] = []
        elif effective_skill == AUTO:
            if meta.get("skill"):
                skill_to_use = resolve_skill(meta["skill"])
            else:
                skill_id = select_skill_with_api(
                    api_base, api_key, model, discover_skills(), user_text
                )
                skill_to_use = resolve_skill(skill_id)
                meta["skill"] = skill_to_use["id"]
                meta["skill_loaded"] = False
        first_load = skill_to_use is not None and not meta.get("skill_loaded")
        if skill_to_use is not None:
            report_lines.append(f"skill={skill_to_use['id']}")
            report_lines.append("skill_first_load=True" if first_load else "skill_reused=True")

        loaded_references = list(meta.get("loaded_references") or [])
        skill_state = _default_flow_state()
        reply = ""
        if first_load:
            if bool(auto_load_references) and skill_to_use is not None:
                loaded_references = list(skill_to_use["references"])
                report_lines.append("auto_references=True")
            for attempt in range(2):
                system_text = _build_system_text(
                    skill_to_use, loaded_references, effective_mode, media, duration
                )
                messages = []
                if system_text:
                    messages.append({"role": "system", "content": system_text})
                messages.extend(_history_to_api_messages(history, media, label_map))
                messages.append(
                    {
                        "role": "user",
                        "content": media_to_api_parts(user_text, media, label_map),
                    }
                )
                raw_reply = chat_completion(
                    api_base, api_key, model, messages, temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    reasoning_effort=(reasoning_effort if enable_thinking else None),
                    strip_thinking=(not bool(output_thinking)),
                    seed=seed_value,
                )
                reply, skill_state = _parse_skill_state(raw_reply)
                requested = [
                    item
                    for item in skill_state["load_references"]
                    if item in skill_to_use["references"] and item not in loaded_references
                ]
                if not requested or attempt == 1:
                    break
                loaded_references.extend(requested)
            meta["skill_loaded"] = True
            meta["loaded_references"] = list(loaded_references)
            report_lines.append(f"stage={skill_state['stage']}")
            report_lines.append(f"options={len(skill_state['options'])}")
            report_lines.append(f"final={bool(skill_state['final'])}")
            report_lines.append(f"loaded_references={len(loaded_references)}")
        else:
            system_text = _build_system_text(None, [], effective_mode, media, duration)
            messages = []
            if system_text:
                messages.append({"role": "system", "content": system_text})
            messages.extend(_history_to_api_messages(history, media, label_map))
            messages.append(
                {
                    "role": "user",
                    "content": media_to_api_parts(user_text, media, label_map),
                }
            )
            reply = chat_completion(
                api_base, api_key, model, messages, temperature=float(temperature),
                max_tokens=int(max_tokens),
                reasoning_effort=(reasoning_effort if enable_thinking else None),
                strip_thinking=(not bool(output_thinking)),
                seed=seed_value,
            )
        if not (reply or "").strip():
            if (prompt_mode or "auto").upper() == "AUTO":
                # The model bailed on the auto instruction: retry once with the
                # deterministic mode inferred from the connected ports.
                inferred = _infer_mode_from_media(media)
                report_lines.append("auto_empty_reply=True")
                system_text = _build_system_text(None, [], inferred, media, duration)
                fallback_messages = []
                if system_text:
                    fallback_messages.append({"role": "system", "content": system_text})
                fallback_messages.extend(_history_to_api_messages(history, media, label_map))
                fallback_messages.append(
                    {"role": "user", "content": media_to_api_parts(user_text, media, label_map)}
                )
                retry_reply = chat_completion(
                    api_base, api_key, model, fallback_messages, temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    reasoning_effort=(reasoning_effort if enable_thinking else None),
                    strip_thinking=(not bool(output_thinking)),
                    seed=seed_value,
                )
                if (retry_reply or "").strip():
                    reply = retry_reply
                    report_lines.append(f"auto_fallback_mode={inferred}")
            if not (reply or "").strip():
                raise ValueError(
                    "LLM returned an empty reply. Try a different model, enable "
                    "output_thinking, or pick an explicit prompt_mode."
                )
        extracted_prompt = extract_prompt_from_reply(reply) or reply
        history.append({"role": "user", "content": prompt_text, "media": tokens})
        history.append({"role": "assistant", "content": reply, "prompt": extracted_prompt, "media": []})
        history = _trim_history(history, int(max_history_rounds))
        history_json = _serialize_history(history, meta)
        prompt_text = extracted_prompt
        report_lines.append("api_call=done")

        report = "\n".join(report_lines)
        ui = {
            "chat_history": [history_json],
            "reply": [reply],
            "prompt_text": [prompt_text],
            "report": [report],
        }
        return io.NodeOutput(reply, prompt_text, history_json, report, ui=ui)


class MiniMaxH3MultimodalChatExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3MultimodalChat]


async def comfy_entrypoint():
    return MiniMaxH3MultimodalChatExtension()
