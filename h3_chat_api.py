"""External multimodal LLM API client, skill discovery, and H3 media token helpers."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

import torch

from .h3_context_ir import _audio_to_data_url, _image_to_data_url


# ---------------------------------------------------------------- skills

_PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
_LLAMA_TE_SKILLS = os.path.normpath(os.path.join(_PLUGIN_DIR, "..", "comfyUI-llama-TE", "skills"))


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as file:
        return file.read()


def _read_meta_value(skill_dir: str, key: str) -> str:
    path = os.path.join(skill_dir, "meta.yaml")
    if not os.path.isfile(path):
        return ""
    for line in _read_text(path).splitlines():
        match = re.match(rf"^{re.escape(key)}:\s*(.+?)\s*$", line)
        if match:
            return match.group(1).strip().strip("\"'")
    return ""


def _parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    values = {}
    for line in text[3:end].splitlines():
        match = re.match(r"^([\w-]+):\s*(.+?)\s*$", line)
        if match:
            values[match.group(1)] = match.group(2).strip().strip("\"'")
    return values


def _list_references(skill_dir: str) -> list[str]:
    reference_dir = os.path.join(skill_dir, "references")
    if not os.path.isdir(reference_dir):
        return []
    files = []
    for root, _, names in os.walk(reference_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() not in (".md", ".txt", ".yaml", ".yml", ".json"):
                continue
            files.append(os.path.relpath(os.path.join(root, name), skill_dir).replace("\\", "/"))
    return sorted(files)


def _discover_in(skill_root: str) -> list[dict]:
    if not os.path.isdir(skill_root):
        return []
    skills = []
    for skill_id in sorted(os.listdir(skill_root)):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", skill_id):
            continue
        skill_dir = os.path.join(skill_root, skill_id)
        if not os.path.isdir(skill_dir):
            continue
        cn_path = os.path.join(skill_dir, "SKILL.cn.md")
        en_path = os.path.join(skill_dir, "SKILL.md")
        if not os.path.isfile(cn_path) and not os.path.isfile(en_path):
            continue
        content_path = cn_path if os.path.isfile(cn_path) else en_path
        metadata = _parse_frontmatter(_read_text(content_path))
        name = (
            _read_meta_value(skill_dir, "display-name-zh")
            or metadata.get("name")
            or skill_id
        )
        description = (
            _read_meta_value(skill_dir, "summary-cn")
            or metadata.get("description")
            or ""
        )
        skills.append(
            {
                "id": skill_id,
                "root": os.path.realpath(skill_root),
                "name": str(name),
                "label": f"{name} [{skill_id}]" if str(name) != skill_id else skill_id,
                "description": str(description),
                "skill_file": os.path.basename(content_path),
                "references": _list_references(skill_dir),
            }
        )
    return skills


def discover_skills() -> list[dict]:
    roots = [os.path.join(_PLUGIN_DIR, "skills")]
    candidates = [_LLAMA_TE_SKILLS]
    try:
        import folder_paths

        for base in folder_paths.get_folder_paths("custom_nodes"):
            candidates.append(os.path.join(base, "comfyUI-llama-TE", "skills"))
    except Exception:  # noqa: BLE001 - fall back to relative paths
        pass
    for candidate in candidates:
        if os.path.isdir(candidate):
            roots.append(candidate)
    seen: set[str] = set()
    skills = []
    for root in roots:
        for skill in _discover_in(root):
            if skill["id"] in seen:
                continue
            seen.add(skill["id"])
            skills.append(skill)
    return sorted(skills, key=lambda item: item["label"])


def find_skill(skill_id: str) -> dict | None:
    return next((skill for skill in discover_skills() if skill["id"] == skill_id), None)


def load_skill_text(skill: dict) -> str:
    return _read_text(os.path.join(skill["root"], skill["id"], skill["skill_file"]))


def load_skill_reference(skill: dict, relative_path: str) -> str:
    normalized = str(relative_path or "").replace("\\", "/").strip("/")
    if normalized not in skill["references"]:
        raise ValueError(f"Skill reference does not exist: {relative_path}")
    skill_root = os.path.realpath(os.path.join(skill["root"], skill["id"]))
    path = os.path.realpath(os.path.join(skill_root, normalized))
    if os.path.commonpath([skill_root, path]) != skill_root:
        raise ValueError("Skill reference path escapes the skill directory")
    return _read_text(path)


def resolve_skill(value: str) -> dict | None:
    """Resolve a combo value (label, id, or '无') to a skill dict."""
    if not value or value in ("无", "auto"):
        return None
    skills = discover_skills()
    for skill in skills:
        if skill["label"] == value or skill["id"] == value:
            return skill
    for skill in skills:
        if f"[{skill['id']}]" in value or skill["id"] in value:
            return skill
    raise ValueError(f"Skill not found: {value}")


def skill_combo_options(include_auto: bool = False) -> list[str]:
    options = []
    if include_auto:
        options.append("auto")
    options.append("无")
    options.extend(skill["label"] for skill in discover_skills())
    return options


# ---------------------------------------------------------------- H3 media tokens

TOKEN_RE = re.compile(
    r"@(first_frame|last_frame|ref_image_\d+|ref_video_audio_\d+|ref_video_\d+|ref_audio_\d+)"
)

TOKEN_ORDER = [
    "first_frame",
    "last_frame",
    "ref_image_",
    "ref_video_audio_",
    "ref_video_",
    "ref_audio_",
]


def collect_media_inputs(
    first_frame=None,
    last_frame=None,
    ref_images=None,
    ref_videos=None,
    ref_video_audios=None,
    ref_audios=None,
) -> dict[str, object]:
    """Return token -> media value, preserving H3 presentation order."""
    media: dict[str, object] = {}

    def add(key: str, value) -> None:
        if value is not None:
            media[key] = value

    add("first_frame", first_frame)
    add("last_frame", last_frame)
    for key, value in sorted((ref_images or {}).items()):
        if value is not None:
            add(key, value)
    # paired soundtracks are attached to their videos later; keep them available
    for key, value in sorted((ref_video_audios or {}).items()):
        if value is not None:
            add(key, value)
    for key, value in sorted((ref_videos or {}).items()):
        if value is not None:
            add(key, value)
    for key, value in sorted((ref_audios or {}).items()):
        if value is not None:
            add(key, value)
    return media


def token_sort_key(token: str) -> tuple[int, int]:
    for index, prefix in enumerate(TOKEN_ORDER):
        if token == prefix or token.startswith(prefix):
            suffix = token[len(prefix):]
            return (index, int(suffix) if suffix.isdigit() else 0)
    return (len(TOKEN_ORDER), 0)


def build_label_map(media: dict[str, object]) -> dict[str, str]:
    """Map @tokens to official H3 labels for all connected media.

    Order: pictures (first, last, ref images) -> <Picture i>;
    videos -> <Video k>; audio (soundtracks + standalone) -> <Audio j>.
    """
    picture_tokens = [t for t in ("first_frame", "last_frame") if t in media] + sorted(
        [t for t in media if t.startswith("ref_image_")], key=token_sort_key
    )
    video_tokens = sorted(
        [t for t in media if t.startswith("ref_video_") and not t.startswith("ref_video_audio_")],
        key=token_sort_key,
    )
    audio_tokens = sorted(
        [t for t in media if t.startswith("ref_video_audio_") or t.startswith("ref_audio_")],
        key=token_sort_key,
    )
    label_map = {}
    for index, token in enumerate(picture_tokens, 1):
        label_map[token] = f"<Picture {index}>"
    for index, token in enumerate(video_tokens, 1):
        label_map[token] = f"<Video {index}>"
    for index, token in enumerate(audio_tokens, 1):
        label_map[token] = f"<Audio {index}>"
    return label_map


def referenced_tokens(prompt: str) -> list[str]:
    return [match.group(1) for match in TOKEN_RE.finditer(prompt or "")]


def render_prompt_text(prompt: str, label_map: dict[str, str]) -> str:
    def replace(match: re.Match) -> str:
        token = match.group(1)
        return label_map.get(token, match.group(0))

    return TOKEN_RE.sub(replace, prompt or "").strip()


# ---------------------------------------------------------------- media -> API parts

def _sample_indices(frame_count: int, max_samples: int = 8) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= max_samples:
        return list(range(frame_count))
    return sorted({round(i * (frame_count - 1) / (max_samples - 1)) for i in range(max_samples)})


def media_to_api_parts(prompt: str, media: dict[str, object], label_map: dict[str, str], max_video_frames: int = 8) -> list[dict]:
    """Build OpenAI-style content parts for referenced media."""
    parts = [{"type": "text", "text": render_prompt_text(prompt, label_map)}]
    for token in referenced_tokens(prompt):
        value = media.get(token)
        if value is None:
            raise ValueError(
                f"{token} is not connected; connect it to the node or remove it from the message."
            )
        label = label_map.get(token, token)
        if isinstance(value, torch.Tensor):
            frames = value.detach().cpu()
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            count = int(frames.shape[0])
            indices = _sample_indices(count, max_video_frames)
            if len(indices) == 1:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(frames[indices[0]].unsqueeze(0))},
                    }
                )
            else:
                for order, index in enumerate(indices, 1):
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_to_data_url(frames[index].unsqueeze(0)),
                            },
                        }
                    )
                parts.append(
                    {
                        "type": "text",
                        "text": f"({label}: {len(indices)} sampled frames of {count})",
                    }
                )
        else:
            audio_url = _audio_to_data_url(value)
            audio_format = "wav"
            parts.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_url.split(",", 1)[1], "format": audio_format},
                }
            )
    return parts


# ---------------------------------------------------------------- OpenAI-compatible client

def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    reasoning_effort: str | None = None,
    strip_thinking: bool = True,
    seed: int | None = None,
    timeout: int = 180,
    retries: int = 3,
) -> str:
    if not base_url or not base_url.strip():
        raise ValueError("api_base is required for external LLM calls.")
    if not api_key or not api_key.strip():
        raise ValueError("api_key is required for external LLM calls.")
    url = base_url.rstrip("/") + "/chat/completions"
    max_tokens = int(min(128000, max(1, int(max_tokens))))
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning_effort"] = reasoning_effort
    if seed is not None:
        payload["seed"] = int(seed)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": "Bearer " + api_key.strip(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    def _make_request(payload: dict):
        return urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer " + api_key.strip(),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    def _parse(data: dict) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Unexpected LLM API response: {json.dumps(data, ensure_ascii=False)[:500]}"
            ) from exc
        if isinstance(content, list):
            text = "\n".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        else:
            text = str(content)
        text = text.strip()
        if strip_thinking:
            text = strip_think_blocks(text)
        return text

    content_types = []
    text_chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text_chars += len(content)
            if "text" not in content_types:
                content_types.append("text")
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    part_type = str(part.get("type") or "")
                    if part_type and part_type not in content_types:
                        content_types.append(part_type)
                    if part_type == "text":
                        text_chars += len(str(part.get("text") or ""))
    request_summary = (
        f"model={model}, reasoning_effort={reasoning_effort or 'none'}, "
        f"messages={len(messages)}, content_types={content_types or ['text']}, text_chars={text_chars}"
    )

    attempts = max(1, int(retries))
    last_code = None
    last_detail = ""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(_make_request(payload), timeout=timeout) as resp:
                return _parse(json.loads(resp.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            last_code = exc.code
            last_detail = exc.read().decode("utf-8", errors="replace")[:800]
            if exc.code >= 500 and attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        except urllib.error.URLError as exc:
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"LLM API network error on {url}: {exc.reason}") from exc

    # Diagnostic fallback: some relays 500 on unknown params like reasoning_effort.
    if last_code is not None and last_code >= 500 and "reasoning_effort" in payload:
        payload.pop("reasoning_effort")
        try:
            with urllib.request.urlopen(_make_request(payload), timeout=timeout) as resp:
                return _parse(json.loads(resp.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            last_code = exc.code
            last_detail = exc.read().decode("utf-8", errors="replace")[:800]
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API network error on {url}: {exc.reason}") from exc

    if last_code is None:
        raise RuntimeError(f"LLM API request failed on {url} without a response.")
    hint = ""
    if last_code >= 500:
        hint = (
            " (server 5xx: transient failure, or the model/relay rejects "
            "reasoning_effort / multimodal content. Request summary: "
            f"{request_summary} - try disabling enable_thinking or media inputs)"
        )
    raise RuntimeError(f"LLM API HTTP {last_code} on {url}: {last_detail}{hint}")

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unexpected LLM API response: {json.dumps(data, ensure_ascii=False)[:500]}") from exc
    if isinstance(content, list):
        text = "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    else:
        text = str(content)
    text = text.strip()
    if strip_thinking:
        text = strip_think_blocks(text)
    return text


def strip_think_blocks(text: str) -> str:
    """Remove common reasoning/think blocks from an LLM reply."""
    import re

    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"^.*?</think>\s*", "", cleaned, count=1, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned


_PROMPT_PREFIX_FILLER = re.compile(
    r"^(好的|好的，|当然|当然可以|没问题|可以|为您|为你|以下是|下面是|这是|生成结果|"
    r"提示词如下|需要的提示词|提示词|请使用|prompt|Prompt|Here is|Here's|Sure|"
    r"Certainly|Of course|The prompt is)[：:，,]?\s*$",
    re.IGNORECASE,
)
_PROMPT_TRAILING_FILLER = re.compile(
    r"^(希望|如果|如需|以上|请问|还有|祝|谢谢|欢迎|有问题|注意|温馨提示|"
    r"你可以|请直接|完整提示词|最终提示词)[^，。\n]{0,60}$",
    re.IGNORECASE,
)

_H3_SECTION_HEADER = re.compile(
    r"(?:subject_definitions|integrated_multimodal_description|summary|"
    r"retention_analysis|detailed_description|overall_soundscape|"
    r"non_diegetic_music|shot_plan|storyboard)\s*[:：]",
    re.IGNORECASE,
)


def extract_prompt_from_reply(reply: str) -> str:
    """Extract the H3 prompt from an assistant reply, removing conversational filler."""
    if not reply or not reply.strip():
        return ""
    text = re.sub(
        r"<mmx_skill_state>\s*\{.*?\}\s*</mmx_skill_state>", "", reply, flags=re.DOTALL
    ).strip()
    # 1) fenced code block is the most reliable marker
    fence = re.search(r"```[^\n`]*\n([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    # 1b) structured H3 prompt: cut everything before the first section header,
    #     so conversational chatter in front of the prompt never leaks through.
    headers = list(_H3_SECTION_HEADER.finditer(text))
    if headers:
        pick = next(
            (m for m in headers if m.start() == 0 or text[m.start() - 1] == "\n"), None
        )
        if pick is None:
            pick = next(
                (m for m in headers if m.start() > 0 and text[m.start() - 1] in ":："), None
            )
        if pick is None:
            pick = headers[0]
        text = text[pick.start():].lstrip("\n ")
    # 2) remove leading filler lines ("您需要的提示词如下：" etc.)
    lines = text.splitlines()
    while lines:
        stripped = lines[0].strip()
        if not stripped:
            lines.pop(0)
            continue
        if _PROMPT_PREFIX_FILLER.match(stripped):
            lines.pop(0)
            continue
        if len(stripped) <= 24 and (stripped.endswith("：") or stripped.endswith(":") or stripped.endswith("如下")):
            if not _H3_SECTION_HEADER.match(stripped):
                lines.pop(0)
                continue
        break
    # 3) remove trailing filler lines
    while lines:
        stripped = lines[-1].strip()
        if not stripped:
            lines.pop()
            continue
        if _PROMPT_TRAILING_FILLER.match(stripped):
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()


def select_skill_with_api(base_url: str, api_key: str, model: str, skills: list[dict], user_text: str) -> str:
    if not skills:
        raise ValueError("No skills found; place SKILL.md folders in the plugin skills directory.")
    catalogue = "\n".join(
        f'- {item["id"]}: {item["name"]} - {item["description"][:300]}' for item in skills
    )
    messages = [
        {
            "role": "system",
            "content": "根据用户任务选择唯一最匹配的 Skill。只输出 Skill ID，不解释，不添加标点。",
        },
        {"role": "user", "content": f"可用 Skills：\n{catalogue}\n\n用户任务：\n{user_text}"},
    ]
    selected = chat_completion(
        base_url, api_key, model, messages, temperature=0.0, max_tokens=80
    ).strip().strip("`'\".,，。")
    valid = {item["id"] for item in skills}
    if selected in valid:
        return selected
    for skill_id in valid:
        if skill_id in selected:
            return skill_id
    raise ValueError(
        f"Auto skill selection failed; model returned: {selected[:120]}. Please select a skill manually."
    )
