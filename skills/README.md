# Skills

把 Codex 风格的 Skill（`SKILL.md` / `SKILL.cn.md`，可选 `meta.yaml` 和 `references/`）放到这里，
`MiniMax H3 Multimodal Chat` 节点即可在 `skill` 下拉框中加载。

本插件已内置官方 `h3-prompt-writing` skill：

- `h3-prompt-writing/SKILL.md` — MiniMax H3 视频提示词编写工作流
- `h3-prompt-writing/references/base-en.txt` — 文生视频 / 关键帧模式官方指南
- `h3-prompt-writing/references/ref-en.txt` — 全参考（Ref2VA）重写官方指南
- `h3-prompt-writing/meta.yaml` 与 `agents/openai.yaml` — 节点显示与接口配置

插件同时会自动发现 `custom_nodes/comfyUI-llama-TE/skills` 下的 Skill。
