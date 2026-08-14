# ComfyUI-MiniMax-ContextIR

MiniMax H3 视频生成全家桶：一个 ComfyUI 插件，八个节点，覆盖
“提示词优化 → 素材加载 → 条件构建 → 音频锁定 → 采样输出”的完整 H3 工作流。
既可直接配合官方 MiniMax H3 本地模型（FLOW_AV / fl2va）使用，也可调用
H3-Context-IR 云端 API 自动把普通提示词优化成结构化 H3 提示词。

## 功能总览

- **MiniMax H3 Unified to Video**：本地 H3 Conditioning（融合官方 Image to Video /
  Reference to Video，支持 t2va / i2va / fl2va / l2va / ref2va / hybrid，自动识别任务类型）。
- **MiniMax H3 Context IR**：独立调用 H3-Context-IR API，把普通提示词 + 媒体优化成
  结构化 H3 提示词（t2va / i2va / r2va），输出字符串。
- **MiniMax H3 Multimodal Chat**：多轮对话节点，可加载内置的 MiniMax 官方 Skill、调用任意
  OpenAI 兼容多模态 LLM API、通过 @ 引用 H3 媒体输入，并支持自动化外部文本输入。
- **MiniMax H3 Audio Lock**：把参考音频锁进 AV latent，成片音频 100% 保持源音频
  （对口型 / 音乐 MV 场景必备）。
- **MiniMax H3 Media Loader (Fant)**：参考 Fantastic MiniMaxH3 PromptBuilder 并修改优化的
  媒体加载器（拖拽上传、预览、排序、音轨拆分、裁剪/抽帧、中英双语界面、预设），
  输出 `references` 素材包。
- **MiniMax H3 Reference Splitter**：把 `references` 素材包拆成独立的图/视频/音频槽位。
- **MiniMax H3 Resolution Selector**：宽高比 + 官方“宽×高”预设 → H3 宽高；
  切换比例时“宽×高”选项自动跟随（原生 DynamicCombo）。
- **MiniMax H3 Concat AV Latent**：把独立的视频/音频 latent 合并成 H3 采样器所需的
  联合 NestedTensor AV latent（参考 PT_H3ConcatAVLatent，主要用于视频的二次采样放大）。

## 特性

- 全部使用 ComfyUI 新版 `ComfyExtension` / V3 io API 注册，与官方及其他 H3 插件可共存；
- 提示词自动规范化媒体标签 `<Picture i>` / `<Video k>` / `<Audio j>`，
  引用未连接的媒体会给出明确警告而不是静默失败；
- 时长自动按官方 17n+5 帧网格对齐：`max(5, round(duration*fps)) + (5 - ... % 17) % 17`；
- 媒体加载器全界面中/英双语可切换；聊天节点支持 @ 媒体引用、多历史对话管理、
  Skill 首轮注入（远端 API 无需访问本地文件）。

## 兼容性

- ComfyUI ≥ 0.31（新式 `ComfyExtension`；不支持旧式 `NODE_CLASS_MAPPINGS` 注册方式）
- MiniMax H3 官方模型：UNET + Qwen3-VL CLIP + 视频 VAE + 音频 VAE
- Python 3.10+；媒体加载器需要 PyAV 或 ffmpeg 之一（音频解码可回退 ComfyUI LoadAudio）

## 安装

### 方式一：git clone（推荐）

```bash
git clone https://github.com/oufeixinxinren/ComfyUI-MiniMax-ContextIR.git
```

把克隆出的 `ComfyUI-MiniMax-ContextIR` 文件夹放进 `ComfyUI/custom_nodes/`。

### 方式二：手动复制

把整个项目文件夹复制到 `ComfyUI/custom_nodes/` 即可。

安装后**重启 ComfyUI**。节点分类：

- `model/conditioning/minimax/unified`：Unified / Audio Lock / Concat AV Latent /
  Resolution Selector 等
- `MiniMax ContextIR`：Context IR
- `MiniMaxH3 Media/Fant`：Media Loader (Fant) / Reference Splitter

## 快速开始

1. 用 **Media Loader (Fant)** 上传参考图/视频/音频，点击 **+ Native-output splitter**
   自动展开 Reference Splitter 槽位；
2. 接入 **Unified to Video**（`mode=auto` 按已连接输入自动判断任务类型），填写提示词；
3. `positive` → BasicGuider；`av_latent` → Audio Lock（需要锁定源音频时）→
   SamplerCustomAdvanced；
4. 采样结果 → VAEDecode + VAEDecodeAudio 保存；需要二次放大采样时用
   Concat AV Latent 把重编码的视频/音频 latent 再喂给采样器；
5. 需要提示词优化时，用 **Multimodal Chat** 或 **Context IR** 节点生成结构化 H3 提示词。

---

## 节点 1：MiniMax H3 Unified to Video

本地 H3 条件构建，`mode=auto` 时按已连接输入自动判断：

| 已连接输入 | 自动任务 |
| --- | --- |
| 无 | t2va |
| 仅首帧 | i2va |
| 首帧 + 末帧 | fl2va |
| 仅末帧 | l2va |
| 参考媒体（图/视频/音频） | ref2va |
| 关键帧 + 参考媒体 | hybrid |

### 输入

| 输入 | 说明 |
| --- | --- |
| `clip` | MiniMax H3 Qwen3-VL CLIP |
| `video_vae` | MiniMax H3 视频 VAE |
| `audio_vae` | 可选；参考视频音轨/参考音频时需要 |
| `prompt` | 输入提示词（用 `<Picture i>` / `<Video k>` / `<Audio j>` 引用已连接媒体） |
| `mode` | auto / t2va / fl2va / ref2va |
| `width` / `height` | 本地画布（32 倍数，上限 1920x1088） |
| `duration` | 场景时长（秒，1–15，浮点） |
| `fps` | 帧率（1–60，默认 24），仅用于时长 → 帧数换算 |
| `ref_image_size` | 本地参考图缩放：match / max |
| `first_frame` / `last_frame` | 关键帧 |
| `ref_images` / `ref_videos` / `ref_video_audios` / `ref_audios` | 参考媒体（接满自动出现下一路） |

### 输出

- `positive`：CONDITIONING，接 BasicGuider
- `av_latent`：联合视频+音频 latent，接 SamplerCustomAdvanced.latent_image
- `conditioned_prompt`：规范化媒体标签后的提示词
- `media_map_json`：媒体编号与来源映射
- `report`：本地任务、帧数、媒体数量等诊断信息

### 注意

- 参考视频建议 48–360 帧（2–15 秒，24fps）。
- 帧数按官方公式自动计算并对齐到 17n+5：
  `max(5, round(duration*fps)) + (5 - (max(5, round(duration*fps)) % 17)) % 17`。
- 显式模式会屏蔽（忽略）不属于该模式的输入，而不是报错；只有缺失必需输入才报错：
  - `mode=t2va`：忽略全部已连接媒体；
  - `mode=fl2va`：忽略参考媒体，按已连接关键帧自动映射（仅首帧 → i2va，
    仅末帧 → l2va，首尾帧都有 → fl2va；无任何关键帧 → 报错）。
- `mode=ref2va` 时首尾帧与参考媒体可同时输入（内部按 hybrid 处理）；
  未接任何参考媒体 → 报错。

---

## 节点 2：MiniMax H3 Context IR

独立调用 H3-Context-IR API：输入普通提示词（+ 关键帧或参考媒体），返回优化后的
结构化 H3 提示词字符串。

### 输入

| 输入 | 说明 |
| --- | --- |
| `mode` | t2va（纯文本）/ i2va（首尾帧）/ r2va（参考媒体，需至少一路） |
| `text` | 原始提示词 |
| `duration` | 浮点，1–15 秒（自动钳制并四舍五入） |
| `ratio` | adaptive / 21:9 / 16:9 / 4:3 / 1:1 / 3:4 / 9:16（t2va 不能 adaptive） |
| `first_frame` / `last_frame` | 关键帧（i2va） |
| `ref_images` / `ref_videos` / `ref_video_audios` / `ref_audios` | 参考媒体（r2va；与 Unified 节点相同，每个类别先显示一路，接入后自动增加下一路） |
| `api_key` / `base_url` / `callback_url` | API 配置（key 必填或设 `MINIMAX_API_KEY`） |

### 输出

- `enhanced_prompt`：优化后的 H3 提示词字符串

### 注意

- 在 https://www.minimaxi.com/ 注册登录后获取 API Key；可通过节点内 `api_key`
  输入框或环境变量 `MINIMAX_API_KEY` 设置。
- 会阻塞到 API 任务完成（最长 180 秒），失败直接抛错。
- r2va 与首尾帧在官方 API 中互斥：r2va 只能带参考媒体，关键帧请用 i2va。

---

## 节点 3：MiniMax H3 Multimodal Chat

多轮对话节点，支持 Skill 加载与外部多模态 API：

- **Skill 加载**：直接在节点的 `skill` 下拉框选择即可生效（自动发现本插件 `skills/`
  下内置的 MiniMax 官方 Skill，含 `h3-prompt-writing`；
  `skill=auto` 时由外部 LLM 根据首个任务自动选择）。选择后 Skill 内容会注入系统提示词，
  并启用阶段式执行协议：模型按阶段推进、可请求按需加载 references、
  通过 `<mmx_skill_state>` 状态标记返回阶段/选项/final。
- **外部多模态 API**：任意 OpenAI 兼容的 `/chat/completions` 接口
  （`api_base` + `api_key` + `model`）。
- **H3 媒体端口**：与 Unified 节点一致的首尾帧、参考图/视频/音频输入。
- **@ 引用**：在消息输入框输入 `@` 弹出媒体引用菜单，选择后插入
  `@first_frame` / `@last_frame` / `@ref_image_N` / `@ref_video_N` /
  `@ref_video_audio_N` / `@ref_audio_N`；后端会把这些引用解析为
  `<Picture i>` / `<Video k>` / `<Audio j>` 标签并随 API 请求发送媒体内容
  （视频自动抽帧为图片）。
- **仅输出提示词**：`prompt_only` 开启时不调用 API，直接输出组装好的提示词文本。
- **多对话管理**：左侧“最近聊天”栏支持新建、切换、删除对话（上限 20 个），
  支持“清空全部”；**双击会话按钮可重命名**；对话列表随工作流保存，重启后仍可继续。
- **Skill 状态可见**：聊天区顶部显示当前 Skill 与阶段（流程条），
  首轮加载后显示“已加载”。

### 输入

| 输入 | 说明 |
| --- | --- |
| `prompt` | 用户消息（聊天窗口内填写，支持 @ 引用） |
| `input_string` | 外部文本输入端口；连接外部文本输出后优先作为消息输入（覆盖聊天窗口手输内容），方便自动化分段生成 |
| `chat_history` / `request_id` | 由前端聊天窗口自动维护 |
| `skill` | auto 或具体 Skill |
| `prompt_mode` | 提示词生成模式：auto / T2VA / I2VA / FL2VA / L2VA / REF2VA；auto 时由模型根据已连接端口和需求自动判断；显式模式会屏蔽不属于该模式的媒体输入：T2VA 忽略全部媒体；I2VA / L2VA / FL2VA 忽略参考媒体（缺必需关键帧时报错）；REF2VA 允许首/尾关键帧与参考媒体并用（无任何参考媒体时报错） |
| `api_base` / `api_key` / `model` | OpenAI 兼容 API 配置 |
| `temperature` | 采样温度 |
| `enable_thinking` / `reasoning_effort` | 是否请求模型思考（reasons），及思考强度（需模型/服务商支持） |
| `output_thinking` | 是否在回复中保留 `<think>` 思考块；关闭则自动清理 |
| `max_tokens` | 最大生成长度（0–128000，发送前自动钳制；实际还受模型限制） |
| `max_history_rounds` | 客户端历史裁剪轮数（默认 1000）；真正的限制是模型上下文窗口 |
| `auto_load_references` | 开关（默认开）；首轮加载 Skill 时自动把本地 reference 文件内容注入系统提示词，远端 API 无需访问本地文件 |
| `first_frame` / `last_frame` / `ref_images` / `ref_videos` / `ref_video_audios` / `ref_audios` | H3 媒体参考输入 |

### 输出

- `reply`：助手回复完整文本
- `prompt_text`：自动提取并清洗后的 H3 提示词（过滤“您需要的提示词如下：”等套话，
  优先取代码块内容）；空输入时复用上一条结果，不再调用 API
- `chat_history`：更新后的对话历史 JSON
- `report`：prompt_only 状态、媒体/引用数量、skill、阶段/选项/final、已加载 references、api_call 状态

### 注意

- Skill 只在首轮加载，后续轮次不重复上传；**切换 Skill 会重启对话**（清空历史后重新加载）。
- `input_string` 有文本时，节点一旦执行就会自动调用 API 并输出新的 `prompt_text`（无需点击发送，
  且每次运行都会调用，不会命中缓存）；`input_string` 为空时，只有点击发送（或运行前
  `prompt` 已有内容）才会发送，否则仅复用上一条输出。
- 点击清空后，下一次对话也会重新加载当前选择的 Skill。
- 新建对话会在第一条消息时重新加载当前 Skill；切换对话则沿用该对话已加载的 Skill 状态。
- 未连接的 @ 引用会直接报错。
- 外部 API 需支持 OpenAI 兼容的多模态消息格式（`image_url` / `input_audio`）。
- 上下文长度无法真正“无限制”：是否够长取决于所调用模型的上下文窗口；
  客户端默认不再主动截断（1000 轮上限），超出窗口时 API 会返回上下文超限错误。
- Skill 的 `references/` 是本地文件，远端 API 读不到；节点会在首轮把文件内容读出来
  注入系统提示词（`auto_load_references` 默认开启）。文件较多/较大时可关闭该开关，
  改由模型按需请求加载。

---

## 节点 4：MiniMax H3 Audio Lock

H3 的 `ref_audio` 本质上只是“参考”：模型会按音色、节奏和内容**重新生成**一条音频，
提示词写得再明确（`fully_copy` / `audio reuse` 等）也无法保证原样输出。要让成片音频
就是接入的参考音频（比如对口型、音乐 MV），需要在采样前把源音频“锁”进 AV latent。

### 输入

| 输入 | 说明 |
| --- | --- |
| `av_latent` | 来自 Unified / Context IR 节点的联合 AV latent |
| `audio` | 要锁定的源音频（通常与 `ref_audio` 接同一文件） |
| `audio_vae` | MiniMax H3 音频 VAE |
| `mode` | `lock`（默认）：音频保持原样，只生成画面；`remix`：按 `strength` 重绘音频 |
| `strength` | remix 模式的去噪强度（0–1，lock 时忽略） |

### 输出

- `av_latent`：替换好源音频并带音频 mask 的 latent，接 `SamplerCustomAdvanced.latent_image`
- `audio`：原始源音频透传，用于最终合成时精确混流（VAE 编解码有损，成品建议用它）
- `report`：锁定模式、音频 latent 长度等诊断信息

### 用法

```
Unified/Context IR 节点
  ├─ positive ──> BasicGuider
  └─ Latent ──> MiniMax H3 Audio Lock ──> SamplerCustomAdvanced.latent_image
                  (audio 接 LoadAudio，audio_vae 接同一个音频 VAE)
保存视频时用 Audio Lock 的 audio 输出，而不是解码 latent 里的生成音频。
```

`lock` 模式下视频会尽量与源音频口型/节拍同步，音频 100% 保持为源音频；
`remix` 模式适合“保留节奏与语义、但想换音色”的场景。

---

## 节点 5/6：MiniMax H3 Media Loader (Fant) + Reference Splitter

参考并重构自 [ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder)
的媒体加载器（MIT License，Copyright (c) 2026 Adudeguyman）。节点 id 为
`MiniMaxH3MediaLoaderFantastic`，在保留原版媒体加载核心能力的基础上进行了修改与优化：

- 拖拽 / 文件选择上传图片、视频、音频，缩略图与播放预览；
- 拖动素材直接交换顺序（无排序按钮）、启停、移除；视频音轨可“配对”或“独立”拆分；
- 缩略图显示像素尺寸与比例角标（16:9、4:3、9:16 …，近似值带 `≈`）；
- 裁剪 / 抽帧编辑器：不改原文件即可裁剪视频时长（`trim`）、拖拽裁剪框（`crop`），
  并可直接从视频抽出一帧作为图片参考；图片也有 ✂ 按钮，可按 1:1 / 4:3 / 3:2 /
  16:9 / 9:16 / 3:4 / 2:3 / 21:9 或自由比例裁剪；
- 面板顶部有 **中 / EN** 语言切换按钮，一键切换全界面中英文（含按钮、提示、
  裁剪弹窗），选择会保存在浏览器本地；
- 2–15s 片段与每类 15s 总量限制提示（前端），最多 9 图 / 3 视频 / 3 音轨 / 3 音频；
- 媒体预设（presets）保存、加载、删除；
- 输出单路 `references`（H3_REFS 素材包）；点击面板上的 **+ Native-output splitter**
  按钮会自动创建并连接 **MiniMax H3 Reference Splitter**，展开
  `picture_1–9`、`video_1–3`、`video_audio_1–3`、`audio_1–3` 独立槽位，
  再接入官方的 Reference to Video 或本插件的 Unified 节点。

本节点不包含原插件的 Prompt Builder / 提示词库；需要提示词编辑请安装原插件。

### HTTP 接口

- `POST /minimax_h3/upload`：接收一个媒体文件并返回元数据（kind、duration、has_audio、宽高）。
- `GET /minimax_h3/capabilities`：返回 av / ffmpeg / ffprobe 解码能力。
- `POST /minimax_h3/probe`：按 `{"file": ...}` 探测时长、音轨、宽高。
- `GET/POST /minimax_h3/presets[/save|load|delete]`：媒体预设的保存/加载/删除。

---

## 节点 7：MiniMax H3 Resolution Selector

与官方 Resolution Selector 类似，但用官方“宽×高”预设替代百万像素滑杆：

- `aspect_ratio`：官方 8 种宽高比（1:1 / 2:3 / 3:2 / 3:4 / 4:3 / 9:16 / 16:9 / 21:9）。
- `size`：只显示当前比例对应的“宽×高”预设，数值沿用官方参考表风格（16:9 使用
  官方尺寸表 608×352 … 1920×1088，其他比例按同一像素面积换算），宽高均为 32 倍数；
  比例与官方参考一致，属于“近似比例”（如 1344×768 对应 16:9、768×1376 对应 9:16）。
- 包含接近常见分辨率的数值：16:9 的 864×480（480p）、1280×736（接近 720p）、
  1344×768（官方原生 768p）；9:16 的 480×864（480p 竖屏）、736×1280（720p 竖屏类）、
  768×1376 等。
- `16:9` 位于 `9:16` 下方；切换比例时下拉选项自动跟随（原生 DynamicCombo，
  无需前端脚本）。

输出：`width`、`height`（INT），以及 `size`（字符串，如 `1344×768`）。

---

## 节点 8：MiniMax H3 Concat AV Latent

参考自 ComfyUI-PT_H3ConcatAVLatent 的 `PT_H3ConcatAVLatent`，转换为插件的新 API
注册（节点 id `MiniMaxH3ConcatAVLatent`，与原插件可共存）。该节点主要用于
**视频的二次采样放大**：把已生成的视频/音频重新编码成 latent，合并后以更高分辨率
或更低步数再次采样（放大 / 重绘）。

- `video_latent`：视频 latent，形状 `[B, 24, T, H/16, W/16]`
- `audio_latent`：音频 latent，形状 `[B, 32, 2, T_audio]`
- 输出 `av_latent`：合并后的联合 `NestedTensor`，直接接 H3 采样器的 latent 输入，
  实现二次采样放大 / 重绘。

---

## 使用教程

更多使用教程（B 站）：https://space.bilibili.com/176339505
