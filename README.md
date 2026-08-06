# ComfyUI MiniMax H3 Context IR

单个节点 `MiniMax H3 Context IR`：调用官方 H3-Context-IR API，把普通提示词（可带媒体）增强成结构化 H3 提示词，输出字符串。

## 安装

1. 复制本文件夹到 `ComfyUI/custom_nodes/`。
2. 注册登录 https://www.minimaxi.com/ 获取并设置 API Key：环境变量 `MINIMAX_API_KEY`，或节点内 `api_key` 输入框。
3. 重启 ComfyUI。节点出现在 **MiniMax ContextIR** 分类。

## 模式

| 模式 | 可用输入 | 说明 |
| --- | --- | --- |
| `t2va` | 仅 `text` | 文生视频；`ratio` 不能为 adaptive |
| `i2va` | `text` + `first_frame` / `last_frame` | 图生视频（首帧/末帧，可只接一个或都接） |
| `r2va` | `text` + `ref_image_0..` / `ref_video_0..`(+`ref_video_audio_0..`) / `ref_audio_0..` | 多模态参考生视频 |

## 输入（动态增长）

- `first_frame` / `last_frame`：固定单输入（各 ≤1）
- `ref_image_0..8`：初始显示 0–3，接满自动出现下一个，最多 9 张
- `ref_video_0..2`：初始 0，最多 3 段（帧序列，自动编码 mp4；需 av/OpenCV/imageio-ffmpeg）
- `ref_video_audio_0..2`：与同编号视频配对，最多 3 段
- `ref_audio_0..2`：初始 0，接满自动出现下一个，最多 3 段

模式与素材不匹配（如 t2va 接图、i2va 接参考、r2va 接首末帧）会在本地直接报错；数量超上限也会报错。

## 输出

`enhanced_prompt`（STRING）：增强后的 H3 提示词文本，可直接接官方节点（Image to Video / Reference to Video）的 `prompt` 输入。

## 注意

- 节点会阻塞到任务完成（最长 180 秒），失败时直接抛错并附上 API 响应。
- 参考视频编码后超过 20MB 会报错（Base64 请求体上限 64MB），请缩短片段或改用图片/音频参考。
