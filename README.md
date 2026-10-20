# 智能人声音效台

[GitHub 仓库](https://github.com/QQ-D/-)

数字信号处理综合实验，Python 桌面版。界面按已确认的 [v2 设计](design/界面设计说明.md) 逐步实现。

## 当前已实现：第一阶段

- WAV 导入，显示真实采样率、时长及源声道数；立体声转单声道。
- 同一原声缓存的真实波形、幅度频谱、语谱图。频谱分析最多取前 8 秒，图标题显示实际时长；语谱图覆盖整段，长音频均匀抽帧显示。
- 输出设备选择，原声播放、暂停、继续、停止、进度、末尾重播；设备不可用时显示明确错误。
- 原声保存为 PCM16 WAV，保持音量，不做隐式归一化。
- 后台导入/分析与保存；损坏文件不覆盖已有原声；文件任务结束后安全关闭窗口。

**尚未实现**：麦克风录音、底噪采集、谱减去噪、人声增强、均衡、音效预设及全部选做功能。界面保留入口并置灰；右列图形保持待实现状态，不显示虚构处理结果。

## 运行

已盘点本机环境，选择 Conda base 的 Python 3.13.5 + 已有 PyQt5、Matplotlib、NumPy、SciPy。详见 [环境检查](docs/环境检查.md)。没有修改 base 或其他既有环境。

当前项目已经创建 `.venv-run`，可以直接启动：

```bash
bash run.sh
```

先用附带合成测试音体验导入与分析：

```bash
bash run.sh --demo
```

也可指定自己的文件：

```bash
bash run.sh --file /绝对路径/录音.wav
```

如果 `.venv-run` 不存在，本机可通过以下方式重建，复用 base 包：

```bash
/home/rwj/anaconda3/bin/python -m venv --system-site-packages .venv-run
```

在其他电脑上建议用 Python 3.11 或更新版本创建独立虚拟环境，再安装 `requirements.txt`；固定版本记录的是本机验证组合。Linux 还需桌面 Qt 所依赖的图形和音频系统库、可用音频输出设备，以及中文字体。字体优先 Noto Sans CJK；Windows 可使用微软雅黑。

## 操作流程

1. 导入 WAV，或点击右上“载入合成测试音”。合成测试音由多个音调与少量噪声构成，**不是真人语音，也不作为去噪效果证据**。
2. 检查左侧元信息与右侧原声三幅图。
3. 选择输出设备后播放；暂停后点击继续；停止将进度归零。本机默认 ALSA 输出不能打开时，选择 `plughw:CARD=Generic,DEV=0`（已通过静音设备验证）。
4. 点击“保存原声 WAV”，导出当前缓存。源音频不会自动归一化或套用任何灰色控件效果。

第一阶段输入上限为 300 秒、128 MiB；支持常见整数 PCM、浮点 WAV。空文件、NaN/Inf 和超出 ±1 的浮点幅度会报错。读取立体声后逐帧平均声道。PCM16 导出存在正常的量化误差，不是字节级复制原 WAV。

设备不支持源采样率或单声道 PCM16 时，播放器按 Qt 返回的设备支持格式重采样/调整输出声道；此转换仅用于播放，不改变原声缓存或导出采样率。沙箱内枚举不到设备；在桌面权限下已找到输入/输出设备，且 Generic 声卡通过全零静音的播放、暂停、继续和自然结束验证。默认 ALSA 设备在本次验证中无法打开，需要显式选择 Generic。静音状态验证不能替代真实语音的听感检查。

## 验证

```bash
QT_QPA_PLATFORM=offscreen .venv-run/bin/python -m pytest -q
QT_QPA_PLATFORM=offscreen bash run.sh --demo --screenshot artifacts/第一阶段实际界面.png
```

自动化测试覆盖文件往返、声道与位深缩放、静音/短音频、440 Hz 频谱、峰值保留显示、播放状态和实际离屏 GUI 文件流程。单元测试的音频设备边界使用测试输出；另已使用 Generic 真设备完成静音状态验证。仍需实际扬声器试听真人语音。

## 项目结构

```text
main.py                       启动与命令行选项
voice_effect_lab/
  audio/data.py               WAV 读写与不可变音频缓存
  audio/player.py             Qt 音频输出与播放状态
  analysis.py                 与 GUI 无关的真实数据分析
  ui/main_window.py           主工作台、状态与功能入口
  ui/plots.py                 六图布局与原声分析
  ui/worker.py                后台文件任务
tests/                        自动化测试
samples/                      明确标注的合成测试音
design/                       已确认的静态设计和要求对照表
docs/                         实施计划、环境与进度
artifacts/                    实际软件截图
```

后续按 [实施计划](docs/superpowers/plans/2026-10-20-stage1.md) 和 [开发进度](docs/开发进度.md) 推进。下一阶段接入录音和底噪采集，再实现谱减去噪。
