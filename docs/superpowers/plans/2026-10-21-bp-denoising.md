# 小型BP降噪与传统方法对比 Implementation Plan

> 按用户已确认的v2界面设计继续实现C栏，并沿用本会话逐步开发、当前目录及GitHub同步授权。后端与界面有固定接口，各自独立文件；使用dispatching-parallel-agents完成两个独立组件，主代理完成整合、真实Qt验收与整分支审查。

**Goal:** 接通可训练小型BP频谱掩膜网络、模型保存/加载、智能/维纳降噪对比及试听/导出。
**Architecture:** NumPy实现一隐藏层BP网络，SciPy STFT/重建与采样率转换；单独NeuralController管理模型、训练数据与比较缓存。复用全局后台任务互斥；对比图使用单独AnalysisPlots对话框，主工作台四阶段语义保持，明确发送结果才作为新原声。
**Tech Stack:** 既有NumPy/SciPy/PyQt5，不安装深度学习框架或下载模型。
**Spec:** `design/界面设计说明.md` 第3节C栏与缓存、失败保护、真实指标要求。

## Global Constraints

- 本次是教学小型BP，不标称通用预训练语音增强，不保证超过维纳；CNN及去混响未完成入口保持禁用并说明。
- 默认训练为公开合成谐波/噪声数据；也可导入2–60秒干净WAV，自动混合训练噪声。训练/验证在生成或时域划分时分离，禁止同一批随机帧同时用于评估。
- 训练1–100轮、批大小32–512；默认20/128；固定seed可重复，逐轮显示真实损失。
- 网络/输入归一化参数保存NPZ，allow_pickle=False；限制文件和解压数组大小，严格键/形状/版本/有限数值；不加载任意Python对象。只支持本项目格式。
- 推理保持原采样率和精确N，强度0–100、默认75；0精确旁路。输入最长300秒且128MiB样本，分批构造特征/推理，不能构造整段frames×bins×features巨大矩阵。
- 模型和训练数据独立于工作台原声；更换原声/模型/智能强度会清比较结果，失败保留已有有效数据，过期快照丢弃。
- 传统基线固定自动维纳60%；不借用已确认底噪，不修改现有去噪/增强/音效缓存。比较原声、传统、BP使用同一份选中缓存试听与导出。
- 公开加噪演示具有独立干净参考，可显示SNR与耗时；任意真实录音无参考时SNR显示“—”，不得伪造。
- 神经试听与TTS、主工作台、采集及后台任务互斥；共享音量/静音、设备回退和安全关闭。
- 维持最小1080×720，C栏滚动可达；不读/不提交voice/与私人录音、不启动硬件。

## Review Focus

1. 训练/验证数据泄漏、真实梯度训练和有证据的指标；非演示输入不可展示虚构SNR。
2. NPZ不可信输入，包括超大解压、坏形状、NaN、未知版本以及带对象数组。
3. 采样率/长度/短片/全零、strength0、资源上限及分批推理。
4. 模型/输入/参数变化及失败后的原子采用、播放/图形/保存一致。
5. 训练/保存/采集/播放互斥、页面切换、设备回退与关闭等待。

## Task 1: BP后端（独立）

Files: `audio/neural.py`、可拆`audio/neural_data.py`，`tests/test_neural.py`。
Interfaces:
- `train_model(clean_clip=None, *, epochs=20, batch_size=128, progress=None)->TrainingResult`；progress(str)逐轮报告；TrainingResult含model、train_losses、validation_losses、elapsed_seconds、train_samples、validation_samples。
- `NeuralModel`只读权重和归一化参数，含`training_label:str`；`denoise_neural(clip,model,strength=75)->NeuralResult`，含clip、elapsed_seconds、clipped_samples。
- `save_model(path,model)`、`load_model(path)->NeuralModel`；`make_demo_pair(seed=20261022,seconds=3,snr_db=5)->(clean_clip,noisy_clip)`，与训练样本独立。
- [x] 编写缺失模块/实际训练/验证分离/保存加载/非法NPZ/长度与采样率/低Fs/静音/0强度/资源上限失败测试，观察RED后实现GREEN。

## Task 2: 智能面板及控制器（独立）

Files: `ui/neural_controller.py`、`ui/neural_panel.py`，`tests/test_neural_ui.py`。
Interfaces: `NeuralController._init_neural(), build_neural_panel(), _neural_controls(), _neural_source_changed(), _close_neural()`；`neural_player`，`neural_model`，`neural_clip/neural_analysis`，`neural_traditional_clip/neural_traditional_analysis`，`neural_source_clip/neural_source_analysis`，`neural_metrics`。
- 使用host._job/_loaded/_status/_error/occupied/player/tts_player/output_combo/tabs/clip/analysis。
- 所有新控件neural_前缀；controller内部QObject桥接训练progress，不修改host信号/worker。
- 提供训练（默认合成/导入干净WAV）、保存/加载NPZ、公开加噪演示、运行BP与维纳比较、原声/传统/BP选择、试听/停止、图形、保存WAV、送入主工作台。
- 主代理接入_main_window各生命周期、两页音量与TTS反向互斥；agent不改host文件。
- [x] 真Qt harness的生成训练、失败/过期保护、比较与指标、选择/导出/试听/送入、资源及控件互斥RED→GREEN。

## Task 3: 整合与交付

- [x] 主窗口C栏与生命周期接通，TTS双向互斥，新增主窗口RED→GREEN测试。
- [x] 公开样本实际训练→保存/重载→比较→三份导出/六图核对，实测SNR和损失、最小窗口、截图与JSON，不自动播放。
- [x] 与后续DeepFilter接入一并完成整分支独立审查；BP完整410项通过，综合444项通过；修正Qt生命周期回归。
- [x] README/开发进度/验收Markdown已更新；Git交付沿用既有main快进同步授权。

## 起点

`65b1c8e`，完整362项通过（87.04秒），当前源码未变化。清理缓存不改变已测试代码；分支`feat/bp-denoising`。清理任务由用户暂停，旧扩展和APT缓存未删除。
