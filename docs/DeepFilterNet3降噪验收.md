# DeepFilterNet3 降噪接入与验收

按用户指定，主工作台的默认降噪已改为直接调用所给工程的 **DeepFilterNet3 预训练模型**，使用已有 **dbaf** 环境在 CPU 运行。现有导入、六图、原声/去噪/增强/音效切换、试听、保存和音效串联继续使用真实缓存。传统维纳/谱减保留为显式课程对照；小型 BP 栏仍用于训练实验。

## 使用

关闭旧窗口，在项目目录启动：

```bash
bash run.sh
```

导入 WAV 后，02 区域默认选择“DeepFilterNet3（默认 · dbaf）”，直接点“一键去噪”或“开始处理”。这个模式不需要麦克风、底噪文件或噪声选段，参考控件会隐藏。

默认强度 **100%**，对应完整模型处理；0% 精确旁路，中间值为原声与模型输出的混合比例。它与传统算法的强度参数含义不同。完成后在底部切换“原声/去噪”试听并查看六图，保存所选阶段 WAV。

勾选“使用去噪结果作为后续处理输入”后，一键增强或应用音效会复用有效 DeepFilterNet3 结果；没有结果则在同一后台任务先降噪。改变方法、强度或原声会清除依赖它的下游结果。失败、超时和过期任务不覆盖原有有效缓存，也不自动换成其他算法。

“重置”恢复 DeepFilterNet3、100%，关闭增强与音效；原声保留。需要传统实验时，在方法框显式选择维纳或谱减，参考底噪操作才会出现。

## 项目来源与运行方式

原工程：`/media/rwj/ZX1 2TB/project/DeepFilterNet`。

已将必要推理文件拷入 [third_party/DeepFilterNet](../third_party/DeepFilterNet/UPSTREAM)：12 个未修改的 `df` Python 运行时源码、包元数据、DeepFilterNet3 配置、8,714,073 字节 checkpoint、三份原许可文件及来源说明。总计 19 文件、8,864,557 字节。没有复制上游 Git 历史、虚拟环境、训练数据、其他模型或日志；运行不再依赖外置盘路径。

上游提交为 `d375b2d8309e0935d165700c91da9de862a99c31`。拷入的 18 个原始文件均与外置盘工程逐字节一致，另一个文件为本项目增加的 UPSTREAM 说明。原作者、原许可及模型来源均保留：[MIT](../third_party/DeepFilterNet/LICENSE-MIT)、[Apache-2.0](../third_party/DeepFilterNet/LICENSE-APACHE)。本项目新增的是界面和进程适配，不把上游模型或网络算法称为自行训练或原创。

主界面仍用原来的 `.venv-run`，避免把两套 Python/Qt/NumPy 混入同一进程；实际模型由 `/home/rwj/anaconda3/envs/dbaf/bin/python` 子进程调用。已验证组合为：

| 组件 | 版本 |
| --- | --- |
| dbaf Python | 3.10.11 |
| torch | 2.0.1+cu118 |
| torchaudio | 2.0.2+cu118 |
| deepfilterlib | 0.5.6 |
| NumPy | 1.26.4 |

没有安装或升级既有环境。虽然 torch 构建包含 CUDA 支持，当前适配固定使用 CPU 两线程，不占用 GPU。若将项目迁移到另一台已有兼容环境的电脑，可通过绝对路径设置 `VOICE_LAB_DEEPFILTER_PYTHON`；缺少解释器、模型或依赖时会明确报错。

## 原理与处理边界

这里实际执行上游的 `init_df` 和 `enhance(..., pad=True)`。网络从含噪谱特征预测 ERB 频带增益和深度滤波系数，再重建音频；不是只照着名称重写几条滤波公式。`pad=True` 补偿 STFT/ISTFT 的处理延迟。

内部使用 48 kHz，适配层先重采样，再把输出恢复为原采样率与精确样本数。异常互质采样率避免生成巨大的多相滤波器。输入支持最长 300 秒、128 MiB float32 样本；这不是整个进程的内存上限。

为控制长音频推理内存，超过 10 秒的音频按 10 秒核心段处理，每段加前后 0.25 秒上下文、使用新的滤波状态并裁掉上下文。同一作业只加载一次模型。分段边界可能与整段推理有差别；短于 10 秒的本次样例直接与上游整段调用核对。

模型与配置有固定 SHA256 校验。checkpoint 从已验证的同一 bytes 快照交给 torch 加载，不接收任意用户 checkpoint；输出临时 NPY 先校验头、形状、类型及实际字节数，再从同一有界快照读取。缓存和导出没有隐式归一化，越界统一限幅并统计。

## 实际验证

```bash
QT_QPA_PLATFORM=offscreen .venv-run/bin/python -m pytest -q
QT_QPA_PLATFORM=offscreen .venv-run/bin/python -m scripts.verify_deepfilter
```

完整功能回归 **444 项通过（114.20 秒）**，其中新增 DF 后端 31 项、真实 Qt 界面 3 项；之前 BP 阶段独立完成 410 项。后端测试包含 dbaf 实际推理、48 kHz/16 kHz/44.1 kHz/47999 Hz/1 Hz、单样本、静音、跨 10 秒尾段、0% 旁路、模型/依赖缺失、坏 checkpoint、超时和恶意 NPY 头。

公开中文验收使用本项目自写 TTS 句子、人工高斯噪声，不使用私人录音。48 kHz、389369 样本、8.11185 秒输入，经主界面处理，与外置盘原工程直接 `init_df + enhance(pad=True)` 的输出 **最大样本绝对误差为 0**，确认调用相同模型且短片输出一致。

该次包含启动、模型加载、推理及分析的 GUI 作业约 1.972 秒；单机耗时不能代表其他设备。22.05 kHz、178866 样本的另一个用例也保持原采样率和精确长度。增强/混响串联复用同一去噪缓存，输出 398968 样本，三份试听/图形/PCM16 导出一致，量化误差不超过 1/32768。1080×720 的处理按钮可滚动到达。

一次独立整分支审查无 Critical/Important；另有 11 项独立恶意文件边界窄测通过。审查发现 BP 栏“预训练引擎未接通”的文字容易与新主工作台混淆，已改为明确说明 DeepFilterNet3 在主工作台使用；文案调整后专项 GUI 回归见开发进度。

产物：

- [公开中文加噪输入](../artifacts/DeepFilterNet3_公开中文加噪输入.wav) · [干净合成参考](../artifacts/DeepFilterNet3_公开中文干净参考.wav) · [DeepFilterNet3 输出](../artifacts/DeepFilterNet3_公开中文降噪输出.wav)
- [降噪→增强→混响](../artifacts/DeepFilterNet3_增强混响串联.wav) · [实际工作台](../artifacts/DeepFilterNet3_实际工作台.png) · [串联截图](../artifacts/DeepFilterNet3_音效串联.png) · [JSON](../artifacts/DeepFilterNet3验收数据.json)

自动验收没有启动麦克风或扬声器。输出一致性证明适配正确，实际录音的主观改善仍以试听为准。
