# DeepFilterNet3 接入 Implementation Plan

用户指定借用`/media/rwj/ZX1 2TB/project/DeepFilterNet`的完整降噪能力，并明确使用已有`dbaf`环境。主工作台的默认降噪改为DeepFilterNet3，现有图形、四阶段试听/保存与增强/音效串联继续使用同一缓存；传统算法只作课程对照。已实测dbaf Python3.10.11 / torch2.0.1+cu118 / torchaudio2.0.2+cu118 / deepfilterlib0.5.6成功加载本地模型并在CPU处理1秒48k公开合成音，输入输出同N有限。

## 设计与接口

- GUI继续既有.venv-run，后端隔离子进程使用`/home/rwj/anaconda3/envs/dbaf/bin/python`。不安装/升级既有环境，不下载模型，不依赖外置盘继续挂载。
- 拷入必要df Python源码、DeepFilterNet3 config及8.7MB模型checkpoint，保留MIT/Apache原许可和上游提交来源；不复制.git、虚拟环境、其他模型、训练数据或输出日志。上游commit d375b2d8309e0935d165700c91da9de862a99c31。
- `audio/deepfilter.py`提供`deepfilter_denoise(clip,noise_clip=None,strength=100)`，返回与现有DenoiseResult兼容的clip/elapsed_seconds/clipped_samples/estimated_frames=0/algorithm='deepfilter'。strength0精确旁路，其他为原声与模型处理的干湿混合；默认100完整调用模型。
- 外部worker单独文件：临时NPY输入/输出（allow_pickle=False），通过参数明确模型/源码路径；CPU2线程，init_df(log_file=None)，enhance(pad=True)补偿延迟。内部48k，主进程负责SciPy重采样并精确恢复原Fs及N。
- 输入最长300秒/128MiB样本；长音频以10秒核心段+前后0.25秒上下文处理，同一进程只载入一次模型，每段使用新的DF状态并裁掉上下文，保持完整输出N。分段是为限制推理内存，可能与整段参考调用在边界略有不同；短音频与参考同一调用等价。
- 固定受信任checkpoint SHA256，在任何torch.load前核验；不支持任意用户checkpoint。子进程失败/超时/坏输出报AudioError并保留旧有效缓存，不回退成传统算法假装成功。
- 后端runtime availability检查只看固定资源/解释器，明确错误；模型为内置只读资源，日志与临时数据仅写项目.runtime或临时目录。
- 主工作台新增DeepFilterNet3默认，去噪强度100%；模型模式不要求底噪并隐藏参考操作，状态明确模型与dbaf CPU；传统对照模式仍显式选择才运行。更改方法/比例使下游依赖失效。

## 实施

- [x] 后端及vendor模块TDD：实际dbaf推理、同N/Fs/有限/输入不变、0%旁路、非法参数/超限、子进程错误、模型缺失/完整性、不写原工程、长段尾部。
- [x] 主界面默认切换与参考规则/快照/串联状态，旧回归显式选择其对应传统算法；新增默认DF真实Qt/导出/失败保护/增强音效串联RED→GREEN。
- [x] 使用公开中文合成样本对照直接参考项目调用，最大绝对误差0；记录耗时/重采样/分段边界，输出不宣称未经试听的提升。
- [x] 独立审查无Critical/Important，完整444项通过，Minor文案修正后14项GUI专项通过；文档标注模型来源、原理、环境与归因。Git交付沿用既有同步授权。

## 继续状态

BP功能与后续DeepFilter整合完成，一次新上下文综合审查已完成；源码/文档/公开产物在feat/bp-denoising工作区。早期BP单独审查因用户切换任务中断，未产生报告；未重复审查。Git提交此前因自动审批超时未执行，收尾按既有授权再试一次。不得读取/提交voice/或私人录音；仅拷入许可要求的上游推理源码和固定公开模型并归因，不上传原参考工程的其他内容。
