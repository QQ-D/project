# 高级音效与中文TTS Implementation Plan

> 按用户已确认的完整界面设计与“中文TTS以及之前的都完成”范围持续实施；复用原目录、.venv-run与既有GitHub同步授权。使用writing-plans梳理接口、TDD验证；独立TTS与变调DSP按dispatching-parallel-agents并行，其余由主代理整合。

**Goal:** 完成独立试听音量、混响、机械电音、动态压缩、可排序音效串联、独立变调/变速及中文TTS生成/试听/保存/送入工作台。
**Architecture:** 扩展不可变EffectsSettings，独立transform/advanced模块实现DSP；既有后台任务快照与四阶段缓存继续使用。TTS独立后端及控制器，合成缓存与工作台原声分开，明确发送才替换原声。
**Tech Stack:** 已有PyQt5/NumPy/SciPy，TTS使用系统libespeak-ng.so.1和cmn语言数据，无Python包安装或模型下载。
**Spec:** `design/界面设计说明.md`选做表A/B与缓存/试听/导出要求；本会话已确认功能清单。智能模型列属于后续范围。

## Global Constraints

- 试听音量0–100%、默认100，独立静音，QAudioOutput实时更新；不修改缓存/导出，不修改系统音量。需要超过原电平用增强增益/压缩补偿。
- 独立音高−12至+12半音、默认0，语速.5–2、默认1；关闭默认；保留采样率。time_stretch输出max(1,round(N/rate))，pitch_shift保持N；phase vocoder有瞬态/音色局限，非身份转换。
- 房间混响默认关闭，湿度0–1默认.25、衰减时间.2–3秒默认1.2；确定性指数衰减噪声/早反射IR，完整卷积尾音，湿度0旁路。
- 机械电音默认关闭，环形调制20–120Hz默认40、混合0–1默认.6，频率不超过.45Fs；保持长度。
- 动态压缩默认关闭，阈值−40至−6dB默认−18、比率1–8默认3、补偿0–12dB默认3；平滑RMS/增益，最终统一限幅统计。
- 基础倍率始终最先执行。后续可排序节点pitch_tempo/telephone/modulation/compression/echo/reverb，每项恰一次；关闭节点跳过。顺序纳入快照，按顺序预测长度后检查128MiB缓存，保留尾音与Fs。
- 非自定义基础预设和重置恢复高级旁路/默认顺序；自定义保留；高级启用会打开音效主开关。参数/顺序变化只清音效，保留有效去噪/增强；继承互斥、失败保留、过期快照与关闭等待。
- 扩展高级区域可滚动，最小1080×720可访问全部功能；只启用真实完成入口。
- TTS离线基础普通话，语速.5–2、最长400字符与300秒/128MiB音频；后台合成和分析，失败不覆盖旧合成/原声，文本/语速变化撤销旧合成，过期快照不采用。
- TTS引擎缺失时给明确不可用说明，不能假装成功；不发送文字至远端，不启动麦克风或自动播放。TTS预览与主播放互斥，发送工作台才替换原声并重置相关噪声确认/缓存。
- 自动验证只使用公开自写测试句与合成信号；未知voice/目录及私人录音保持不动、不上传。

## Review Focus

1. 时长/音高独立正确，短片/静音/极低Fs有限，长变换中间分配受控。
2. 不同串联顺序的长度预判和卷积尾音正确，零湿度/关闭模块精确旁路。
3. 试听静音/音量不重启播放，不影响导出，设备回退后仍生效。
4. TTS库回调/并发/超限/缺失引擎安全；文本或参数改变不采用过期结果。
5. TTS预览、原声替换、采集与全局后台任务互斥；所有新参数/顺序重置和缓存一致。

## Task 1: 独立变调变速DSP（并行）

Files: audio/transform.py, tests/test_transform.py。
Interfaces: time_stretch(samples,rate,sample_rate)->ndarray，pitch_shift(samples,semitones,sample_rate)->ndarray；以AudioError报告非法输入/资源上限，保留输入不变，不在模块内限幅。最大中间样本32*1024*1024。
- [x] RED→GREEN：音调/时长独立、±12半音、.5/2语速、短音频/静音、精确中性、非法参数/资源限制。
- [x] 只运行本模块必要测试，主代理统一全套。

## Task 2: 中文TTS模块（并行）

Files: audio/tts.py（可加隔离worker），ui/tts_controller.py、ui/tts_panel.py、tests/test_tts.py、tests/test_tts_ui.py。
Interfaces: synthesize_chinese(text,speed=1.0)->AudioClip；tts_available()->(bool,str)；TtsController._init_tts(), _tts_controls(), build_tts_panel()->QWidget，tts_player/tts_clip/tts_analysis；使用host._job/_loaded/_error/_status/occupied。
- [x] RED→GREEN：实际中文非零音频、速度影响、输入/长度验证、引擎失败、异步失败/过期与独立缓存，试听/保存/送入工作台。
- [x] 系统合成使用retrieval而非扬声器；后端全局锁/隔离避免C库并发和生命周期错误；本机实际无声合成验证。

## Task 3: 音量与其余高级DSP/界面

Files: player.py, advanced.py, effects.py, ui/advanced_controls.py，main_window/effects_controller，相关测试。
- [x] RED→GREEN：实时音量/静音及重播持续；混响脉冲尾音、湿度0旁路；机械边带；压缩大/小信号与补偿；串联顺序/限幅/总长度。
- [x] 接通滚动高级区和排序，更新控件互斥/缓存/预设/重置与状态；TTS生命周期及音量整合。
- [x] 新GUI失败回归，完整测试通过。

## Task 4: 验收/文档/审查/交付

- [x] 实际Qt：公开中文句生成→保存→送入原声→高级链路→试听缓存/导出核对、截图与JSON（不自动启动音频设备）。
- [x] 完整测试、compileall/Shell/diff/Markdown链接检查；一位新上下文整分支审查，重要发现RED→GREEN修复。
- [x] Markdown说明、合成产物、进度更新；Git交付沿用用户已授权的main快进合并与同步。

## 起点

main 33d3e86，已验证248项；未跟踪voice/属于用户共享内容，保留且不读取。继续复用既有环境。

## 执行记录

- 主体提交`3d7e8dd`，完整361项通过（84.00秒）；TTS23项、独立变换52项、高级DSP25项、音量7项、主窗口6项均经失败后实现。
- `scripts.verify_advanced_tts`实际Qt无声验收完成，生成四份公开中文合成/音效WAV、三张截图和JSON；最小1080×720检查通过。
- 一位新上下文审查无未解决问题，80项独立窄测试通过。实际截图暴露首次TTS进度0/0问题，已加入先失败后通过的回归并刷新采用/清空后的时长。
- 最终完整362项通过（87.04秒），实际Qt验收复跑通过；编译、Shell、diff和88个本地Markdown链接检查通过。
