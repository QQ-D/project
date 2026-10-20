"""WAV 读取与导出。保持原声幅度，不隐式归一化。"""
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from scipy.io import wavfile


class AudioError(ValueError):
    """可以直接向用户展示的音频输入错误。"""


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray
    sample_rate: int
    name: str
    source_channels: int = 1

    def __post_init__(self):
        samples = np.array(self.samples, dtype=np.float32, copy=True)
        if samples.ndim != 1 or not samples.size:
            raise AudioError('音频必须是非空的单声道数据。')
        if not np.isfinite(samples).all():
            raise AudioError('音频包含 NaN 或无穷值，无法处理。')
        if np.max(np.abs(samples)) > 1:
            raise AudioError('音频幅度超出 −1 到 1，请先修正文件幅度。')
        if not isinstance(self.sample_rate, (int, np.integer)) or self.sample_rate <= 0:
            raise AudioError('采样率必须为正整数。')
        samples.setflags(write=False)
        object.__setattr__(self, 'samples', samples)

    @property
    def duration(self):
        return self.samples.size / self.sample_rate


def load_wav(path):
    path = Path(path)
    try:
        if path.stat().st_size > 128 * 1024 * 1024:
            raise AudioError('当前版本支持 128 MiB 以内的 WAV 文件。')
        with warnings.catch_warnings():
            # SciPy 对截断 WAV 默认只发警告并返回部分音频；不能将其当作成功导入。
            warnings.filterwarnings('error', category=wavfile.WavFileWarning,
                                    message=r'.*(EOF|end of file|premature|truncat).*')
            rate, data = wavfile.read(path)
    except AudioError:
        raise
    except (OSError, ValueError, EOFError, wavfile.WavFileWarning) as exc:
        raise AudioError(f'无法读取 WAV：{exc}') from exc
    if data.ndim not in (1, 2) or not data.size:
        raise AudioError('WAV 中没有可用音频。')
    if rate <= 0:
        raise AudioError('WAV 采样率必须为正整数。')
    if len(data) / rate > 300:
        raise AudioError('当前版本支持 300 秒以内的音频。')
    channels = data.shape[1] if data.ndim == 2 else 1
    if data.dtype == np.uint8:
        samples = (data.astype(np.float32) - 128) / 128
    elif np.issubdtype(data.dtype, np.signedinteger):
        # SciPy 将 24 bit PCM 左对齐存储在 int32 中，同样用 2**31 缩放。
        samples = data.astype(np.float64) / (2 ** (np.iinfo(data.dtype).bits - 1))
    elif np.issubdtype(data.dtype, np.floating):
        samples = data
    else:
        raise AudioError(f'暂不支持此 WAV 数据类型：{data.dtype}')
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1:
        raise AudioError('WAV 包含非有限值或超出满幅的浮点音频。')
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return AudioClip(samples, int(rate), path.name, channels)


def save_wav(path, clip):
    # −1 映射到 −32768；正满幅受 PCM16 的最大值约束。无自动增益。
    pcm = np.clip(np.rint(clip.samples.astype(np.float64) * 32768), -32768, 32767).astype(np.int16)
    try:
        wavfile.write(Path(path), clip.sample_rate, pcm)
    except (OSError, ValueError) as exc:
        raise AudioError(f'无法保存 WAV：{exc}') from exc
