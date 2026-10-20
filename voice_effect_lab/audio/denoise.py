"""基于独立噪声参考的功率谱减；原声保持不变。

周期 Hann 分析/合成窗，75% 重叠时窗平方之和为 1.5。
两端补一整窗后裁回原长，保留边界；FFT 每批最多256帧。
"""
from dataclasses import dataclass
from math import gcd
from numbers import Real
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import resample_poly
from scipy.signal.windows import hann

from .data import AudioClip, AudioError


@dataclass(frozen=True)
class DenoiseResult:
    clip: AudioClip
    elapsed_seconds: float
    clipped_samples: int


def _frame_batches(samples, size, hop):
    frames = np.lib.stride_tricks.sliding_window_view(samples, size)[::hop]
    for first in range(0, len(frames), 256):
        yield first, frames[first:first+256]


def spectral_subtract(clip: AudioClip, noise_clip: AudioClip, strength=60) -> DenoiseResult:
    """强度0精确旁路；0–100对应谱减系数0–3，幅度增益下限0.05。

    适用于相对稳定的环境噪声。保留输入相位，不估计说话内容或声纹。
    """
    started = perf_counter()
    if not isinstance(strength, Real) or not np.isfinite(strength) or not 0 <= strength <= 100:
        raise AudioError('去噪强度必须是 0–100 的有限数值。')
    if noise_clip.duration < .1:
        raise AudioError('底噪参考至少需要 0.10 秒。')
    name = Path(clip.name).stem + '_去噪.wav'
    if strength == 0:
        return DenoiseResult(AudioClip(clip.samples, clip.sample_rate, name),
                             perf_counter()-started, 0)

    # 重采样只用于参考谱，不改变原声缓存、结果时长或导出采样率。
    noise = noise_clip.samples
    if noise_clip.sample_rate != clip.sample_rate:
        divisor = gcd(clip.sample_rate, noise_clip.sample_rate)
        noise = resample_poly(noise, clip.sample_rate//divisor,
                              noise_clip.sample_rate//divisor)
    size = max(32, min(4096, 4*round(clip.sample_rate*.032/4)))
    hop = size//4
    window = hann(size, sym=False).astype(np.float32)
    if len(noise) < size:
        noise = np.pad(noise, (0, size-len(noise)))
    noise_power = np.zeros(size//2+1, dtype=np.float64)
    count = 0
    for _, frames in _frame_batches(noise, size, hop):
        spectrum = rfft(frames*window, axis=-1)
        noise_power += (spectrum.real**2+spectrum.imag**2).sum(axis=0, dtype=np.float64)
        count += len(frames)
    noise_power = (noise_power/count).astype(np.float32)

    padded = np.pad(clip.samples, (size, size))
    output = np.zeros_like(padded)
    alpha = float(strength)*.03
    for first, frames in _frame_batches(padded, size, hop):
        spectrum = rfft(frames*window, axis=-1)
        power = spectrum.real**2+spectrum.imag**2
        gain = np.sqrt(np.maximum(1-alpha*noise_power/np.maximum(power, 1e-20), .05**2))
        restored = irfft(spectrum*gain, n=size, axis=-1)*window
        for offset, frame in enumerate(restored):
            start = (first+offset)*hop
            output[start:start+size] += frame
    samples = output[size:size+len(clip.samples)]/1.5
    clipped = int(np.count_nonzero(np.abs(samples) > 1))
    np.clip(samples, -1, 1, out=samples)
    result = AudioClip(samples, clip.sample_rate, name)
    return DenoiseResult(result, perf_counter()-started, clipped)
