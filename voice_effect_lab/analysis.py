"""从同一音频缓存生成波形、频谱和语谱数据，不依赖 GUI。"""
from dataclasses import dataclass

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann

from .audio.data import AudioClip


@dataclass
class AudioAnalysis:
    wave_times: np.ndarray
    wave_values: np.ndarray
    frequencies: np.ndarray
    spectrum_db: np.ndarray
    spec_times: np.ndarray
    spec_frequencies: np.ndarray
    spectrogram_db: np.ndarray
    spectrum_duration: float


def _waveform(samples, rate):
    if len(samples) <= 6000:
        return np.arange(len(samples)) / rate, samples
    # 每个时间桶保留最小值与最大值，防止简单抽点漏掉瞬态。
    edges = np.linspace(0, len(samples), 3001, dtype=int)
    indices = []
    for start, end in zip(edges[:-1], edges[1:]):
        block = samples[start:end]
        indices.extend(sorted((start + int(block.argmin()), start + int(block.argmax()))))
    indices = np.asarray(indices)
    return indices / rate, samples[indices]


def analyze(clip: AudioClip) -> AudioAnalysis:
    data, rate = clip.samples, clip.sample_rate
    times, values = _waveform(data, rate)
    # 限制频谱计算到前 8 秒；GUI 明确显示实际分析时长。
    segment = data[:min(len(data), rate * 8)]
    win = hann(len(segment), sym=False) if len(segment) > 1 else np.ones(1)
    mag = np.abs(rfft(segment * win)) / max(win.sum(), 1e-12)
    if len(segment) % 2 == 0:
        mag[1:-1] *= 2
    else:
        mag[1:] *= 2
    spectrum_db = 20 * np.log10(np.maximum(mag, 1e-6))

    frame_size = max(32, min(2048, round(rate * .032)))
    hop = max(1, frame_size // 4)
    half = frame_size // 2
    padded = np.pad(data, (half, frame_size - half))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_size)[::hop]
    starts = np.arange(len(frames)) * hop
    if len(frames) > 2000:
        selection = np.linspace(0, len(frames)-1, 2000, dtype=int)
        frames, starts = frames[selection], starts[selection]
    window = hann(frame_size, sym=False)
    stft = np.abs(rfft(frames * window, axis=1)) / window.sum()
    stft[:, 1:-1 if frame_size % 2 == 0 else None] *= 2
    return AudioAnalysis(
        times, values, rfftfreq(len(segment), 1/rate), spectrum_db,
        starts/rate, rfftfreq(frame_size, 1/rate),
        (20*np.log10(np.maximum(stft, 1e-6))).T,
        len(segment)/rate,
    )
