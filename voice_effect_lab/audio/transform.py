"""Independent tempo and pitch using a two-frame phase vocoder.

Centered Hann analysis, interpolated magnitudes and unwrapped instantaneous
frequencies feed weighted overlap-add. Only the current two spectra are cached;
there is no whole-recording STFT matrix. Pitch shifting stretches by the inverse
pitch factor, then Fourier-resamples to the original exact sample count.
Transient smearing and changed formants are inherent to this basic algorithm.
"""
from numbers import Integral, Real

import numpy as np
from scipy.fft import irfft, rfft
from scipy.signal import resample

from .data import AudioError

MAX_OUTPUT_SAMPLES = 32 * 1024 * 1024


def _check_count(count):
    if count > MAX_OUTPUT_SAMPLES:
        raise AudioError('变调变速的输入、中间结果或输出超过 32 Mi 样本上限。')


def _parameter(value, low, high, label):
    if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or not low <= value <= high):
        raise AudioError(f'{label}必须是 {low} 到 {high} 范围内的有限数值。')
    return float(value)


def _samples(samples, sample_rate):
    if (not isinstance(sample_rate, Integral)
            or isinstance(sample_rate, (bool, np.bool_)) or sample_rate <= 0):
        raise AudioError('采样率必须为正整数。')
    try:
        values = np.asarray(samples)
    except (ValueError, TypeError) as exc:
        raise AudioError('音频必须是非空的单声道实数数据。') from exc
    if (values.ndim != 1 or not values.size
            or values.dtype.kind not in 'iuf'):
        raise AudioError('音频必须是非空的单声道实数数据。')
    _check_count(len(values))
    _check_finite(values)
    return values


def _check_finite(values):
    # Bound validation's temporary boolean arrays even for long input signals.
    for start in range(0, len(values), 65536):
        if not np.isfinite(values[start:start + 65536]).all():
            raise AudioError('变调变速数据包含 NaN 或无穷值。')


def _stretch(values, rate, sample_rate, count):
    if len(values) == 1:
        return np.full(count, values[0], dtype=np.float64)
    # About 64 ms, bounded for unusually low/high sample rates and short audio.
    frame_size = 32
    desired = min(4096, sample_rate // 16, max(32, len(values)))
    while frame_size < desired:
        frame_size *= 2
    hop = frame_size // 4
    half = frame_size // 2
    window = np.hanning(frame_size + 1)[:-1]
    window_power = window * window
    expected = 2 * np.pi * hop * np.arange(half + 1) / frame_size

    def spectrum(index):
        start = index * hop - half
        left, right = max(0, start), min(len(values), start + frame_size)
        frame = np.zeros(frame_size, dtype=np.float64)
        if right > left:
            frame[left - start:right - start] = values[left:right]
        return rfft(frame * window)

    output = np.zeros(count, dtype=np.float64)
    weight = np.zeros(count, dtype=np.float64)
    cached_index = -1
    previous = following = phase = None
    for output_index, center in enumerate(range(0, count + half, hop)):
        position = output_index * rate
        index = int(position)
        fraction = position - index
        if index != cached_index:
            previous = (following if cached_index >= 0 and index == cached_index + 1
                        else spectrum(index))
            following = spectrum(index + 1)
            cached_index = index
        if phase is None:
            phase = np.angle(previous)
        magnitude = (1 - fraction) * np.abs(previous) + fraction * np.abs(following)
        frame = irfft(magnitude * np.exp(1j * phase), n=frame_size) * window
        start = center - half
        left, right = max(0, start), min(count, start + frame_size)
        output[left:right] += frame[left - start:right - start]
        weight[left:right] += window_power[left - start:right - start]
        deviation = np.angle(following) - np.angle(previous) - expected
        deviation -= 2 * np.pi * np.round(deviation / (2 * np.pi))
        phase += expected + deviation
        # Keep phase bounded to avoid cumulative precision loss on long signals.
        phase = np.remainder(phase + np.pi, 2 * np.pi) - np.pi
    np.divide(output, weight, out=output, where=weight > 1e-12)
    return output


def time_stretch(samples, rate, sample_rate) -> np.ndarray:
    """Change tempo at fixed Fs/pitch; output max(1, round(N / rate)) samples."""
    rate = _parameter(rate, .5, 2, '语速')
    values = _samples(samples, sample_rate)
    count = max(1, round(len(values) / rate))
    _check_count(count)
    if rate == 1:
        return values.copy()
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            output = _stretch(values, rate, sample_rate, count)
    except (FloatingPointError, OverflowError, MemoryError) as exc:
        raise AudioError('变速计算超出可用内存或数值范围。') from exc
    _check_finite(output)
    return output


def pitch_shift(samples, semitones, sample_rate) -> np.ndarray:
    """Shift pitch by −12..12 semitones, retaining exactly N samples and Fs."""
    semitones = _parameter(semitones, -12, 12, '音高半音数')
    values = _samples(samples, sample_rate)
    if semitones == 0:
        return values.copy()
    factor = 2 ** (semitones / 12)
    # Check before stretching, including the factor=2 (+12) endpoint.
    _check_count(max(1, round(len(values) * factor)))
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            stretched = time_stretch(values, 1 / factor, sample_rate)
            output = resample(stretched, len(values))
    except (FloatingPointError, OverflowError, MemoryError) as exc:
        raise AudioError('变调计算超出可用内存或数值范围。') from exc
    _check_finite(output)
    return output
