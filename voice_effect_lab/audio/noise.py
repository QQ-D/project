"""从原声提取明确确认过的底噪片段。"""
import math

from .data import AudioClip, AudioError


def noise_bounds(clip, start, end):
    if (not math.isfinite(start) or not math.isfinite(end)
            or start < 0 or end > clip.duration or end <= start):
        raise AudioError('选段应满足：0 ≤ 起点 < 终点 ≤ 音频时长。')
    first, last = round(start*clip.sample_rate), round(end*clip.sample_rate)
    if last-first < math.ceil(.1*clip.sample_rate):
        raise AudioError('底噪片段至少需要 0.10 秒。')
    return first, last


def extract_noise(clip, start, end):
    first, last = noise_bounds(clip, start, end)
    return AudioClip(clip.samples[first:last], clip.sample_rate,
                     f'底噪_{start:.3f}-{end:.3f}s.wav')
