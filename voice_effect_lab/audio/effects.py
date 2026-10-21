"""基础音效：真实倍率重采样→电话带通→有限次回声→越界限幅。

算法接口参考 SciPy 官方 resample_poly / butter / sosfilt 文档。
倍率以0.01为步进；输出保持原采样率，实际改变样本数量与音高。
"""
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real, Integral
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from .data import AudioClip, AudioError

MAX_OUTPUT_SAMPLES = 128 * 1024 * 1024 // 4


@dataclass(frozen=True)
class EffectsSettings:
    enabled: bool = True
    rate: float = 1.0
    telephone: bool = False
    echo_delay: float = .3
    echo_strength: float = 0.0
    echo_count: int = 1

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not isinstance(self.telephone, bool):
            raise AudioError('音效与电话滤波开关必须为布尔值。')
        for key, low, high in [('rate',.5,2),('echo_delay',0,1),('echo_strength',0,.8)]:
            value=getattr(self,key)
            if (not isinstance(value,Real) or isinstance(value,bool)
                    or not np.isfinite(value) or not low<=value<=high):
                raise AudioError(f'音效参数 {key} 必须在 {low} 到 {high} 范围内且为有限数值。')
        if not isinstance(self.echo_count,Integral) or isinstance(self.echo_count,bool) or not 1<=self.echo_count<=8:
            raise AudioError('回声次数必须为1–8的整数。')
        if abs(self.rate-round(self.rate,2))>1e-9:
            raise AudioError('基础倍率以0.01为步进。')


@dataclass(frozen=True)
class EffectsResult:
    clip: AudioClip
    elapsed_seconds: float
    clipped_samples: int


def apply_effects(clip: AudioClip, settings: EffectsSettings = EffectsSettings()) -> EffectsResult:
    started=perf_counter()
    ratio=Fraction(str(round(settings.rate,2))) if settings.enabled else Fraction(1)
    count=(len(clip.samples)*ratio.denominator+ratio.numerator-1)//ratio.numerator
    delay=round(settings.echo_delay*clip.sample_rate) if settings.enabled and settings.echo_strength else 0
    if count+delay*settings.echo_count>MAX_OUTPUT_SAMPLES:
        raise AudioError('音效结果超过128 MiB缓存上限，请截短输入或提高倍率。')
    values=clip.samples.astype(np.float64)
    if settings.enabled:
        if ratio!=1:
            values=resample_poly(values,ratio.denominator,ratio.numerator)
        if settings.telephone:
            frequencies=[min(300,clip.sample_rate*.1),min(3400,clip.sample_rate*.4)]
            sections=butter(4,frequencies,btype='bandpass',fs=clip.sample_rate,output='sos')
            values=sosfilt(sections,values)
        if settings.echo_strength:
            if delay:
                echoed=np.zeros(len(values)+delay*settings.echo_count,dtype=np.float64)
                echoed[:len(values)]=values
                for repeat in range(1,settings.echo_count+1):
                    start=repeat*delay
                    echoed[start:start+len(values)]+=settings.echo_strength**repeat*values
                values=echoed
            else:
                values *= sum(settings.echo_strength**repeat for repeat in range(settings.echo_count+1))
    if not np.isfinite(values).all():
        raise AudioError('音效产生非有限样本，请检查输入与参数。')
    clipped=int(np.count_nonzero(np.abs(values)>1))
    np.clip(values,-1,1,out=values)
    output=AudioClip(values,clip.sample_rate,Path(clip.name).stem+'_音效.wav')
    return EffectsResult(output,perf_counter()-started,clipped)
