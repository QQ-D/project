"""三段EQ/清晰度→峰值归一化→用户增益；不修改输入缓存。

双二阶系数采用 https://www.w3.org/TR/audio-eq-cookbook/ 的架式/峰式公式。
"""
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import sosfilt

from .data import AudioClip, AudioError


@dataclass(frozen=True)
class EnhancementSettings:
    enabled: bool = True
    normalize_peak: bool = True
    gain: float = 1.0
    clarity: float = 0.0
    low_db: float = 0.0
    mid_db: float = 0.0
    high_db: float = 0.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not isinstance(self.normalize_peak, bool):
            raise AudioError('增强与归一化开关必须为布尔值。')
        for key, low, high in [('gain',0,2),('clarity',0,100),
                               ('low_db',-12,12),('mid_db',-12,12),('high_db',-12,12)]:
            value=getattr(self,key)
            if (not isinstance(value,Real) or isinstance(value,bool)
                    or not np.isfinite(value) or not low <= value <= high):
                raise AudioError(f'增强参数 {key} 必须在 {low} 到 {high} 范围内且为有限数值。')


@dataclass(frozen=True)
class EnhancementResult:
    clip: AudioClip
    elapsed_seconds: float
    clipped_samples: int
    input_peak: float
    output_peak: float


def _biquad(kind, frequency, db, rate):
    a=10**(db/40)
    omega=2*np.pi*frequency/rate
    c,s=np.cos(omega),np.sin(omega)
    if kind=='mid':
        alpha=s/2  # Q=1
        b=[1+alpha*a,-2*c,1-alpha*a]
        denominator=[1+alpha/a,-2*c,1-alpha/a]
    else:
        alpha=s/np.sqrt(2)  # S=1
        term=2*np.sqrt(a)*alpha
        if kind=='low':
            b=[a*((a+1)-(a-1)*c+term),2*a*((a-1)-(a+1)*c),a*((a+1)-(a-1)*c-term)]
            denominator=[(a+1)+(a-1)*c+term,-2*((a-1)+(a+1)*c),(a+1)+(a-1)*c-term]
        else:
            b=[a*((a+1)+(a-1)*c+term),-2*a*((a-1)+(a+1)*c),a*((a+1)+(a-1)*c-term)]
            denominator=[(a+1)-(a-1)*c+term,2*((a-1)-(a+1)*c),(a+1)-(a-1)*c-term]
    return np.asarray(b+denominator,dtype=np.float64)/denominator[0]


def enhance_audio(clip: AudioClip, settings: EnhancementSettings = EnhancementSettings()) -> EnhancementResult:
    started=perf_counter()
    input_peak=float(np.max(np.abs(clip.samples)))
    name=Path(clip.name).stem+'_增强.wav'
    if not settings.enabled:
        return EnhancementResult(AudioClip(clip.samples,clip.sample_rate,name),
                                 perf_counter()-started,0,input_peak,input_peak)
    values=clip.samples.astype(np.float64)
    fraction=settings.clarity/100
    bands=[('low',min(300,clip.sample_rate*.1),settings.low_db-3*fraction),
           ('mid',min(1500,clip.sample_rate*.25),settings.mid_db+4*fraction),
           ('high',min(3000,clip.sample_rate*.4),settings.high_db+2*fraction)]
    sections=[_biquad(kind,freq,db,clip.sample_rate) for kind,freq,db in bands if db != 0]
    if sections:
        values=sosfilt(np.stack(sections),values)
    peak=float(np.max(np.abs(values)))
    if settings.normalize_peak and peak > 0:
        values *= .95/peak
    values *= settings.gain
    if not np.isfinite(values).all():
        raise AudioError('增强滤波产生非有限样本，请检查音频与参数。')
    clipped=int(np.count_nonzero(np.abs(values)>1))
    np.clip(values,-1,1,out=values)
    result=AudioClip(values,clip.sample_rate,name)
    return EnhancementResult(result,perf_counter()-started,clipped,input_peak,
                             float(np.max(np.abs(result.samples))))
