"""基础倍率后按所选顺序串联变调/变速、电话、调制、压缩、回声和混响。"""
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real, Integral
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from .data import AudioClip, AudioError
from .advanced import room_reverb, ring_modulate, compress
from .transform import pitch_shift, time_stretch

MAX_OUTPUT_SAMPLES = 128 * 1024 * 1024 // 4
DEFAULT_CHAIN_ORDER=('pitch_tempo','telephone','modulation','compression','echo','reverb')
EFFECT_NAMES={'pitch_tempo':'独立变调/变速','telephone':'电话滤波','modulation':'机械电音',
              'compression':'动态压缩','echo':'回声','reverb':'房间混响'}


@dataclass(frozen=True)
class EffectsSettings:
    enabled: bool = True
    rate: float = 1.0
    telephone: bool = False
    echo_delay: float = .3
    echo_strength: float = 0.0
    echo_count: int = 1
    independent_enabled: bool = False
    pitch_semitones: float = 0.0
    tempo: float = 1.0
    reverb_enabled: bool = False
    reverb_wet: float = .25
    reverb_decay: float = 1.2
    modulation_enabled: bool = False
    modulation_frequency: float = 40.0
    modulation_mix: float = .6
    compression_enabled: bool = False
    compression_threshold: float = -18.0
    compression_ratio: float = 3.0
    compression_makeup: float = 3.0
    chain_order: tuple = DEFAULT_CHAIN_ORDER

    def __post_init__(self):
        for key in ('enabled','telephone','independent_enabled','reverb_enabled','modulation_enabled','compression_enabled'):
            if not isinstance(getattr(self,key),bool):
                raise AudioError('音效开关必须为布尔值。')
        for key, low, high in [('rate',.5,2),('echo_delay',0,1),('echo_strength',0,.8),
             ('pitch_semitones',-12,12),('tempo',.5,2),('reverb_wet',0,1),('reverb_decay',.2,3),
             ('modulation_frequency',20,120),('modulation_mix',0,1),
             ('compression_threshold',-40,-6),('compression_ratio',1,8),('compression_makeup',0,12)]:
            value=getattr(self,key)
            if (not isinstance(value,Real) or isinstance(value,bool)
                    or not np.isfinite(value) or not low<=value<=high):
                raise AudioError(f'音效参数 {key} 必须在 {low} 到 {high} 范围内且为有限数值。')
        if not isinstance(self.echo_count,Integral) or isinstance(self.echo_count,bool) or not 1<=self.echo_count<=8:
            raise AudioError('回声次数必须为1–8的整数。')
        if abs(self.rate-round(self.rate,2))>1e-9:
            raise AudioError('基础倍率以0.01为步进。')
        order=self.chain_order
        if (not isinstance(order,tuple) or len(order)!=len(DEFAULT_CHAIN_ORDER)
                or any(not isinstance(key,str) for key in order) or set(order)!=set(DEFAULT_CHAIN_ORDER)):
            raise AudioError('串联顺序必须恰好包含每个已知模块一次。')


@dataclass(frozen=True)
class EffectsResult:
    clip: AudioClip
    elapsed_seconds: float
    clipped_samples: int


def _check_length(count):
    if count>MAX_OUTPUT_SAMPLES:
        raise AudioError('音效中间结果或输出超过128 MiB缓存上限，请截短输入或提高倍率。')


def _predicted_length(count, sample_rate, settings):
    _check_length(count)
    if not settings.enabled:
        return count
    for key in settings.chain_order:
        if key=='pitch_tempo' and settings.independent_enabled:
            _check_length(max(1,round(count*2**(settings.pitch_semitones/12))))
            count=max(1,round(count/settings.tempo))
        elif key=='echo' and settings.echo_strength:
            count+=round(settings.echo_delay*sample_rate)*settings.echo_count
        elif key=='reverb' and settings.reverb_enabled and settings.reverb_wet:
            count+=max(1,int(np.ceil(settings.reverb_decay*sample_rate)))-1
        _check_length(count)
    return count


def apply_effects(clip: AudioClip, settings: EffectsSettings = EffectsSettings()) -> EffectsResult:
    started=perf_counter()
    ratio=Fraction(str(round(settings.rate,2))) if settings.enabled else Fraction(1)
    count=(len(clip.samples)*ratio.denominator+ratio.numerator-1)//ratio.numerator
    expected=_predicted_length(count,clip.sample_rate,settings)
    values=clip.samples.astype(np.float64)
    if settings.enabled:
        if ratio!=1:
            values=resample_poly(values,ratio.denominator,ratio.numerator)
        for key in settings.chain_order:
            if key=='pitch_tempo' and settings.independent_enabled:
                if settings.pitch_semitones:
                    values=pitch_shift(values,settings.pitch_semitones,clip.sample_rate)
                if settings.tempo!=1:
                    values=time_stretch(values,settings.tempo,clip.sample_rate)
            elif key=='telephone' and settings.telephone:
                frequencies=[min(300,clip.sample_rate*.1),min(3400,clip.sample_rate*.4)]
                values=sosfilt(butter(4,frequencies,btype='bandpass',fs=clip.sample_rate,output='sos'),values)
            elif key=='modulation' and settings.modulation_enabled:
                values=ring_modulate(values,clip.sample_rate,settings.modulation_frequency,settings.modulation_mix)
            elif key=='compression' and settings.compression_enabled:
                values=compress(values,clip.sample_rate,settings.compression_threshold,
                                settings.compression_ratio,settings.compression_makeup)
            elif key=='echo' and settings.echo_strength:
                delay=round(settings.echo_delay*clip.sample_rate)
                if delay:
                    echoed=np.zeros(len(values)+delay*settings.echo_count,dtype=np.float64)
                    echoed[:len(values)]=values
                    for repeat in range(1,settings.echo_count+1):
                        start=repeat*delay
                        echoed[start:start+len(values)]+=settings.echo_strength**repeat*values
                    values=echoed
                else:
                    values*=sum(settings.echo_strength**repeat for repeat in range(settings.echo_count+1))
            elif key=='reverb' and settings.reverb_enabled:
                values=room_reverb(values,clip.sample_rate,settings.reverb_wet,settings.reverb_decay)
    if len(values)!=expected or not np.isfinite(values).all():
        raise AudioError('音效结果长度或数值无效，请检查参数。')
    clipped=int(np.count_nonzero(np.abs(values)>1))
    np.clip(values,-1,1,out=values)
    output=AudioClip(values,clip.sample_rate,Path(clip.name).stem+'_音效.wav')
    return EffectsResult(output,perf_counter()-started,clipped)
