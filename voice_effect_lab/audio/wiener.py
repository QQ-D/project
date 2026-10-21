"""决策导向维纳与自动噪声谱跟踪，原声/相位/时长保持不变。

DD先验SNR思想：Ephraim & Malah (1984), doi:10.1109/TASSP.1984.1164453。
SPP噪声估计思想：Gerkmann & Hendriks (2012), doi:10.1109/TASL.2011.2180896。
本实现另加频率中值上限保护，不声称完整复现上述论文算法。
"""
from numbers import Real
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import rfft, irfft
from scipy.ndimage import percentile_filter
from scipy.signal.windows import hann

from .data import AudioClip, AudioError
from .denoise import DenoiseResult, _automatic_noise_power, _reference_noise_power, _frame_batches


def _protect_harmonic_peaks(power):
    """只限制明显高于宽邻域谱底的峰，避免把有色背景整体压低。"""
    floor=percentile_filter(power,percentile=20,size=31,mode='nearest')
    return np.where(power>8*np.maximum(floor,1e-20), np.minimum(power,2*floor), power)


def wiener_denoise(clip: AudioClip, noise_clip: AudioClip | None = None, strength=60) -> DenoiseResult:
    started=perf_counter()
    if (not isinstance(strength,Real) or isinstance(strength,bool)
            or not np.isfinite(strength) or not 0<=strength<=100):
        raise AudioError('去噪强度必须是0–100的有限数值。')
    automatic=noise_clip is None
    mode='automatic' if automatic else 'reference'
    if not automatic and noise_clip.duration<.1:
        raise AudioError('底噪参考至少需要0.10秒。')
    name=Path(clip.name).stem+'_维纳去噪.wav'

    def result(values, estimated=0):
        values=np.asarray(values)
        if not np.isfinite(values).all():
            raise AudioError('维纳去噪产生非有限样本，请检查输入。')
        clipped=int(np.count_nonzero(np.abs(values)>1))
        output=AudioClip(np.clip(values,-1,1),clip.sample_rate,name)
        return DenoiseResult(output,perf_counter()-started,clipped,mode,estimated,'wiener')

    if strength==0:
        return result(clip.samples)
    if automatic and clip.duration<.1:
        raise AudioError('自动去噪至少需要0.10秒原声；请使用较长的音频。')
    size=max(32,min(4096,4*round(clip.sample_rate*.032/4)))
    hop=size//4
    window=hann(size,sym=False).astype(np.float32)
    estimated=0
    if automatic:
        noise,estimated=_automatic_noise_power(clip.samples,size,hop,window)
        noise=_protect_harmonic_peaks(noise)
    else:
        noise=_reference_noise_power(noise_clip,clip.sample_rate,size,hop,window)
    if not np.any(noise):
        return result(clip.samples,estimated)
    noise=np.maximum(noise.astype(np.float64),1e-20)
    previous=np.zeros_like(noise)
    pmean=np.zeros_like(noise)
    padded=np.pad(clip.samples,(size,size))
    output=np.zeros_like(padded)
    prior=10**1.5
    noise_scale=max(1.,float(strength)/60)
    wet=min(1.,float(strength)/60)
    for first,frames in _frame_batches(padded,size,hop):
        spectrum=rfft(frames*window,axis=-1)
        powers=spectrum.real**2+spectrum.imag**2
        gains=np.empty_like(powers)
        for offset,power in enumerate(powers):
            frame_index=first+offset
            start=frame_index*hop
            if automatic and size<=start<=len(clip.samples) and np.any(power):
                gamma=power/np.maximum(noise,1e-20)
                probability=1/(1+(1+prior)*np.exp(-np.minimum(gamma*prior/(1+prior),700)))
                pmean=.9*pmean+.1*probability
                probability=np.where(pmean>.99,np.minimum(probability,.99),probability)
                estimate=probability*noise+(1-probability)*power
                noise=.8*noise+.2*estimate
                if frame_index%8==0:
                    noise=_protect_harmonic_peaks(noise)
            effective_noise=np.maximum(noise*noise_scale,1e-20)
            gamma=power/effective_noise
            xi=.98*previous/effective_noise+.02*np.maximum(gamma-1,0)
            gain=np.maximum(xi/(1+xi),.05)
            previous=gain*gain*power
            gains[offset]=gain
        restored=irfft(spectrum*gains,n=size,axis=-1)*window
        for offset,frame in enumerate(restored):
            start=(first+offset)*hop
            output[start:start+size]+=frame
    samples=output[size:size+len(clip.samples)]/1.5
    if wet<1:
        samples=wet*samples+(1-wet)*clip.samples
    return result(samples,estimated)
