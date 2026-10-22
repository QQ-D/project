"""决策导向维纳与自动噪声谱跟踪，原声/相位/时长保持不变。

DD先验SNR思想：Ephraim & Malah (1984), doi:10.1109/TASSP.1984.1164453。
SPP噪声估计思想：Gerkmann & Hendriks (2012), doi:10.1109/TASL.2011.2180896。
噪声谱保留下限与起音响应是本项目的改进，不声称完整复现论文或开源项目。
开源对照及局限见docs/维纳背景噪声优化.md。
"""
from numbers import Real
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import rfft, irfft
from scipy.ndimage import percentile_filter
from scipy.signal.windows import hann

from .data import AudioClip, AudioError
from .denoise import DenoiseResult, _reference_noise_power, _frame_batches


def _protect_harmonic_peaks(power):
    """只限制明显高于宽邻域谱底的峰，避免把有色背景整体压低。"""
    floor=percentile_filter(power,percentile=20,size=31,mode='nearest')
    return np.where(power>8*np.maximum(floor,1e-20), np.minimum(power,2*floor), power)


def _stationary_bins(frames, window):
    """用约128ms局部平均的时间变化排除轻声/正常发声谐波。

    最多128个时间位置，每批16位置×16帧；不分配整段STFT。
    """
    if len(frames) < 32:
        return np.zeros(len(window)//2+1, dtype=bool)
    starts = np.linspace(0, len(frames)-16, min(128, len(frames)-15), dtype=int)
    averages = np.empty((len(starts), len(window)//2+1), dtype=np.float64)
    for first in range(0, len(starts), 16):
        indices = starts[first:first+16, None]+np.arange(16)
        spectrum = rfft(frames[indices.ravel()]*window, axis=-1)
        powers = spectrum.real**2+spectrum.imag**2
        averages[first:first+len(indices)] = powers.reshape(len(indices),16,-1).mean(axis=1)
    low, high = np.quantile(averages, [.1, .9], axis=0)
    return high <= 6*np.maximum(low, 1e-20)


def _wiener_noise_profile(samples, size, hop, window):
    """低能量帧初始化；有明显能量变化时保留部分原始噪声谱峰。

    每批至多256帧，避免长音频构造完整浮点STFT。数字零帧不作为底噪。
    能量对比只是参考可信度启发式，不保证所选帧没有弱语音。
    """
    if len(samples) < size:
        samples = np.pad(samples, (0, size-len(samples)))
    frames = np.lib.stride_tricks.sliding_window_view(samples, size)[::hop]
    energies = np.empty(len(frames), dtype=np.float64)
    for first, batch in _frame_batches(samples, size, hop):
        energies[first:first+len(batch)] = np.mean(batch.astype(np.float64)**2, axis=1)
    eligible = np.flatnonzero(energies > 0)
    if not len(eligible):
        eligible = np.arange(len(frames))
    count = max(1, int(np.ceil(len(eligible)*.1)))
    selected = eligible[np.argpartition(energies[eligible], count-1)[:count]]
    power = np.zeros(size//2+1, dtype=np.float64)
    for first in range(0, count, 256):
        spectrum = rfft(frames[selected[first:first+256]]*window, axis=-1)
        power += (spectrum.real**2+spectrum.imag**2).sum(axis=0, dtype=np.float64)
    power /= count
    low, high = np.quantile(energies[eligible], [.1, .9])
    # 不再把所有稳定窄峰一律当作人声：存在明显停顿线索时，
    # 保留原始估计的50%作为跟踪下限；近乎纯谐波的轻声段仍受保护。
    protected = _protect_harmonic_peaks(power)
    broadband = np.sum(protected) > .1*max(np.sum(power), 1e-20)
    floor = np.zeros_like(power)
    if broadband and high > 6*max(low, 1e-20):
        floor = .5*power*_stationary_bins(frames, window)
    return power, count, floor


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
    noise_floor=0.
    if automatic:
        noise,estimated,noise_floor=_wiener_noise_profile(clip.samples,size,hop,window)
        noise=np.maximum(_protect_harmonic_peaks(noise),np.minimum(noise_floor,noise))
    else:
        noise=_reference_noise_power(noise_clip,clip.sample_rate,size,hop,window)
    if not np.any(noise):
        return result(clip.samples,estimated)
    noise=np.maximum(noise.astype(np.float64),1e-20)
    observed_power=noise.copy()
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
                observed_power=.9*observed_power+.1*power
                gamma=power/np.maximum(noise,1e-20)
                probability=1/(1+(1+prior)*np.exp(-np.minimum(gamma*prior/(1+prior),700)))
                pmean=.9*pmean+.1*probability
                probability=np.where(pmean>.99,np.minimum(probability,.99),probability)
                estimate=probability*noise+(1-probability)*power
                noise=.8*noise+.2*estimate
                if frame_index%8==0:
                    # 实际观测下降时同步放松保底，不将消失的背景反复抬回。
                    floor=np.minimum(noise_floor,2*observed_power)
                    noise=np.maximum(_protect_harmonic_peaks(noise),floor)
            effective_noise=np.maximum(noise*noise_scale,1e-20)
            gamma=power/effective_noise
            # 高后验SNR时加快响应，减少固定0.98对辅音/起音的拖慢。
            speech_frame=np.mean(gamma>10)>.25
            alpha=np.where((gamma>10) & speech_frame,.9,.98)
            xi=alpha*previous/effective_noise+(1-alpha)*np.maximum(gamma-1,0)
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
