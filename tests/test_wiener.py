"""新增维纳的真实输出、非平稳噪声与语音保护回归；不读取私人参考。"""
import importlib
import numpy as np
import pytest
from voice_effect_lab.audio.data import AudioClip, AudioError
from voice_effect_lab.audio.denoise import spectral_subtract
from test_denoise import snr


def apply(clip,noise=None,strength=60):
    return importlib.import_module('voice_effect_lab.audio.wiener').wiener_denoise(clip,noise,strength)


def independent_case(kind):
    rate=16000;rng=np.random.default_rng(73017);t=np.arange(rate*4)/rate
    envelope=.4+.6*(np.sin(2*np.pi*2.3*t)>.1)
    envelope[t<.35]=0
    clean=envelope*(.16*np.sin(2*np.pi*510*t)+.065*np.sin(2*np.pi*1020*t))
    amplitude=np.where(t<2,.025,.085) if kind=='varying' else .045
    noise=rng.normal(size=len(t))*amplitude
    if kind=='colored':noise=np.convolve(noise,np.ones(3)/3,mode='same')
    return clean,AudioClip(clean+noise,rate,'独立合成.wav')


@pytest.mark.parametrize('kind',['white','varying','colored'])
def test_wiener_beats_spectral_error_and_preserves_signal(kind):
    clean,source=independent_case(kind);before=source.samples.copy()
    out=apply(source)
    old=spectral_subtract(source).clip.samples
    assert snr(clean,out.clip.samples)>snr(clean,old)+1
    assert np.dot(out.clip.samples,clean)/np.dot(clean,clean)>.90
    assert out.clip.sample_rate==source.sample_rate and len(out.clip.samples)==len(source.samples)
    assert out.algorithm=='wiener' and out.estimated_frames>0
    np.testing.assert_array_equal(source.samples,before)
    if kind=='varying':
        assert snr(clean[40000:],out.clip.samples[40000:])>snr(clean[40000:],old[40000:])+3


def test_continuous_clean_tone_is_not_eaten_by_tracker():
    t=np.arange(16000*5)/16000
    source=AudioClip(.25*np.sin(2*np.pi*440*t),16000,'持续纯音.wav')
    result=apply(source).clip.samples
    assert np.dot(result[-16000:],source.samples[-16000:])/np.dot(source.samples[-16000:],source.samples[-16000:])>.95


def test_digital_padding_does_not_hide_active_noise():
    clean,source=independent_case('white')
    x=source.samples.copy();x[:5600]=0
    source=AudioClip(x,16000,'补零.wav')
    out=apply(source).clip.samples
    assert snr(clean,out)>snr(clean,source.samples)+4
    np.testing.assert_allclose(out[:4000],0,atol=1e-7)


@pytest.mark.parametrize('length',[1,7,511,8003])
def test_zero_reference_is_exact_bypass_including_boundaries(length):
    source=AudioClip(np.random.default_rng(5).uniform(-.5,.5,length),8000,'边界.wav')
    out=apply(source,AudioClip(np.zeros(800),8000,'零参考.wav'))
    np.testing.assert_array_equal(out.clip.samples,source.samples)
    assert out.algorithm=='wiener' and out.noise_mode=='reference'


def test_mismatched_reference_rate_reduces_error_without_mutating_reference():
    clean,source=independent_case('white')
    noise=AudioClip(np.random.default_rng(19).normal(0,.045,44100),44100,'参考.wav')
    original=noise.samples.copy();out=apply(source,noise)
    assert snr(clean,out.clip.samples)>snr(clean,source.samples)+4
    np.testing.assert_array_equal(noise.samples,original)
    assert out.clip.sample_rate==16000 and out.clip.duration==source.duration


@pytest.mark.parametrize('rate',[100,8000,44100,192000])
def test_silence_and_low_rate_nonzero_short_input_finite(rate):
    silence=AudioClip(np.zeros(max(1,round(rate*.11))),rate,'静音.wav')
    np.testing.assert_array_equal(apply(silence).clip.samples,silence.samples)
    pulse=silence.samples.copy();pulse[len(pulse)//2]=.2
    source=AudioClip(pulse,rate,'短脉冲.wav')
    out=apply(source,AudioClip(np.zeros(max(1,round(rate*.11))),rate,'参考.wav'))
    assert np.isfinite(out.clip.samples).all() and len(out.clip.samples)==len(source.samples)


def test_zero_strength_tiny_input_and_half_strength_mix():
    source=AudioClip([.25],16000,'单样本.wav')
    np.testing.assert_array_equal(apply(source,strength=0).clip.samples,source.samples)
    _,source=independent_case('white')
    full=apply(source).clip.samples
    half=apply(source,strength=30).clip.samples
    np.testing.assert_allclose(half,(full+source.samples)*.5,atol=2e-7)


@pytest.mark.parametrize('strength',[-1,101,float('nan'),float('inf'),True,'60'])
def test_invalid_strength_is_rejected(strength):
    with pytest.raises(AudioError,match='强度'):
        apply(AudioClip(np.zeros(1600),16000,'音频.wav'),strength=strength)


def test_short_automatic_and_reference_report_minimum():
    source=AudioClip([.1],16000,'短音频.wav')
    with pytest.raises(AudioError,match='0.10'):apply(source)
    with pytest.raises(AudioError,match='0.10'):apply(source,source)


def test_dispatch_keeps_spectral_exact_and_rejects_unknown_method():
    denoise=importlib.import_module('voice_effect_lab.audio.denoising').denoise_audio
    _,source=independent_case('white')
    np.testing.assert_array_equal(denoise(source,method='spectral').clip.samples,spectral_subtract(source).clip.samples)
    assert denoise(source).algorithm=='wiener'
    with pytest.raises(AudioError):denoise(source,method='unrecognized')
