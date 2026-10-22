"""公开中文的背景残留回归；干净参考只参与测量，不传给降噪。"""
from pathlib import Path
import numpy as np
import pytest
from scipy.signal import butter, sosfilt, resample_poly
from voice_effect_lab.audio.data import AudioClip, load_wav
from voice_effect_lab.audio.wiener import wiener_denoise


@pytest.mark.parametrize('seed',[371,9209])
@pytest.mark.parametrize('kind',['fan','hiss'])
def test_background_is_reduced_without_muting_chinese_speech(seed,kind):
    clean=load_wav(Path(__file__).resolve().parents[1]/'artifacts/中文TTS_公开测试句.wav')
    rate=16000
    x=resample_poly(clean.samples,rate,clean.sample_rate).astype(float)*.7
    t=np.arange(len(x))/rate
    noise=np.random.default_rng(seed).normal(size=len(x))
    if kind=='fan':
        noise=sosfilt(butter(3,850,fs=rate,output='sos'),noise)
        noise+=.65*np.sin(2*np.pi*139*t)+.2*np.sin(2*np.pi*278*t)
    else:
        noise=sosfilt(butter(2,700,fs=rate,btype='high',output='sos'),noise)
    noise*=np.sqrt(np.mean(x*x)/np.mean(noise*noise)/10**.5)
    source=AudioClip(x+noise,rate,'独立中文背景回归.wav')
    result=wiener_denoise(source).clip
    pause=np.convolve(x*x,np.ones(640)/640,'same')<1e-9
    speech=np.convolve(x*x,np.ones(320)/320,'same')>1e-5
    attenuation=10*np.log10(np.sum(noise[pause]**2)/np.sum(result.samples[pause]**2))
    projection=np.dot(result.samples[speech],x[speech])/np.dot(x[speech],x[speech])
    snr=10*np.log10(np.sum(x*x)/np.sum((result.samples-x)**2))
    assert attenuation>12,(kind,attenuation)
    assert projection>.84,(kind,projection)
    assert snr>8,(kind,snr)
    assert result.sample_rate==rate and len(result.samples)==len(x)
    np.testing.assert_array_equal(source.samples,(x+noise).astype(np.float32))


def test_quiet_voiced_passage_is_not_assumed_to_be_a_noise_reference():
    rate=16000
    t=np.arange(rate*4)/rate
    envelope=np.where(t<1,.18,1.)
    clean=envelope*sum(.14/k*np.sin(2*np.pi*157*k*t) for k in range(1,8))
    result=wiener_denoise(AudioClip(clean,rate,'没有噪声的轻声与正常说话.wav')).clip.samples
    for part in [slice(rate//4,3*rate//4),slice(2*rate,3*rate)]:
        projection=np.dot(result[part],clean[part])/np.dot(clean[part],clean[part])
        assert projection>.9,projection


def test_broadband_voiced_quiet_passage_is_not_erased():
    rate=16000;t=np.arange(rate*4)/rate
    voiced=sum(.14/k*np.sin(2*np.pi*157*k*t) for k in range(1,8))
    breath=sosfilt(butter(3,[300,4000],fs=rate,btype='bandpass',output='sos'),np.random.default_rng(413).normal(size=len(t)))
    breath*=.5*np.sqrt(np.mean(voiced**2)/np.mean(breath**2))
    clean=np.where(t<1,.18,1.)*(voiced+breath)
    output=wiener_denoise(AudioClip(clean,rate,'无噪声含气声的轻声.wav')).clip.samples
    part=slice(rate//4,3*rate//4)
    assert np.dot(output[part],clean[part])/np.dot(clean[part],clean[part])>.85


def test_noise_floor_can_drop_after_fan_stops():
    rate=16000;t=np.arange(rate*6)/rate
    clean=sum(.04/k*np.sin(2*np.pi*157*k*t) for k in range(1,8))
    clean*=np.where(t<1,0,np.where(t<4,5,1))
    noise=sosfilt(butter(3,850,fs=rate,output='sos'),np.random.default_rng(341).normal(size=len(t)))
    noise+=.65*np.sin(2*np.pi*139*t)+.2*np.sin(2*np.pi*278*t)
    noise*=.015/np.sqrt(np.mean(noise**2));noise[t>=4]=0
    output=wiener_denoise(AudioClip(clean+noise,rate,'风扇停止后轻声.wav')).clip.samples
    for start in [4.1,4.75,5.25]:
        part=slice(round(start*rate),round((start+.5)*rate))
        assert np.dot(output[part],clean[part])/np.dot(clean[part],clean[part])>.95
