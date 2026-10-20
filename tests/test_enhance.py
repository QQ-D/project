"""增强DSP的幅度、频率响应和边界；不以更响冒充更高语音质量。"""
import numpy as np
import pytest
from voice_effect_lab.audio.data import AudioClip, AudioError


def test_neutral_settings_without_normalization_are_exact_bypass():
    from voice_effect_lab.audio.enhance import EnhancementSettings, enhance_audio
    clip=AudioClip(np.random.default_rng(4).uniform(-.8,.8,8003),8000,'neutral.wav')
    result=enhance_audio(clip,EnhancementSettings(normalize_peak=False))
    np.testing.assert_array_equal(result.clip.samples,clip.samples)


def test_disabled_enhancement_bypasses_even_non_neutral_settings():
    from voice_effect_lab.audio.enhance import EnhancementSettings, enhance_audio
    clip=AudioClip(np.array([.2,-.5,.1]),8000,'bypass.wav')
    result=enhance_audio(clip,EnhancementSettings(enabled=False,gain=2,clarity=100,low_db=12))
    np.testing.assert_array_equal(result.clip.samples,clip.samples)
    assert result.clipped_samples==0


def test_normalization_reaches_target_then_applies_gain_without_mutating_source():
    from voice_effect_lab.audio.enhance import EnhancementSettings, enhance_audio
    clip=AudioClip(np.array([0,.25,-.5]),16000,'peak.wav')
    result=enhance_audio(clip,EnhancementSettings(gain=.5))
    np.testing.assert_allclose(result.clip.samples,[0,.2375,-.475],atol=1e-7)
    np.testing.assert_array_equal(clip.samples,[0,.25,-.5])
    assert result.input_peak==.5
    assert result.output_peak==pytest.approx(.475,abs=1e-7)


def test_gain_is_linear_when_normalization_is_off():
    from voice_effect_lab.audio.enhance import EnhancementSettings, enhance_audio
    clip=AudioClip(np.array([.2,-.4]),8000,'gain.wav')
    result=enhance_audio(clip,EnhancementSettings(normalize_peak=False,gain=1.5))
    np.testing.assert_allclose(result.clip.samples,[.3,-.6],atol=1e-7)


def test_clipping_is_counted_and_mute_is_finite():
    from voice_effect_lab.audio.enhance import EnhancementSettings, enhance_audio
    clip=AudioClip(np.array([.5,-.5,0]),8000,'clip.wav')
    result=enhance_audio(clip,EnhancementSettings(gain=2))
    np.testing.assert_array_equal(result.clip.samples,[1,-1,0])
    assert result.clipped_samples==2
    muted=enhance_audio(clip,EnhancementSettings(gain=0,clarity=100))
    np.testing.assert_array_equal(muted.clip.samples,[0,0,0])


@pytest.mark.parametrize('rate,length',[(8000,1),(16000,3),(44100,137),(48000,48003),(1000,1003)])
def test_short_silent_and_low_rate_inputs_are_finite_and_keep_shape(rate,length):
    from voice_effect_lab.audio.enhance import EnhancementSettings, enhance_audio
    clip=AudioClip(np.zeros(length),rate,'silence.wav')
    result=enhance_audio(clip,EnhancementSettings(clarity=100,low_db=12,mid_db=-12,high_db=12))
    np.testing.assert_array_equal(result.clip.samples,clip.samples)
    assert result.clip.sample_rate==rate
    assert result.clipped_samples==0


def tone_gain(result,frequency,rate):
    # 舍去前0.25秒的因果滤波瞬态，以正余弦投影估计稳态幅度。
    data=result.clip.samples[rate//4:]
    times=np.arange(len(data))/rate
    return 2*np.hypot(np.dot(data,np.sin(2*np.pi*frequency*times)),
                      np.dot(data,np.cos(2*np.pi*frequency*times)))/len(data)/.05


@pytest.mark.parametrize('parameter,frequency',[('low_db',60),('mid_db',1500),('high_db',6500)])
@pytest.mark.parametrize('db',[-6,6])
def test_each_eq_band_applies_requested_tone_gain(parameter,frequency,db):
    from voice_effect_lab.audio.enhance import EnhancementSettings, enhance_audio
    rate=16000;t=np.arange(rate*2)/rate
    clip=AudioClip(.05*np.sin(2*np.pi*frequency*t),rate,'tone.wav')
    result=enhance_audio(clip,EnhancementSettings(normalize_peak=False,**{parameter:db}))
    measured=20*np.log10(tone_gain(result,frequency,rate))
    assert measured==pytest.approx(db,abs=.15)


def test_clarity_shifts_low_and_mid_bands_without_losing_duration():
    from voice_effect_lab.audio.enhance import EnhancementSettings, enhance_audio
    rate=16000;t=np.arange(rate*2)/rate
    low=AudioClip(.05*np.sin(2*np.pi*60*t),rate,'low.wav')
    mid=AudioClip(.05*np.sin(2*np.pi*1500*t),rate,'mid.wav')
    settings=EnhancementSettings(normalize_peak=False,clarity=100)
    assert 20*np.log10(tone_gain(enhance_audio(low,settings),60,rate)) < -2.5
    assert 20*np.log10(tone_gain(enhance_audio(mid,settings),1500,rate)) > 3.5


@pytest.mark.parametrize('kwargs',[{'gain':-1},{'gain':2.01},{'gain':float('nan')},
    {'clarity':101},{'clarity':-1},{'low_db':13},{'mid_db':-13},{'high_db':float('inf')}])
def test_invalid_parameters_are_rejected(kwargs):
    from voice_effect_lab.audio.enhance import EnhancementSettings
    with pytest.raises(AudioError): EnhancementSettings(**kwargs)
