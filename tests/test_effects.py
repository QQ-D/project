"""捕获倍率未落入样本、回声截尾、带通失效与超大分配等故障。"""
import importlib
import numpy as np
import pytest
from voice_effect_lab.audio.data import AudioClip, AudioError


def api():
    module=importlib.import_module('voice_effect_lab.audio.effects')
    return module.EffectsSettings,module.apply_effects


def clip(values, rate=8000):
    return AudioClip(np.asarray(values),rate,'合成.wav')


def test_neutral_and_disabled_preserve_exact_samples():
    Settings,apply=api();source=clip([.1,-.2,.3])
    for settings in [Settings(),Settings(enabled=False,rate=2,telephone=True,echo_strength=.8)]:
        result=apply(source,settings)
        np.testing.assert_array_equal(result.clip.samples,source.samples)
        assert result.clip is not source
        assert result.clip.sample_rate==8000
        assert not result.clip.samples.flags.writeable
        assert result.clipped_samples==0


@pytest.mark.parametrize('speed,want_length,want_hz',[(2,8000,880),(.5,32000,220),(1.35,11852,594)])
def test_speed_changes_samples_pitch_and_duration(speed,want_length,want_hz):
    Settings,apply=api();t=np.arange(16000)/8000
    source=clip(.2*np.sin(2*np.pi*440*t))
    result=apply(source,Settings(rate=speed)).clip
    assert len(result.samples)==want_length
    assert result.sample_rate==8000
    frequencies=np.fft.rfftfreq(len(result.samples),1/8000)
    peak=frequencies[np.argmax(np.abs(np.fft.rfft(result.samples)))]
    assert abs(peak-want_hz)<1
    np.testing.assert_array_equal(source.samples,clip(.2*np.sin(2*np.pi*440*t)).samples)


def test_echo_retains_entire_tail_and_exact_delay():
    Settings,apply=api()
    result=apply(clip([.5,0],10),Settings(echo_delay=.2,echo_strength=.4)).clip
    np.testing.assert_allclose(result.samples,[.5,0,.2,0],atol=1e-7)
    assert result.sample_rate==10


def test_zero_delay_overlaps_without_extra_tail_and_reports_clipping():
    Settings,apply=api()
    result=apply(clip([.8,-.8,.1]),Settings(echo_delay=0,echo_strength=.8))
    np.testing.assert_allclose(result.clip.samples,[1,-1,.18],atol=1e-7)
    assert result.clipped_samples==2


def test_zero_strength_does_not_append_silence():
    Settings,apply=api()
    result=apply(clip([.5,-.3]),Settings(echo_delay=1)).clip
    np.testing.assert_array_equal(result.samples,clip([.5,-.3]).samples)


@pytest.mark.parametrize('hz,low,high',[(60,0,.01),(1000,.19,.21),(6000,0,.01)])
def test_telephone_keeps_mid_band_and_attenuates_outside(hz,low,high):
    Settings,apply=api();rate=16000;t=np.arange(rate*2)/rate
    result=apply(clip(.2*np.sin(2*np.pi*hz*t),rate),Settings(telephone=True)).clip
    amplitude=np.sqrt(2)*np.std(result.samples[rate:])
    assert low<=amplitude<=high


@pytest.mark.parametrize('rate',[1,1000,8000,11025,44100,192000])
def test_nonzero_short_low_rate_audio_remains_finite(rate):
    Settings,apply=api()
    source=clip([.2],rate)
    result=apply(source,Settings(rate=2,telephone=True,echo_delay=0,echo_strength=.8)).clip
    assert len(result.samples)==1
    assert np.isfinite(result.samples).all()


@pytest.mark.parametrize('values',[{'rate':0},{'rate':2.1},{'rate':np.nan},{'rate':True},
                                  {'rate':1.234},{'telephone':1},{'enabled':0},
                                  {'echo_delay':-1},{'echo_delay':np.inf},{'echo_strength':.81}])
def test_invalid_settings_rejected(values):
    Settings,_=api()
    with pytest.raises(AudioError):Settings(**values)


def test_output_bound_checked_before_resampling(monkeypatch):
    Settings,apply=api()
    module=importlib.import_module('voice_effect_lab.audio.effects')
    monkeypatch.setattr(module,'MAX_OUTPUT_SAMPLES',5)
    with pytest.raises(AudioError,match='128 MiB'):
        apply(clip([.1]*4),Settings(rate=.5))
