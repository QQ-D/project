"""高级音效真实波形、串联顺序与资源边界。"""
import importlib
import numpy as np
import pytest
from voice_effect_lab.audio.data import AudioClip,AudioError
from voice_effect_lab.audio.effects import EffectsSettings,apply_effects


def module():return importlib.import_module('voice_effect_lab.audio.advanced')


def test_reverb_impulse_has_predelay_full_tail_and_deterministic_response():
    x=np.zeros(800);x[0]=.5;before=x.copy()
    out=module().room_reverb(x,8000,wet=.4,decay= .2)
    assert len(out)==2399 and out[0]==.3
    np.testing.assert_array_equal(out[1:100],0)
    assert np.sum(out[800:]**2)>1e-6
    np.testing.assert_array_equal(out,module().room_reverb(x,8000,wet=.4,decay=.2))
    np.testing.assert_array_equal(x,before)
    np.testing.assert_array_equal(module().room_reverb(x,8000,wet=0,decay=.2),x)


def test_ring_modulation_has_two_expected_sidebands():
    t=np.arange(8000)/8000;x=.2*np.sin(2*np.pi*440*t)
    out=module().ring_modulate(x,8000,frequency=50,mix=1)
    spectrum=np.abs(np.fft.rfft(out))
    assert set(np.argsort(spectrum)[-2:])=={390,490}
    assert spectrum[440]<1e-6
    np.testing.assert_array_equal(module().ring_modulate(x,8000,mix=0),x)


def test_compression_reduces_loud_part_and_keeps_quiet_part():
    x=np.r_[np.full(8000,.01),np.full(8000,.5)];before=x.copy()
    out=module().compress(x,8000,threshold_db=-18,ratio=4,makeup_db=0)
    assert .99<np.mean(out[4000:7000])/.01<1.01
    assert np.mean(out[-4000:])/.5<.5
    boosted=module().compress(x,8000,threshold_db=-18,ratio=4,makeup_db=6)
    np.testing.assert_allclose(boosted,out*10**.3,rtol=1e-12)
    np.testing.assert_array_equal(x,before)


def test_independent_neutral_and_disabled_advanced_modules_are_exact():
    source=AudioClip([.1,-.2,.3],8000,'旁路.wav')
    result=apply_effects(source,EffectsSettings(independent_enabled=True,pitch_semitones=0,tempo=1,
                          reverb_enabled=True,reverb_wet=0,modulation_enabled=True,modulation_mix=0,
                          compression_enabled=True,compression_ratio=1,compression_makeup=0))
    np.testing.assert_array_equal(result.clip.samples,source.samples)


def test_chain_order_predicts_and_preserves_tails_after_time_stretch():
    source=AudioClip(np.full(8000,.01),8000,'串联.wav')
    first=('pitch_tempo','telephone','modulation','compression','echo','reverb')
    second=('reverb','pitch_tempo','telephone','modulation','compression','echo')
    args=dict(independent_enabled=True,tempo=2,echo_delay=.1,echo_strength=.3,echo_count=3,
              reverb_enabled=True,reverb_wet=.3,reverb_decay=.2)
    assert len(apply_effects(source,EffectsSettings(chain_order=first,**args)).clip.samples)==7999
    assert len(apply_effects(source,EffectsSettings(chain_order=second,**args)).clip.samples)==7200


def test_nonlinear_compression_order_changes_result():
    source=AudioClip(np.full(8000,.3),8000,'非线性串联.wav')
    args=dict(compression_enabled=True,compression_threshold=-18,compression_ratio=4,compression_makeup=0,
              echo_delay=0,echo_strength=.8)
    early=('pitch_tempo','telephone','modulation','compression','echo','reverb')
    late=('pitch_tempo','telephone','modulation','echo','compression','reverb')
    a=apply_effects(source,EffectsSettings(chain_order=early,**args)).clip.samples
    b=apply_effects(source,EffectsSettings(chain_order=late,**args)).clip.samples
    assert np.mean(a[-4000:])>np.mean(b[-4000:])*1.2


@pytest.mark.parametrize('rate',[1,100,8000,48000])
def test_advanced_short_nonzero_and_silence_are_finite(rate):
    for values in [np.zeros(3),np.array([.1,-.1,.05])]:
        source=AudioClip(values,rate,'低采样率.wav')
        out=apply_effects(source,EffectsSettings(reverb_enabled=True,reverb_wet=.3,reverb_decay=.2,
                             modulation_enabled=True,compression_enabled=True))
        assert np.isfinite(out.clip.samples).all()
        assert len(out.clip.samples)==3+max(1,int(np.ceil(rate*.2)))-1
        if not np.any(values):np.testing.assert_array_equal(out.clip.samples,0)


@pytest.mark.parametrize('values',[{'pitch_semitones':13},{'tempo':.4},{'reverb_decay':0},
 {'reverb_wet':1.1},{'modulation_frequency':10},{'modulation_mix':-.1},
 {'compression_threshold':0},{'compression_ratio':9},{'compression_makeup':13},
 {'independent_enabled':1},{'reverb_enabled':1},{'compression_enabled':1},{'modulation_enabled':1},
 {'chain_order':('echo',)*6}])
def test_invalid_advanced_settings_rejected(values):
    with pytest.raises(AudioError):EffectsSettings(**values)


def test_total_reverb_tail_checked_before_processing(monkeypatch):
    from voice_effect_lab.audio import effects
    monkeypatch.setattr(effects,'MAX_OUTPUT_SAMPLES',1003)
    with pytest.raises(AudioError,match='128 MiB'):
        apply_effects(AudioClip(np.zeros(1000),1000,'资源.wav'),EffectsSettings(reverb_enabled=True,reverb_decay=.2))
