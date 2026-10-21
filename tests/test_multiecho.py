"""有限多重回声的延迟、几何衰减、重叠与完整尾音。"""
import numpy as np
import pytest
from voice_effect_lab.audio.data import AudioClip,AudioError
from voice_effect_lab.audio.effects import EffectsSettings,apply_effects


def test_three_echoes_have_exact_positions_and_geometric_amplitudes():
    source=AudioClip([.5,0],10,'脉冲.wav')
    out=apply_effects(source,EffectsSettings(echo_delay=.2,echo_strength=.5,echo_count=3))
    np.testing.assert_allclose(out.clip.samples,[.5,0,.25,0,.125,0,.0625,0],atol=1e-7)
    np.testing.assert_array_equal(source.samples,[.5,0])
    assert out.clip.sample_rate==10 and out.clipped_samples==0


def test_overlapping_repeats_use_original_input_and_preserve_last_tail():
    source=AudioClip([.1,.2,.3],10,'重叠.wav')
    out=apply_effects(source,EffectsSettings(echo_delay=.1,echo_strength=.5,echo_count=3)).clip
    np.testing.assert_allclose(out.samples,[.1,.25,.425,.2125,.1,.0375],atol=1e-7)


def test_zero_delay_sums_all_echoes_and_counts_clipped_samples():
    source=AudioClip([.8,-.8,.1],8000,'零延迟.wav')
    out=apply_effects(source,EffectsSettings(echo_delay=0,echo_strength=.5,echo_count=3))
    np.testing.assert_allclose(out.clip.samples,[1,-1,.1875],atol=1e-7)
    assert out.clipped_samples==2


@pytest.mark.parametrize('settings',[{'echo_strength':0},{'enabled':False,'echo_strength':.8}])
def test_bypass_does_not_add_tail(settings):
    source=AudioClip([.2,-.1],8000,'旁路.wav')
    out=apply_effects(source,EffectsSettings(echo_delay=1,echo_count=8,**settings)).clip
    np.testing.assert_array_equal(out.samples,source.samples)


def test_speed_then_multiple_echo_lengths_use_output_sample_rate():
    source=AudioClip([.1]*10,10,'倍率.wav')
    out=apply_effects(source,EffectsSettings(rate=2,echo_delay=.2,echo_strength=.2,echo_count=3)).clip
    assert len(out.samples)==11 and out.sample_rate==10


@pytest.mark.parametrize('count',[0,9,True,2.5,'3',None])
def test_invalid_echo_count_rejected(count):
    with pytest.raises(AudioError,match='次数'):
        EffectsSettings(echo_count=count)


def test_total_tail_is_bounded_before_resampling(monkeypatch):
    from voice_effect_lab.audio import effects
    monkeypatch.setattr(effects,'MAX_OUTPUT_SAMPLES',8)
    with pytest.raises(AudioError,match='128 MiB'):
        apply_effects(AudioClip([.2,.1,.1],10,'长度.wav'),
                      EffectsSettings(echo_delay=.2,echo_strength=.5,echo_count=3))
