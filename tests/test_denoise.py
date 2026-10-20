"""有干净参考的人工加噪验证；不将合成指标当作真人语音效果。"""
import numpy as np
import pytest

from voice_effect_lab.audio.data import AudioClip, AudioError


def synthetic_case(rate=16000):
    rng = np.random.default_rng(20261020)
    t = np.arange(rate*3)/rate
    clean = (.20*np.sin(2*np.pi*440*t) + .08*np.sin(2*np.pi*880*t))
    clean *= (.65+.35*np.sin(2*np.pi*2*t)**2)
    clean[t < .4] = 0
    noisy = clean + rng.normal(0, .055, len(t))
    noise = rng.normal(0, .055, rate*2)
    return clean, AudioClip(noisy, rate, '人工加噪.wav'), AudioClip(noise, rate, '独立合成底噪.wav')


def snr(clean, actual):
    return 10*np.log10(np.sum(clean**2)/np.sum((actual-clean)**2))


def test_known_reference_noise_is_reduced_without_mutating_source():
    from voice_effect_lab.audio.denoise import spectral_subtract
    clean, source, noise = synthetic_case()
    before_source, before_noise = source.samples.copy(), noise.samples.copy()
    result = spectral_subtract(source, noise, 60)
    assert snr(clean, result.clip.samples)-snr(clean, source.samples) > 3
    assert np.std(result.clip.samples[1600:4800]) < np.std(source.samples[1600:4800])*.55
    # 语音段的投影幅度接近干净信号，不能用整体静音冒充去噪。
    assert .85 < np.dot(result.clip.samples, clean)/np.dot(clean, clean) < 1.05
    assert result.clip.sample_rate == 16000
    assert result.clip.samples.shape == source.samples.shape
    assert result.elapsed_seconds >= 0
    np.testing.assert_array_equal(source.samples, before_source)
    np.testing.assert_array_equal(noise.samples, before_noise)


@pytest.mark.parametrize('strength', [0, 0.0])
def test_zero_strength_is_exact_bypass(strength):
    from voice_effect_lab.audio.denoise import spectral_subtract
    _, source, noise = synthetic_case()
    result = spectral_subtract(source, noise, strength)
    np.testing.assert_array_equal(result.clip.samples, source.samples)


@pytest.mark.parametrize('length', [1, 7, 511, 8000, 8003])
def test_zero_noise_reference_reconstructs_full_signal_including_boundaries(length):
    from voice_effect_lab.audio.denoise import spectral_subtract
    values = np.random.default_rng(1).uniform(-.6, .6, length).astype(np.float32)
    source = AudioClip(values, 8000, '边界.wav')
    noise = AudioClip(np.zeros(800), 8000, '零底噪.wav')
    result = spectral_subtract(source, noise)
    np.testing.assert_allclose(result.clip.samples, values, atol=3e-7)
    assert len(result.clip.samples) == length


def test_silent_source_has_no_nan_or_invented_signal():
    from voice_effect_lab.audio.denoise import spectral_subtract
    _, _, noise = synthetic_case()
    source = AudioClip(np.zeros(11003), 16000, '静音.wav')
    result = spectral_subtract(source, noise, 100)
    np.testing.assert_array_equal(result.clip.samples, source.samples)
    assert result.clipped_samples == 0


def test_mismatched_reference_rate_preserves_source_duration_and_rate():
    from voice_effect_lab.audio.denoise import spectral_subtract
    clean, source, _ = synthetic_case(rate=16000)
    noise = AudioClip(np.random.default_rng(7).normal(0, .055, 44100), 44100, '44k底噪.wav')
    result = spectral_subtract(source, noise, 60)
    assert result.clip.sample_rate == 16000
    assert result.clip.duration == source.duration
    assert snr(clean, result.clip.samples) > snr(clean, source.samples)


@pytest.mark.parametrize('value', [-1, 101, float('nan'), float('inf'), '60', None])
def test_invalid_strength_is_rejected(value):
    from voice_effect_lab.audio.denoise import spectral_subtract
    _, source, noise = synthetic_case()
    with pytest.raises(AudioError, match='强度'):
        spectral_subtract(source, noise, value)


def test_too_short_reference_is_rejected():
    from voice_effect_lab.audio.denoise import spectral_subtract
    _, source, _ = synthetic_case()
    with pytest.raises(AudioError, match='0.10'):
        spectral_subtract(source, AudioClip(np.zeros(1599), 16000, '短底噪.wav'))
