"""无独立参考去噪：检查减噪与信号保留，明确不是纯噪声识别。"""
import numpy as np
import pytest

from test_denoise import synthetic_case, snr
from test_noise_import import noise_file
from test_processing_ui import prepare_window, block_processing
from test_ui import wait_idle
from voice_effect_lab.audio.data import AudioClip, AudioError
from voice_effect_lab.audio.denoise import spectral_subtract


def test_automatic_noise_estimation_improves_known_reference_without_mutation():
    clean, source, _ = synthetic_case()
    before = source.samples.copy()
    result = spectral_subtract(source, None, 60)
    assert snr(clean, result.clip.samples)-snr(clean, source.samples) > 3
    assert .85 < np.dot(result.clip.samples, clean)/np.dot(clean, clean) < 1.05
    assert result.noise_mode == 'automatic'
    assert result.estimated_frames > 0
    assert result.clip.sample_rate == source.sample_rate
    assert len(result.clip.samples) == len(source.samples)
    np.testing.assert_array_equal(source.samples, before)


def test_automatic_mode_preserves_continuous_harmonics_without_silent_prefix():
    rate = 16000
    t = np.arange(rate*2)/rate
    clean = .2*np.sin(2*np.pi*440*t)+.08*np.sin(2*np.pi*880*t)
    source = AudioClip(clean+np.random.default_rng(3).normal(0, .04, len(t)), rate, '连续合成信号.wav')
    result = spectral_subtract(source, None, 60)
    assert np.dot(result.clip.samples, clean)/np.dot(clean, clean) > .85
    assert snr(clean, result.clip.samples)-snr(clean, source.samples) > 1


def test_automatic_clean_tone_is_not_treated_as_background_noise():
    t = np.arange(16000)/16000
    source = AudioClip(.3*np.sin(2*np.pi*440*t), 16000, '纯音.wav')
    result = spectral_subtract(source, None)
    assert np.dot(result.clip.samples, source.samples)/np.dot(source.samples, source.samples) > .95


def test_digital_zero_padding_does_not_hide_noise_in_the_active_audio():
    clean, source, _ = synthetic_case()
    samples = source.samples.copy()
    samples[:6400] = 0  # 模拟录音前补入的数字静音，不代表语音段的环境底噪。
    padded = AudioClip(samples, source.sample_rate, '数字静音开头.wav')
    result = spectral_subtract(padded, None)
    assert snr(clean, result.clip.samples)-snr(clean, padded.samples) > 1
    assert np.dot(result.clip.samples, clean)/np.dot(clean, clean) > .85


def test_automatic_silence_stays_exactly_silent():
    source = AudioClip(np.zeros(8003), 8000, '静音.wav')
    result = spectral_subtract(source, None)
    np.testing.assert_array_equal(result.clip.samples, source.samples)


@pytest.mark.parametrize('length', [1, 1599])
def test_automatic_mode_short_input_reports_clear_minimum(length):
    with pytest.raises(AudioError, match='自动.*0.10'):
        spectral_subtract(AudioClip(np.zeros(length), 16000, '过短.wav'), None)


def test_zero_strength_automatic_mode_bypasses_even_tiny_input():
    source = AudioClip(np.array([.25]), 16000, '单样本.wav')
    result = spectral_subtract(source, None, 0)
    np.testing.assert_array_equal(result.clip.samples, [.25])


def test_fresh_window_can_denoise_with_just_source_and_no_noise_reference(app, tmp_path):
    window, _ = prepare_window(app, tmp_path)
    try:
        source = window.clip
        assert window.auto_noise_check.isChecked()
        assert window.noise_clip is None
        assert window.denoise_button.isEnabled()
        window.denoise_button.click()
        wait_idle(app, window)
        assert not window.last_error
        assert window.current_clip is window.denoised_clip
        assert window.player._clip is window.denoised_clip
        assert window.clip is source
        assert window.noise_clip is None
        assert not window.noise_confirmed
        assert '自动估计' in window.denoise_status_label.text()
        assert window.recorder.state == 'idle'
    finally:
        window.close()


def test_auto_manual_switch_invalidates_cache_and_manual_selection_selects_reference_mode(app, tmp_path):
    window, _ = prepare_window(app, tmp_path)
    try:
        window.start_denoise()
        wait_idle(app, window)
        assert window.denoised_clip is not None
        window.auto_noise_check.setChecked(False)
        assert window.denoised_clip is None
        assert window.stages.checkedId() == 0
        assert not window.denoise_button.isEnabled()
        window.set_noise_segment(0, .35, confirmed=True)
        assert not window.auto_noise_check.isChecked()
        window.start_denoise()
        wait_idle(app, window)
        assert '参考底噪' in window.denoise_status_label.text()
        window.clear_noise()
        assert window.auto_noise_check.isChecked()
        assert window.denoised_clip is None
        assert window.denoise_button.isEnabled()
    finally:
        window.close()


def test_explicit_auto_mode_ignores_unconfirmed_imported_reference(app, tmp_path):
    window, _ = prepare_window(app, tmp_path)
    try:
        window.load_noise_path(noise_file(tmp_path))
        wait_idle(app, window)
        reference = window.noise_clip
        assert not window.noise_confirmed
        assert not window.auto_noise_check.isChecked()
        assert not window.denoise_button.isEnabled()
        expected = spectral_subtract(window.clip, None, 60).clip
        window.auto_noise_check.setChecked(True)
        window.denoise_button.click()
        wait_idle(app, window)
        assert window.noise_clip is reference
        assert not window.noise_confirmed
        np.testing.assert_array_equal(window.denoised_clip.samples, expected.samples)
    finally:
        window.close()


def test_background_result_from_changed_noise_mode_is_discarded(app, tmp_path, monkeypatch):
    window, _ = prepare_window(app, tmp_path)
    release = block_processing(monkeypatch)
    try:
        window.start_denoise()
        assert window.busy
        assert not window.auto_noise_check.isEnabled()
        window.auto_noise_check.setChecked(False)
        release.set()
        wait_idle(app, window)
        assert window.denoised_clip is None
        assert window.current_clip is window.clip
        assert not window.denoise_button.isEnabled()
    finally:
        release.set()
        wait_idle(app, window)
        window.close()
