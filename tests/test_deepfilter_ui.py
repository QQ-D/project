"""默认模型去噪、无参考策略和既有四阶段链路的真实Qt验收。"""
import numpy as np
from scipy.io import wavfile

from test_ui import wait_idle
from voice_effect_lab.audio.data import AudioClip
from voice_effect_lab.ui.main_window import MainWindow


def loaded_window(app, tmp_path):
    rate = 16000
    times = np.arange(rate, dtype=float)/rate
    samples = .12*np.sin(2*np.pi*220*times) + .015*np.random.default_rng(2026).standard_normal(rate)
    path = tmp_path/'公开合成DF测试.wav'
    wavfile.write(path, rate, samples.astype(np.float32))
    window = MainWindow()
    window.show()
    window.load_path(path)
    wait_idle(app, window)
    return window


def test_default_deepfilter_runs_without_noise_and_exports_same_cache(app, tmp_path):
    window = loaded_window(app, tmp_path)
    try:
        assert window.denoise_method_combo.currentData() == 'deepfilter'
        assert window.denoise_spin.value() == 100
        assert window.noise_reference_panel.isHidden()
        assert not window.auto_noise_check.isEnabled()
        source = window.clip
        window.auto_noise_check.setChecked(False)
        window.noise_clip = AudioClip(np.zeros(1600), 16000, '不应使用的噪声参考')
        window.noise_confirmed = False
        assert window._can_denoise() and window._denoise_snapshot()[0]
        window.start_denoise()
        wait_idle(app, window, timeout=60)
        assert not window.last_error, window.last_error
        result = window.denoised_clip
        assert result is window.current_clip and window.player._clip is result
        assert window.clip is source and result.sample_rate == source.sample_rate
        assert len(result.samples) == len(source.samples) and np.isfinite(result.samples).all()
        assert 'DeepFilterNet3' in window.denoise_status_label.text()
        assert 'dbaf' in window.denoise_status_label.text()
        assert '估计' not in window.denoise_status_label.text()
        path = tmp_path/'DF输出.wav'
        window.save_to(path)
        wait_idle(app, window)
        rate, samples = wavfile.read(path)
        assert rate == source.sample_rate
        np.testing.assert_allclose(samples/32768, result.samples, atol=1/32768)
    finally:
        window.close()


def test_model_failure_preserves_result_and_reset_uses_full_deepfilter(app, tmp_path, monkeypatch):
    from voice_effect_lab.ui import processing_controller as module
    window = loaded_window(app, tmp_path)
    try:
        window.start_denoise()
        wait_idle(app, window, timeout=60)
        previous = window.denoised_clip
        assert previous is not None and not window.last_error
        def fail(*args):
            raise RuntimeError('公开模型进程失败测试')
        monkeypatch.setattr(module, 'denoise_audio', fail)
        window.start_denoise()
        wait_idle(app, window)
        assert window.denoised_clip is previous and window.current_clip is previous
        assert '模型进程失败' in window.last_error
        assert window.denoise_button.isEnabled()
        window.denoise_method_combo.setCurrentIndex(window.denoise_method_combo.findData('spectral'))
        assert not window.noise_reference_panel.isHidden()
        window.denoise_spin.setValue(20)
        window.reset_processing()
        assert window.denoise_method_combo.currentData() == 'deepfilter'
        assert window.denoise_spin.value() == 100 and window.denoise_slider.value() == 100
        assert window.noise_reference_panel.isHidden()
    finally:
        window.close()


def test_enhancement_and_effect_chain_use_deepfilter_input(app, tmp_path):
    window = loaded_window(app, tmp_path)
    try:
        source = window.clip
        window.use_denoise_check.setChecked(True)
        window.enhance_check.setChecked(True)
        window.gain_spin.setValue(.8)
        window.echo_strength_spin.setValue(.3)
        window.start_effects()
        wait_idle(app, window, timeout=60)
        assert not window.last_error, window.last_error
        assert window.denoised_clip is not None and window.enhanced_clip is not None
        assert 'DeepFilterNet3' in window.denoise_status_label.text()
        assert window.current_clip is window.effected_clip
        assert window._effects_from_denoise and window._enhanced_from_denoise
        assert window.clip is source
        window.denoise_spin.setValue(50)
        assert window.denoised_clip is None and window.enhanced_clip is None and window.effected_clip is None
    finally:
        window.close()
