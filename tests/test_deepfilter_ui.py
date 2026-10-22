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


def select_deepfilter(window):
    window.tabs.setCurrentIndex(1)
    window.extension_tabs.setCurrentIndex(2)
    window.neural_engine_combo.setCurrentIndex(window.neural_engine_combo.findData('deepfilter'))


def test_extension_deepfilter_runs_without_training_or_noise_and_exports_same_cache(app, tmp_path):
    window = loaded_window(app, tmp_path)
    try:
        select_deepfilter(window)
        assert window.denoise_method_combo.currentData() == 'wiener'
        assert window.deepfilter_strength_spin.value() == 100 and window.neural_model is None
        assert not window.noise_reference_panel.isHidden()
        source = window.clip
        window.auto_noise_check.setChecked(False)
        window.noise_clip = AudioClip(np.zeros(1600), 16000, '不应使用的噪声参考')
        window.noise_confirmed = False
        assert not window._can_denoise() and window.neural_compare_button.isEnabled()
        window.run_neural_comparison()
        wait_idle(app, window, timeout=60)
        assert not window.last_error, window.last_error
        result = window.neural_deepfilter_clip
        assert result is window.neural_player._clip
        assert window.clip is source and window.current_clip is source and window.denoised_clip is None
        assert result.sample_rate == source.sample_rate and len(result.samples) == len(source.samples)
        assert np.isfinite(result.samples).all()
        np.testing.assert_array_equal(window.intelligent_plots.axes[0, 1].lines[0].get_ydata(), window.neural_deepfilter_analysis.wave_values)
        assert 'DeepFilterNet3' in window.neural_metrics_label.text()
        assert window.neural_metrics['deepfilter_snr_db'] is None
        path = tmp_path/'DF输出.wav'
        window.save_neural_to(path)
        wait_idle(app, window)
        rate, samples = wavfile.read(path)
        assert rate == source.sample_rate
        np.testing.assert_allclose(samples/32768, result.samples, atol=1/32768)
    finally:
        window.close()


def test_model_failure_preserves_comparison_and_basic_reset_is_independent(app, tmp_path, monkeypatch):
    from voice_effect_lab.ui import neural_controller as module
    window = loaded_window(app, tmp_path)
    try:
        select_deepfilter(window)
        window.run_neural_comparison()
        wait_idle(app, window, timeout=60)
        previous = window.neural_deepfilter_clip
        assert previous is not None and not window.last_error
        def fail(*args, **kwargs):
            raise RuntimeError('公开模型进程失败测试')
        monkeypatch.setattr(module, 'deepfilter_denoise', fail)
        window.run_neural_comparison()
        wait_idle(app, window)
        assert window.neural_deepfilter_clip is previous and window.neural_player._clip is previous
        assert '模型进程失败' in window.last_error and window.neural_compare_button.isEnabled()
        window.denoise_method_combo.setCurrentIndex(window.denoise_method_combo.findData('spectral'))
        window.denoise_spin.setValue(20)
        window.reset_processing()
        assert window.denoise_method_combo.currentData() == 'wiener'
        assert window.denoise_spin.value() == 60 and window.denoise_slider.value() == 60
        assert not window.noise_reference_panel.isHidden()
        assert window.neural_deepfilter_clip is previous
    finally:
        window.close()


def test_sending_deepfilter_output_then_enhancement_and_effects_use_that_source(app, tmp_path):
    from voice_effect_lab.audio.enhance import enhance_audio
    from voice_effect_lab.audio.effects import apply_effects
    window = loaded_window(app, tmp_path)
    try:
        select_deepfilter(window)
        window.run_neural_comparison()
        wait_idle(app, window, timeout=60)
        assert not window.last_error, window.last_error
        result = window.neural_deepfilter_clip
        window.send_neural_to_workbench()
        assert window.clip is result and window.neural_deepfilter_clip is None
        window.enhance_check.setChecked(True)
        window.gain_spin.setValue(.8)
        window.echo_strength_spin.setValue(.3)
        window.start_effects()
        wait_idle(app, window, timeout=60)
        assert not window.last_error, window.last_error
        assert window.denoised_clip is None and window.enhanced_clip is not None
        assert window.current_clip is window.effected_clip and window.clip is result
        assert not window._effects_from_denoise and not window._enhanced_from_denoise
        expected = apply_effects(enhance_audio(result, window._enhancement_settings()).clip, window._effect_settings()).clip
        np.testing.assert_array_equal(window.effected_clip.samples, expected.samples)
    finally:
        window.close()
