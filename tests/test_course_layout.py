"""课程分区、同源比较和模型结果独立失效的真实Qt回归。"""
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from test_deepfilter_ui import loaded_window
from test_ui import wait_idle
from voice_effect_lab.analysis import analyze
from voice_effect_lab.audio.data import AudioClip
from voice_effect_lab.audio.neural import train_model
from voice_effect_lab.ui.main_window import MainWindow


def choose(window, key):
    window.neural_engine_combo.setCurrentIndex(window.neural_engine_combo.findData(key))


def compare(app, window):
    window.run_neural_comparison()
    wait_idle(app, window, timeout=60)
    assert not window.last_error, window.last_error


def fake_df(monkeypatch):
    from voice_effect_lab.ui import neural_controller
    calls = []
    def process(clip, noise_clip=None, strength=100):
        calls.append((clip, noise_clip, strength))
        return SimpleNamespace(clip=AudioClip(clip.samples*.4, clip.sample_rate, '公开DF结果.wav'),
                               elapsed_seconds=.125, clipped_samples=0)
    monkeypatch.setattr(neural_controller, 'deepfilter_denoise', process)
    return calls


def test_basic_page_keeps_noise_reference_and_reset_is_traditional(app):
    window = MainWindow()
    try:
        assert {window.denoise_method_combo.itemData(i) for i in range(window.denoise_method_combo.count())} == {'wiener', 'spectral'}
        assert window.denoise_method_combo.currentData() == 'wiener'
        assert window.denoise_spin.value() == 60
        assert not window.noise_reference_panel.isHidden()
        assert window.auto_noise_check.isEnabled()
        window.denoise_method_combo.setCurrentIndex(1)
        window.denoise_spin.setValue(20)
        window.reset_processing()
        assert window.denoise_method_combo.currentData() == 'wiener'
        assert window.denoise_spin.value() == 60
    finally:
        window.close()


def test_extension_subpages_and_model_training_visibility(app):
    window = MainWindow()
    try:
        assert [window.extension_tabs.tabText(i) for i in range(3)] == ['高级音效', '中文 TTS', '智能降噪']
        assert window.neural_engine_combo.currentData() == 'deepfilter'
        assert window.neural_training_panel.isHidden()
        assert window.neural_training_panel.isAncestorOf(window.neural_training_load_button)
        assert window.neural_training_panel.isAncestorOf(window.neural_model_load_button)
        assert not window.neural_training_load_button.isVisibleTo(window.neural_panel)
        assert not window.deepfilter_parameters.isHidden()
        assert not window.neural_compare_button.isEnabled()
        choose(window, 'bp')
        assert not window.neural_training_panel.isHidden()
        assert window.deepfilter_parameters.isHidden()
        assert not window.neural_compare_button.isEnabled()
        assert window.neural_scope_label.text().find('教学') >= 0
    finally:
        window.close()


def test_df_without_training_shares_plot_player_export_and_preserves_basic(app, tmp_path, monkeypatch):
    window = loaded_window(app, tmp_path)
    try:
        calls = fake_df(monkeypatch)
        source = window.clip
        window.tabs.setCurrentIndex(1)
        window.extension_tabs.setCurrentIndex(2)
        assert window.neural_model is None and window.neural_compare_button.isEnabled()
        compare(app, window)
        result = window.neural_deepfilter_clip
        assert calls == [(source, None, 100)]
        assert window.clip is source and window.denoised_clip is None
        assert window.denoise_method_combo.currentData() == 'wiener'
        assert window.neural_player._clip is result
        np.testing.assert_array_equal(window.intelligent_plots.axes[0, 1].lines[0].get_ydata(), window.neural_deepfilter_analysis.wave_values)
        assert window.neural_metrics['deepfilter_snr_db'] is None
        assert window.neural_metrics['deepfilter_elapsed_seconds'] == .125
        path = tmp_path/'所选DF.wav'
        window.save_neural_to(path)
        wait_idle(app, window)
        rate, data = wavfile.read(path)
        assert rate == result.sample_rate
        np.testing.assert_allclose(data/32768, result.samples, atol=1/32768)
        window.send_neural_to_workbench()
        assert window.clip is result and window.tabs.currentIndex() == 0
        assert window.neural_deepfilter_clip is None
        assert window.denoise_method_combo.currentData() == 'wiener'
    finally:
        window.close()


def test_model_results_remain_independent_when_strength_or_model_changes(app, tmp_path, monkeypatch):
    window = loaded_window(app, tmp_path)
    try:
        fake_df(monkeypatch)
        compare(app, window)
        df = window.neural_deepfilter_clip
        choose(window, 'bp')
        assert not window.neural_compare_button.isEnabled()
        window._neural_adopt_model(train_model(epochs=1).model)
        compare(app, window)
        bp = window.neural_clip
        assert df is window.neural_deepfilter_clip
        window.deepfilter_strength_spin.setValue(80)
        assert window.neural_deepfilter_clip is None and window.neural_clip is bp
        choose(window, 'deepfilter')
        compare(app, window)
        df = window.neural_deepfilter_clip
        assert window.neural_clip is bp
        window.neural_strength_spin.setValue(25)
        assert window.neural_clip is None and window.neural_deepfilter_clip is df
        assert 'bp_elapsed_seconds' not in window.neural_metrics
        window._neural_adopt_model(train_model(epochs=1).model)
        assert window.neural_deepfilter_clip is df
        window.neural_selection_combo.setCurrentIndex(window.neural_selection_combo.findData('deepfilter'))
        assert window.neural_player._clip is df
        replacement = AudioClip(np.zeros(8000), 8000, '新原声')
        window._loaded((replacement, analyze(replacement)))
        assert window.neural_deepfilter_clip is None and window.neural_clip is None
        assert not window.neural_metrics and window.neural_player._clip is None
    finally:
        window.close()


def test_df_failure_keeps_previous_comparison_and_recovers_controls(app, tmp_path, monkeypatch):
    from voice_effect_lab.ui import neural_controller
    window = loaded_window(app, tmp_path)
    try:
        fake_df(monkeypatch)
        compare(app, window)
        previous = window.neural_deepfilter_clip
        def fail(*args, **kwargs):
            raise RuntimeError('公开模型故障')
        monkeypatch.setattr(neural_controller, 'deepfilter_denoise', fail)
        window.run_neural_comparison()
        wait_idle(app, window)
        assert '公开模型故障' in window.last_error
        assert window.neural_deepfilter_clip is previous and window.neural_player._clip is previous
        assert not window.busy and window.neural_compare_button.isEnabled()
    finally:
        window.close()


def test_df_job_is_discarded_after_source_change(app, tmp_path, monkeypatch):
    from voice_effect_lab.ui import neural_controller
    window = loaded_window(app, tmp_path)
    release = Event()
    try:
        def blocked(clip, noise_clip=None, strength=100):
            assert release.wait(10)
            return SimpleNamespace(clip=clip, elapsed_seconds=.1, clipped_samples=0)
        monkeypatch.setattr(neural_controller, 'deepfilter_denoise', blocked)
        window.run_neural_comparison()
        replacement = AudioClip(np.zeros(16000), 16000, '后载入原声')
        window._loaded((replacement, analyze(replacement)))
        release.set()
        wait_idle(app, window)
        assert window.neural_deepfilter_clip is None and window.neural_source_clip is None
        assert '丢弃' in window.status_label.text()
    finally:
        release.set()
        window.close()


def test_demo_reference_metrics_apply_only_to_exact_source(app, tmp_path, monkeypatch):
    window = loaded_window(app, tmp_path)
    try:
        fake_df(monkeypatch)
        window.load_neural_demo()
        wait_idle(app, window)
        compare(app, window)
        assert window.neural_metrics['source_snr_db'] is not None
        assert window.neural_metrics['deepfilter_snr_db'] is not None
        source = window.clip
        replacement = AudioClip(source.samples, source.sample_rate, '相同样本重新导入')
        window._loaded((replacement, analyze(replacement)))
        compare(app, window)
        assert window.neural_metrics['source_snr_db'] is None
        assert window.neural_metrics['deepfilter_snr_db'] is None
    finally:
        window.close()


def test_intelligent_volume_and_child_page_changes_stop_preview(app, tmp_path, monkeypatch):
    from test_player import DeviceOutput, audio_format
    window = loaded_window(app, tmp_path)
    try:
        fake_df(monkeypatch)
        window.tabs.setCurrentIndex(1)
        window.extension_tabs.setCurrentIndex(2)
        compare(app, window)
        output = DeviceOutput(audio_format(window.clip.sample_rate))
        window.neural_player._output_factory = lambda rate, parent: (output, output.fmt)
        window.play_neural()
        assert window.neural_player.state == 'playing'
        window.neural_volume_spin.setValue(42)
        assert output.volume == .42
        assert window.listen_volume_spin.value() == window.tts_volume_spin.value() == 42
        window.neural_mute_button.setChecked(True)
        assert window.mute_button.isChecked() and window.tts_mute_button.isChecked()
        assert output.volume == 0
        window.extension_tabs.setCurrentIndex(1)
        assert window.neural_player.state == 'stopped'
        window.tts_clip = window.clip
        window.tts_player.set_clip(window.clip)
        window.tts_player._output_factory = lambda rate, parent: (output, output.fmt)
        window.play_tts()
        assert window.tts_player.state == 'playing'
        window.extension_tabs.setCurrentIndex(0)
        assert window.tts_player.state == 'stopped'
    finally:
        window.close()


@pytest.mark.parametrize('paused', [False, True])
def test_leaving_basic_page_stops_its_playing_or_paused_preview(app, tmp_path, paused):
    from test_player import DeviceOutput, audio_format
    window = loaded_window(app, tmp_path)
    try:
        source = window.clip
        output = DeviceOutput(audio_format(source.sample_rate))
        window.player._output_factory = lambda rate, parent: (output, output.fmt)
        window._play()
        assert window.player.state == 'playing'
        if paused:
            window.player.pause()
            assert window.player.state == 'paused'
        window.tabs.setCurrentIndex(1)
        assert window.player.state == 'stopped'
        assert window.clip is source and window.player._clip is source
        window.extension_tabs.setCurrentIndex(2)
        assert window.player.state == 'stopped'
    finally:
        window.close()
