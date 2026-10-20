"""真实 DSP / Qt 图形 / WAV 导出；仅在错误与并发测试控制后台边界。"""
from threading import Event

import numpy as np
from scipy.io import wavfile

from test_denoise import synthetic_case
from test_player import DeviceOutput, audio_format
from test_ui import wait_idle
from voice_effect_lab.ui.main_window import MainWindow


def prepare_window(app, tmp_path):
    _, source, _ = synthetic_case()
    path = tmp_path/'人工加噪.wav'
    wavfile.write(path, source.sample_rate, source.samples)
    window = MainWindow()
    window.load_path(path)
    wait_idle(app, window)
    return window, path


def process(app, window):
    assert window.set_noise_segment(0, .35, confirmed=True)
    window.denoise_button.click()
    wait_idle(app, window)
    assert not window.last_error, window.last_error
    assert window.denoised_clip is not None


def test_denoise_requires_original_and_confirmed_reference(app, tmp_path):
    window, _ = prepare_window(app, tmp_path)
    try:
        assert not window.denoise_button.isEnabled()
        window.start_denoise()
        assert window.last_error
        assert not window.busy
        assert window.set_noise_segment(0, .35, confirmed=True)
        assert window.denoise_button.isEnabled()
        assert window.process_button.isEnabled()
        window.noise_confirmed = False
        window.start_denoise()
        assert not window.busy
        assert window.denoised_clip is None
    finally:
        window.close()


def test_denoise_updates_actual_comparison_playback_and_selected_export(app, tmp_path):
    window, _ = prepare_window(app, tmp_path)
    original = window.clip
    try:
        process(app, window)
        result = window.denoised_clip
        assert window.clip is original
        assert window.stages.checkedId() == 1
        assert window.current_clip is result
        assert window.player._clip is result
        assert window.stages.button(1).isEnabled()
        assert not window.stages.button(2).isEnabled()
        assert not window.stages.button(3).isEnabled()
        np.testing.assert_array_equal(window.plots.axes[0, 1].lines[0].get_ydata(),
                                      window.denoised_analysis.wave_values)
        np.testing.assert_array_equal(window.plots.axes[2, 1].images[0].get_array(),
                                      window.denoised_analysis.spectrogram_db)
        assert '去噪' in window.file_label.text()
        assert '去噪' in window.save_button.text()
        assert 'SNR' not in window.denoise_status_label.text()
        dest = tmp_path/'去噪输出.wav'
        window.save_to(dest)
        wait_idle(app, window)
        rate, values = wavfile.read(dest)
        assert rate == original.sample_rate
        np.testing.assert_allclose(values/32768, result.samples, atol=1/32768)
        window.select_stage(0)
        assert window.current_clip is original
        assert window.player._clip is original
        np.testing.assert_array_equal(window.plots.axes[0, 1].lines[0].get_ydata(),
                                      window.analysis.wave_values)
        window.save_to(tmp_path/'原声输出.wav')
        wait_idle(app, window)
        _, raw = wavfile.read(tmp_path/'原声输出.wav')
        np.testing.assert_allclose(raw/32768, original.samples, atol=1/32768)
    finally:
        window.close()


def test_strength_change_stops_stale_audio_and_invalidates_result(app, tmp_path):
    window, _ = prepare_window(app, tmp_path)
    try:
        process(app, window)
        output = DeviceOutput(audio_format(rate=16000))
        window.player._output_factory = lambda rate, parent: (output, output.fmt)
        window.play_button.click()
        assert window.player.state == 'playing'
        window.denoise_spin.setValue(80)
        assert window.denoised_clip is None
        assert not window.stages.button(1).isEnabled()
        assert window.stages.checkedId() == 0
        assert window.player._clip is window.clip
        assert window.player.state == 'stopped'
        assert not window.plots.axes[2, 1].images
        window.denoise_button.click()
        wait_idle(app, window)
        assert window.denoised_clip is not None
        assert '80' in window.denoise_status_label.text()
    finally:
        window.close()


def test_new_reference_and_input_change_each_invalidate_processed_cache(app, tmp_path):
    window, _ = prepare_window(app, tmp_path)
    try:
        process(app, window)
        window.set_noise_segment(.05, .35, confirmed=True)
        assert window.denoised_clip is None
        window.start_denoise()
        wait_idle(app, window)
        assert window.denoised_clip is not None
        window.input_combo.addItem('另一个麦克风', 'other')
        window.input_combo.setCurrentIndex(window.input_combo.count()-1)
        assert not window.noise_confirmed
        assert window.denoised_clip is None
        assert not window.denoise_button.isEnabled()
        window.confirm_noise()
        assert window.denoise_button.isEnabled()
    finally:
        window.close()


def test_bad_import_preserves_result_but_successful_replacement_clears_it(app, tmp_path):
    window, path = prepare_window(app, tmp_path)
    try:
        process(app, window)
        result = window.denoised_clip
        bad = tmp_path/'bad.wav'
        bad.write_bytes(b'not wav')
        window.load_path(bad)
        wait_idle(app, window)
        assert window.denoised_clip is result
        assert window.current_clip is result
        window.load_path(path)
        wait_idle(app, window)
        assert window.denoised_clip is None
        assert window.noise_clip is None
        assert window.current_clip is window.clip
        assert not window.stages.button(1).isEnabled()
    finally:
        window.close()


def test_processing_failure_preserves_last_valid_result(app, tmp_path, monkeypatch):
    from voice_effect_lab.ui import processing_controller
    window, _ = prepare_window(app, tmp_path)
    try:
        process(app, window)
        result = window.denoised_clip
        def fail(*args):
            raise RuntimeError('模拟计算失败')
        monkeypatch.setattr(processing_controller, 'spectral_subtract', fail)
        window.start_denoise()
        wait_idle(app, window)
        assert '模拟计算失败' in window.last_error
        assert window.denoised_clip is result
        assert window.current_clip is result
        assert window.play_button.isEnabled()
    finally:
        window.close()


def block_processing(monkeypatch):
    from voice_effect_lab.ui import processing_controller
    release = Event()
    original = processing_controller.spectral_subtract
    def blocked(*args):
        assert release.wait(10), '测试未释放后台任务'
        return original(*args)
    monkeypatch.setattr(processing_controller, 'spectral_subtract', blocked)
    return release


def test_background_processing_blocks_conflicting_actions_and_discards_stale_snapshot(app, tmp_path, monkeypatch):
    window, path = prepare_window(app, tmp_path)
    release = block_processing(monkeypatch)
    try:
        window.set_noise_segment(0, .35, confirmed=True)
        window.start_denoise()
        assert window.busy
        for control in (window.import_button, window.record_button, window.noise_button,
                        window.denoise_button, window.denoise_spin, window.save_button,
                        window.process_button, window.stages.button(0)):
            assert not control.isEnabled()
        window.load_path(path)
        window.start_recording()
        assert window.recorder.state == 'idle'
        # disabled只阻止用户输入，程序化更改仍可能发生，必须检查结果快照。
        window.denoise_spin.setValue(90)
        release.set()
        wait_idle(app, window)
        assert window.denoised_clip is None
        assert window.current_clip is window.clip
        assert window.denoise_button.isEnabled()
    finally:
        release.set()
        wait_idle(app, window)
        window.close()


def test_closing_during_processing_waits_then_releases_window(app, tmp_path, monkeypatch):
    window, _ = prepare_window(app, tmp_path)
    release = block_processing(monkeypatch)
    try:
        window.show()
        window.set_noise_segment(0, .35, confirmed=True)
        window.start_denoise()
        window.close()
        assert window._close_pending
        assert window.isVisible()
        release.set()
        wait_idle(app, window)
        assert not window.isVisible()
        assert window.player.state == 'stopped'
    finally:
        release.set()
        wait_idle(app, window)
        window.close()


def test_reset_and_clear_noise_preserve_original_and_remove_processed_cache(app, tmp_path):
    window, _ = prepare_window(app, tmp_path)
    try:
        process(app, window)
        source, noise = window.clip, window.noise_clip
        window.denoise_spin.setValue(85)
        window.start_denoise()
        wait_idle(app, window)
        window.reset_button.click()
        assert window.denoise_spin.value() == 60
        assert window.denoised_clip is None
        assert window.clip is source
        assert window.noise_clip is noise
        window.start_denoise()
        wait_idle(app, window)
        window.clear_noise()
        assert window.denoised_clip is None
        assert window.noise_clip is None
        assert window.clip is source
    finally:
        window.close()
