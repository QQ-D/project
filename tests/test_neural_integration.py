"""智能比较在真实主窗口中的缓存、后台任务和三路播放器整合。"""
import numpy as np
import pytest
from scipy.io import wavfile

from test_player import DeviceOutput, audio_format
from test_processing_ui import prepare_window
from test_ui import wait_idle


@pytest.fixture(scope='module')
def model():
    from voice_effect_lab.audio.neural import train_model
    return train_model(epochs=2, batch_size=128).model


def compare(app, window, model):
    window.neural_model = model
    window._neural_controls()
    window.neural_compare_button.click()
    wait_idle(app, window, timeout=40)
    assert not window.last_error, window.last_error
    assert window.neural_clip is not None


def test_training_and_comparison_preserve_workbench_and_export_selected_cache(app, tmp_path):
    window, _ = prepare_window(app, tmp_path)
    try:
        assert hasattr(window, 'neural_train_button')
        assert not window.neural_compare_button.isEnabled()
        window.start_denoise()
        wait_idle(app, window)
        source, denoised = window.clip, window.denoised_clip
        window.neural_epochs_spin.setValue(2)
        window.neural_train_button.click()
        assert window.busy and not window.record_button.isEnabled()
        assert not window.neural_demo_button.isEnabled()
        wait_idle(app, window, timeout=40)
        assert not window.last_error and window.neural_model is not None
        window.neural_compare_button.click()
        wait_idle(app, window, timeout=40)
        assert not window.last_error
        result = window.neural_clip
        assert result.sample_rate == source.sample_rate and len(result.samples) == len(source.samples)
        assert window.clip is source and window.current_clip is denoised
        assert window.denoised_clip is denoised and window.player._clip is denoised
        window.neural_selection_combo.setCurrentIndex(window.neural_selection_combo.findData('bp'))
        path = tmp_path/'独立BP输出.wav'
        window.save_neural_to(path)
        wait_idle(app, window)
        rate, data = wavfile.read(path)
        assert rate == source.sample_rate
        np.testing.assert_allclose(data/32768, result.samples, atol=1/32768)
        window.neural_strength_spin.setValue(30)
        assert window.neural_clip is None and window.neural_traditional_clip is None
        assert window.denoised_clip is denoised and window.current_clip is denoised
    finally:
        window.close()


def test_three_players_are_exclusive_and_share_live_volume(app, tmp_path, model):
    window, _ = prepare_window(app, tmp_path)
    try:
        compare(app, window, model)
        window.neural_selection_combo.setCurrentIndex(window.neural_selection_combo.findData('bp'))
        fmt = audio_format(window.clip.sample_rate)
        outputs = []
        def output_factory(rate, parent):
            output = DeviceOutput(audio_format(rate))
            outputs.append(output)
            return output, output.fmt
        for player in (window.player, window.tts_player, window.neural_player):
            player._output_factory = output_factory
        window.tabs.setCurrentIndex(1)
        window.play_neural()
        assert window.neural_player.state == 'playing'
        window.listen_volume_spin.setValue(45)
        assert window.neural_player.volume == 45 and outputs[-1].volume == .45
        window.tts_mute_button.setChecked(True)
        assert window.neural_player.muted and outputs[-1].volume == 0
        window.tts_mute_button.setChecked(False)
        window.tts_text.setPlainText('这是公开编写的播放器互斥测试。')
        window.start_tts()
        assert window.neural_player.state == 'stopped'
        wait_idle(app, window)
        window.play_neural()
        window.play_tts()
        assert window.tts_player.state == 'playing' and window.neural_player.state == 'stopped'
        window.play_neural()
        assert window.tts_player.state == 'stopped' and window.neural_player.state == 'playing'
        window._play()
        assert window.neural_player.state == 'stopped' and window.player.state == 'playing'
        window.play_neural()
        window.tabs.setCurrentIndex(0)
        assert window.neural_player.state == 'stopped'
        window.tabs.setCurrentIndex(1)
        window.play_neural()
        window._choose_output()
        assert window.neural_player.state == 'stopped'
        window.play_neural()
    finally:
        window.close()
    assert window.neural_player.state == 'stopped'


def test_sending_tts_replaces_neural_source_and_clears_comparison_but_keeps_model(app, tmp_path, model):
    window, _ = prepare_window(app, tmp_path)
    try:
        compare(app, window, model)
        source, bp = window.clip, window.neural_clip
        window.tts_text.setPlainText('公开编写的新原声测试。')
        window.start_tts()
        wait_idle(app, window)
        assert window.clip is source and window.neural_clip is bp
        window.send_tts_to_workbench()
        assert window.clip is window.tts_clip
        assert window.neural_clip is None and window.neural_traditional_clip is None
        assert window.neural_source_clip is None and window.neural_model is model
        assert window.neural_compare_button.isEnabled()
    finally:
        window.close()
