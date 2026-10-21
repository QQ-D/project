"""不依赖麦克风的底噪文件导入、确认和去噪流程。"""
import numpy as np
import pytest
from scipy.io import wavfile

from test_capture_ui import make_window
from test_processing_ui import prepare_window, process, block_processing
from test_ui import wait_idle
from voice_effect_lab.ui.main_window import MainWindow


def noise_file(tmp_path, rate=44100):
    path = tmp_path/'独立底噪文件.wav'
    values = np.random.default_rng(28).normal(0, .055, rate).astype(np.float32)
    wavfile.write(path, rate, values)
    return path


def test_imported_noise_requires_confirmation_and_preserves_original(app, tmp_path):
    window, _ = prepare_window(app, tmp_path)
    try:
        process(app, window)
        original = window.clip
        window.load_noise_path(noise_file(tmp_path))
        wait_idle(app, window)
        assert window.clip is original
        assert window.noise_clip.sample_rate == 44100
        assert not window.noise_confirmed
        assert window.denoised_clip is None
        assert window.confirm_noise_button.isEnabled()
        assert not window.denoise_button.isEnabled()
        assert '文件' in window.noise_status_label.text()
        assert '独立底噪文件.wav' in window.noise_status_label.text()
        window.confirm_noise_button.click()
        window.denoise_button.click()
        wait_idle(app, window)
        assert not window.last_error
        assert window.denoised_clip.sample_rate == 16000
        assert window.denoised_clip.duration == original.duration
        assert window.clip is original
        assert window.recorder.state == 'idle'
    finally:
        window.close()


def test_noise_can_be_imported_before_voice_without_input_device(app, tmp_path):
    window = MainWindow()
    window.denoise_method_combo.setCurrentIndex(window.denoise_method_combo.findData('wiener'))
    window.denoise_spin.setValue(60)
    try:
        window.load_noise_path(noise_file(tmp_path))
        wait_idle(app, window)
        reference = window.noise_clip
        assert window.clip is None
        assert window.confirm_noise_button.isEnabled()
        window.confirm_noise()
        window.load_path(tmp_path/'独立底噪文件.wav')
        wait_idle(app, window)
        assert window.noise_clip is reference
        assert not window.noise_confirmed
        assert not window.denoise_button.isEnabled()
        assert window.recorder.state == 'idle'
    finally:
        window.close()


@pytest.mark.parametrize('kind', ['garbage', 'short', 'truncated'])
def test_invalid_noise_import_preserves_valid_reference_and_result(app, tmp_path, kind):
    window, _ = prepare_window(app, tmp_path)
    try:
        process(app, window)
        original, noise, result = window.clip, window.noise_clip, window.denoised_clip
        path = tmp_path/'损坏底噪.wav'
        if kind == 'garbage':
            path.write_bytes(b'bad wav')
        elif kind == 'short':
            wavfile.write(path, 16000, np.zeros(1599, dtype=np.int16))
        else:
            path.write_bytes(noise_file(tmp_path).read_bytes()[:8000])
        window.load_noise_path(path)
        wait_idle(app, window)
        assert window.last_error
        assert window.clip is original
        assert window.noise_clip is noise
        assert window.noise_confirmed
        assert window.denoised_clip is result
        assert window.current_clip is result
        assert window.player._clip is result
    finally:
        window.close()


def test_noise_file_dialog_import_and_cancel(app, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    window, _ = prepare_window(app, tmp_path)
    path = noise_file(tmp_path)
    try:
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(path), 'WAV'))
        window.import_noise_button.click()
        wait_idle(app, window)
        reference = window.noise_clip
        assert reference.name == path.name
        window.confirm_noise()
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: ('', ''))
        window.import_noise_button.click()
        assert window.noise_clip is reference
        assert window.noise_confirmed
        assert not window.busy
    finally:
        window.close()


def test_noise_import_cannot_interrupt_background_processing(app, tmp_path, monkeypatch):
    window, _ = prepare_window(app, tmp_path)
    release = block_processing(monkeypatch)
    try:
        window.set_noise_segment(0, .35, confirmed=True)
        reference = window.noise_clip
        window.start_denoise()
        assert not window.import_noise_button.isEnabled()
        window.load_noise_path(noise_file(tmp_path))
        assert window.noise_clip is reference
        release.set()
        wait_idle(app, window)
        assert window.denoised_clip is not None
        assert window.import_noise_button.isEnabled()
    finally:
        release.set()
        wait_idle(app, window)
        window.close()


def test_file_reference_requires_reconfirmation_after_new_microphone_voice(app, tmp_path):
    window, device = make_window(app)
    try:
        path = noise_file(tmp_path, rate=8000)
        window.load_noise_path(path)
        wait_idle(app, window)
        reference = window.noise_clip
        window.confirm_noise()
        window.start_recording()
        assert not window.import_noise_button.isEnabled()
        window.load_noise_path(path)
        assert window.noise_clip is reference
        device.write(np.zeros(8000, dtype='<i2').tobytes())
        window.stop_recording()
        wait_idle(app, window)
        assert window.noise_clip is reference
        assert not window.noise_confirmed
        assert window.confirm_noise_button.isEnabled()
    finally:
        window.close()
