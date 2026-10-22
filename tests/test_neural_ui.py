"""Real Qt controller/cache workflows, with deterministic stand-ins for training only."""
import importlib
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QWidget, QComboBox, QTabWidget
from scipy.io import wavfile

from voice_effect_lab.analysis import analyze
from voice_effect_lab.audio.data import AudioClip
from voice_effect_lab.audio.player import AudioPlayer
from voice_effect_lab.ui.worker import Job
from test_ui import wait_idle
from test_player import DeviceOutput, audio_format


def make_host(app, monkeypatch):
    spec = importlib.util.find_spec('voice_effect_lab.ui.neural_controller')
    assert spec is not None, 'BP controller is not implemented'
    module = importlib.import_module('voice_effect_lab.ui.neural_controller')

    class Host(QWidget, module.NeuralController):
        def __init__(self):
            super().__init__()
            self.busy = False
            self.last_error = self.message = ''
            self.clip = AudioClip(.2*np.sin(2*np.pi*330*np.arange(2400)/8000), 8000, '公开测试音')
            self.analysis = analyze(self.clip)
            self.player, self.tts_player = AudioPlayer(self), AudioPlayer(self)
            self.pool = QThreadPool(self)
            self.tabs = QTabWidget(self)
            self.tabs.addTab(QWidget(), '工作台')
            self.tabs.addTab(QWidget(), '扩展')
            self.tabs.setCurrentIndex(1)
            self.output_combo = QComboBox()
            self.output_combo.addItem('测试设备', 'test-output')
            self._init_neural()
            self.panel = self.build_neural_panel()
            self.neural_engine_combo.setCurrentIndex(self.neural_engine_combo.findData('bp'))
            self._controls()

        @property
        def occupied(self):
            return self.busy

        def _controls(self):
            self._neural_controls()

        def _status(self, message):
            self.message = message

        def _error(self, message):
            self.last_error = message

        def _job(self, compute, apply, message):
            if self.occupied:
                return
            self.busy = True
            self.last_error = ''
            self.player.stop()
            self.tts_player.stop()
            self._status(message)
            self._controls()
            self.job = Job(compute)
            self.job.signals.result.connect(apply)
            self.job.signals.failed.connect(self._error)
            self.job.signals.finished.connect(self._finished)
            self.pool.start(self.job)

        def _finished(self):
            self.busy = False
            self.job = None
            self._controls()

        def _loaded(self, pair):
            self.clip, self.analysis = pair
            self._neural_source_changed()
            self.player.set_clip(self.clip)

    return Host(), module


def cleanup(app, host):
    wait_idle(app, host)
    host._close_neural()
    host.tts_player.close()
    host.player.close()
    host.close()
    app.processEvents()


def fast_training(monkeypatch, module, model=None):
    model = model or SimpleNamespace(training_label='公开合成谐波')
    def train(clean_clip=None, *, epochs=20, batch_size=128, progress=None):
        if progress:
            for epoch in range(epochs):
                progress(f'第 {epoch+1} 轮 train=0.2 validation=0.3')
        return SimpleNamespace(model=model, train_losses=(.2,)*epochs,
                               validation_losses=(.3,)*epochs, elapsed_seconds=.125,
                               train_samples=100, validation_samples=25)
    monkeypatch.setattr(module, 'train_model', train)
    return model


def ready_comparison(app, host, module, monkeypatch):
    fast_training(monkeypatch, module)
    host.neural_epochs_spin.setValue(2)
    host.neural_train_button.click()
    wait_idle(app, host)
    assert not host.last_error, host.last_error
    def infer(clip, model, strength=75):
        return SimpleNamespace(clip=AudioClip(clip.samples*.5, clip.sample_rate, 'BP.wav'),
                               elapsed_seconds=.123, clipped_samples=0)
    monkeypatch.setattr(module, 'denoise_neural', infer)
    host.neural_compare_button.click()
    wait_idle(app, host)
    assert not host.last_error, host.last_error
    assert host.neural_clip is not None


def test_training_progress_is_bounded_and_model_does_not_replace_source(app, monkeypatch):
    host, module = make_host(app, monkeypatch)
    try:
        source = host.clip
        assert not host.neural_compare_button.isEnabled()
        assert host.neural_epochs_spin.value() == 20
        assert host.neural_batch_spin.value() == 128
        fast_training(monkeypatch, module)
        host.neural_epochs_spin.setValue(2)
        host.neural_train_button.click()
        assert host.busy and not host.neural_demo_button.isEnabled()
        wait_idle(app, host)
        assert host.neural_model.training_label == '公开合成谐波'
        assert host.clip is source
        assert '第 2 轮' in host.neural_log.toPlainText()
        assert '0.300' in host.neural_model_label.text()
        assert host.neural_compare_button.isEnabled()
        for i in range(350):
            host._neural_progress.message.emit(str(i))
        app.processEvents()
        assert host.neural_log.document().blockCount() <= 300
    finally:
        cleanup(app, host)


def test_comparison_selection_export_send_and_plots_use_same_cache(app, monkeypatch, tmp_path):
    host, module = make_host(app, monkeypatch)
    try:
        source = host.clip
        ready_comparison(app, host, module, monkeypatch)
        assert host.clip is source
        assert host.neural_source_clip is source
        assert host.neural_metrics['source_snr_db'] is None
        assert host.neural_metrics['bp_elapsed_seconds'] == .123
        assert '—' in host.neural_metrics_label.text()
        for key, expected in [('source', source), ('traditional', host.neural_traditional_clip), ('bp', host.neural_clip)]:
            host.neural_selection_combo.setCurrentIndex(host.neural_selection_combo.findData(key))
            assert host.neural_player._clip is expected
            dest = tmp_path/f'{key}.wav'
            host.save_neural_to(dest)
            wait_idle(app, host)
            rate, samples = wavfile.read(dest)
            assert rate == expected.sample_rate
            np.testing.assert_allclose(samples/32768, expected.samples, atol=1/32768)
        host.show_neural_plots()
        first = host.neural_plot_dialog
        host.show_neural_plots()
        assert host.neural_plot_dialog is first
        np.testing.assert_allclose(host.neural_plots.axes[0, 1].lines[0].get_ydata(), host.neural_analysis.wave_values)
        target = host.neural_clip
        host.neural_send_button.click()
        assert host.clip is target and host.tabs.currentIndex() == 0
        assert host.neural_clip is None and host.neural_model is not None
    finally:
        cleanup(app, host)


def test_changes_clear_comparison_but_failure_preserves_valid_state(app, monkeypatch):
    host, module = make_host(app, monkeypatch)
    try:
        ready_comparison(app, host, module, monkeypatch)
        model, result = host.neural_model, host.neural_clip
        def fail(*args, **kwargs):
            raise ValueError('公开失败测试')
        monkeypatch.setattr(module, 'train_model', fail)
        host.start_neural_training()
        wait_idle(app, host)
        assert host.neural_model is model and host.neural_clip is result
        monkeypatch.setattr(module, 'denoise_neural', fail)
        host.run_neural_comparison()
        wait_idle(app, host)
        assert host.neural_clip is result and host.last_error
        host.neural_strength_slider.setValue(30)
        assert host.neural_clip is None and host.neural_player._clip is None
        assert host.neural_model is model
        assert not host.neural_save_button.isEnabled()
    finally:
        cleanup(app, host)


def test_late_comparison_is_discarded_after_source_changes(app, monkeypatch):
    host, module = make_host(app, monkeypatch)
    release = Event()
    try:
        ready_comparison(app, host, module, monkeypatch)
        def blocked(clip, model, strength=75):
            assert release.wait(10)
            return SimpleNamespace(clip=clip, elapsed_seconds=.01, clipped_samples=0)
        monkeypatch.setattr(module, 'denoise_neural', blocked)
        host.run_neural_comparison()
        host._loaded((AudioClip(np.zeros(3000), 8000, '新原声'), analyze(AudioClip(np.zeros(3000), 8000, '新原声'))))
        release.set()
        wait_idle(app, host)
        assert host.neural_clip is None
        assert '丢弃' in host.message
    finally:
        release.set()
        cleanup(app, host)


def test_strength_spin_and_slider_invalidate_only_comparison(app, monkeypatch):
    host, module = make_host(app, monkeypatch)
    try:
        ready_comparison(app, host, module, monkeypatch)
        model = host.neural_model
        host.neural_strength_spin.setValue(25)
        assert host.neural_strength_slider.value() == 25
        assert host.neural_clip is None and host.neural_model is model
        host.neural_strength_slider.setValue(50)
        assert host.neural_strength_spin.value() == 50
    finally:
        cleanup(app, host)


def test_training_wav_is_independent_and_invalid_replacement_preserves_data(app, monkeypatch, tmp_path):
    host, module = make_host(app, monkeypatch)
    try:
        ready_comparison(app, host, module, monkeypatch)
        source, model, result = host.clip, host.neural_model, host.neural_clip
        good, short = tmp_path/'干净.wav', tmp_path/'短.wav'
        wavfile.write(good, 8000, np.zeros(16000, np.int16))
        wavfile.write(short, 8000, np.zeros(8000, np.int16))
        host.load_neural_training_path(good)
        wait_idle(app, host)
        training = host.neural_training_clip
        assert training.duration == 2
        assert host.clip is source and host.neural_model is model and host.neural_clip is result
        host.load_neural_training_path(short)
        wait_idle(app, host)
        assert host.neural_training_clip is training and host.last_error
        host.neural_training_synthetic_button.click()
        assert host.neural_training_clip is None and host.neural_model is model
    finally:
        cleanup(app, host)


def test_preview_stops_other_players_and_busy_actions_are_disabled(app, monkeypatch, tmp_path):
    host, module = make_host(app, monkeypatch)
    release = Event()
    try:
        ready_comparison(app, host, module, monkeypatch)
        fmt = audio_format(host.clip.sample_rate)
        outputs = [DeviceOutput(fmt) for _ in range(3)]
        for player, output in zip((host.player, host.tts_player, host.neural_player), outputs):
            player._output_factory = lambda rate, parent, output=output: (output, fmt)
        host.player.set_clip(host.clip)
        host.tts_player.set_clip(host.clip)
        host.player.play()
        host.tts_player.play()
        host.neural_play_button.click()
        assert host.player.state == host.tts_player.state == 'stopped'
        assert host.neural_player.state == 'playing'
        assert host.neural_player.output_device_name == 'test-output'
        assert outputs[2].consume(30)
        assert host.neural_stop_button.isEnabled()
        host.neural_stop_button.click()
        assert host.neural_player.state == 'stopped'
        def blocked(*args, **kwargs):
            assert release.wait(10)
            raise ValueError('训练失败')
        monkeypatch.setattr(module, 'train_model', blocked)
        host.start_neural_training()
        assert not host.neural_save_button.isEnabled()
        assert not host.neural_selection_combo.isEnabled()
        host.save_neural_to(tmp_path/'blocked.wav')
        host.play_neural()
        host.send_neural_to_workbench()
        assert not (tmp_path/'blocked.wav').exists()
        assert host.neural_player.state == 'stopped'
    finally:
        release.set()
        cleanup(app, host)


def test_demo_reference_only_applies_to_exact_demo_source(app, monkeypatch):
    host, module = make_host(app, monkeypatch)
    try:
        clean = AudioClip(np.full(2400, .2), 8000, '公开干净参考')
        noisy = AudioClip(np.full(2400, .3), 8000, '公开加噪演示')
        monkeypatch.setattr(module, 'make_demo_pair', lambda *, seed: (clean, noisy))
        host.load_neural_demo()
        wait_idle(app, host)
        assert host.clip is noisy
        ready_comparison(app, host, module, monkeypatch)
        assert host.neural_metrics['source_snr_db'] == pytest.approx(6.0206, abs=.0001)
        arbitrary = AudioClip(np.full(2400, .3), 8000, '任意录音')
        host._loaded((arbitrary, analyze(arbitrary)))
        host.run_neural_comparison()
        wait_idle(app, host)
        assert host.neural_metrics['source_snr_db'] is None
        assert host.neural_metrics['bp_snr_db'] is None
    finally:
        cleanup(app, host)


def test_stale_training_discards_model_and_close_releases_dialog(app, monkeypatch):
    host, module = make_host(app, monkeypatch)
    release = Event()
    try:
        ready_comparison(app, host, module, monkeypatch)
        old_model, old_result = host.neural_model, host.neural_clip
        train = module.train_model
        def blocked(*args, **kwargs):
            assert release.wait(10)
            return train(*args, **kwargs)
        monkeypatch.setattr(module, 'train_model', blocked)
        host.start_neural_training()
        host.neural_epochs_spin.setValue(3)
        release.set()
        wait_idle(app, host)
        assert host.neural_model is old_model and host.neural_clip is old_result
        assert '丢弃' in host.message
        host.show_neural_plots()
        assert host.neural_plot_dialog is not None
        host._close_neural()
        assert host.neural_player.state == 'stopped'
        assert host.neural_plot_dialog is None and host.neural_plots is None
    finally:
        release.set()
        cleanup(app, host)


def test_real_training_model_roundtrip_and_invalid_load_keep_existing_cache(app, monkeypatch, tmp_path):
    host, _ = make_host(app, monkeypatch)
    try:
        host.neural_epochs_spin.setValue(2)
        host.start_neural_training()
        wait_idle(app, host, timeout=30)
        assert not host.last_error, host.last_error
        assert host.neural_model is not None
        path = tmp_path/'公开BP.npz'
        host.save_neural_model_to(path)
        wait_idle(app, host)
        assert path.exists() and path.stat().st_size > 0
        host.run_neural_comparison()
        wait_idle(app, host, timeout=30)
        assert not host.last_error, host.last_error
        previous_model, previous_result = host.neural_model, host.neural_clip
        invalid = tmp_path/'bad.npz'
        invalid.write_bytes(b'bad archive')
        host.load_neural_model_path(invalid)
        wait_idle(app, host)
        assert host.last_error
        assert host.neural_model is previous_model and host.neural_clip is previous_result
        host.load_neural_model_path(path)
        wait_idle(app, host)
        assert not host.last_error, host.last_error
        assert host.neural_model is not previous_model
        assert host.neural_model.training_label == previous_model.training_label
        assert host.neural_clip is None and host.neural_model_save_button.isEnabled()
    finally:
        cleanup(app, host)


def test_closing_plot_releases_controller_before_deferred_delete_and_can_reopen(app, monkeypatch):
    import gc
    from PyQt5.QtCore import QCoreApplication, QEvent
    host, module = make_host(app, monkeypatch)
    try:
        ready_comparison(app, host, module, monkeypatch)
        host.show_neural_plots()
        previous = host.neural_plot_dialog
        previous.close()
        assert host.neural_plot_dialog is None and host.neural_plots is None
        host.show_neural_plots()
        current = host.neural_plot_dialog
        assert current is not previous
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        gc.collect()
        assert host.neural_plot_dialog is current and current.isVisible()
    finally:
        cleanup(app, host)
