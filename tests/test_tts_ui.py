"""Real Qt worker/controller tests; local synthesis never opens audio devices."""
import importlib
from threading import Event

import numpy as np
import pytest
from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QWidget, QComboBox, QTabWidget
from scipy.io import wavfile

from voice_effect_lab.audio.data import AudioClip
from voice_effect_lab.audio.player import AudioPlayer
from voice_effect_lab.ui.worker import Job
from test_ui import wait_idle
from test_player import DeviceOutput, audio_format


def make_host(app, monkeypatch, available=True):
    spec = importlib.util.find_spec('voice_effect_lab.ui.tts_controller')
    assert spec is not None, 'Offline TTS controller is not implemented'
    module = importlib.import_module('voice_effect_lab.ui.tts_controller')
    monkeypatch.setattr(module, 'tts_available', lambda: (available, '离线基础普通话' if available else '缺少 cmn 语音'))

    class Host(QWidget, module.TtsController):
        def __init__(self):
            super().__init__()
            self.busy = False
            self.last_error = ''
            self.message = ''
            self.clip = AudioClip(np.full(80, .2), 8000, '原声')
            self.player = AudioPlayer(self)
            self.pool = QThreadPool(self)
            self.tabs = QTabWidget(self)
            self.tabs.addTab(QWidget(), '工作台')
            self.tabs.addTab(QWidget(), '扩展')
            self.tabs.setCurrentIndex(1)
            self.output_combo = QComboBox()
            self.output_combo.addItem('测试设备', 'test-output')
            self._init_tts()
            self.panel = self.build_tts_panel()
            self._controls()

        @property
        def occupied(self):
            return self.busy

        def _controls(self):
            self._tts_controls()

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
            job = Job(compute)
            self._active_job = job
            job.signals.result.connect(apply)
            job.signals.failed.connect(self._error)
            job.signals.finished.connect(self._finished)
            self.pool.start(job)

        def _finished(self):
            self.busy = False
            self._active_job = None
            self._controls()

        def _loaded(self, pair):
            self.clip, self.analysis = pair
            self.player.set_clip(self.clip)

    return Host(), module


def generate(app, host):
    host.tts_generate_button.click()
    wait_idle(app, host)
    assert not host.last_error, host.last_error
    assert host.tts_clip is not None


def cleanup(app, host):
    wait_idle(app, host)
    host.tts_player.close()
    host.player.close()
    host.close()


def test_generate_save_send_keep_separate_source_until_explicit_action(app, tmp_path, monkeypatch):
    host, _ = make_host(app, monkeypatch)
    try:
        original = host.clip
        assert not host.tts_play_button.isEnabled()
        host.tts_text.setPlainText('这是独立中文合成测试。')
        generate(app, host)
        assert host.clip is original and host.player._clip is None
        assert host.tts_player._clip is host.tts_clip
        assert host.tts_analysis.wave_values.size
        path = tmp_path/'中文输出.wav'
        host.save_tts_to(path)
        wait_idle(app, host)
        rate, samples = wavfile.read(path)
        assert rate == host.tts_clip.sample_rate
        np.testing.assert_allclose(samples/32768, host.tts_clip.samples, atol=1/32768)
        host.tts_send_button.click()
        assert host.clip is host.tts_clip and host.player._clip is host.tts_clip
        assert host.tabs.currentIndex() == 0
    finally:
        cleanup(app, host)


def test_text_speed_changes_invalidate_and_busy_snapshot_discards_result(app, monkeypatch):
    host, module = make_host(app, monkeypatch)
    release = Event()
    try:
        host.tts_text.setPlainText('中文公开测试。')
        generate(app, host)
        host.tts_speed_spin.setValue(1.2)
        assert host.tts_clip is None and host.tts_analysis is None
        assert host.tts_player._clip is None
        assert host.tts_speed_slider.value() == 120
        real = module.synthesize_chinese
        def blocked(*args):
            assert release.wait(10)
            return real(*args)
        monkeypatch.setattr(module, 'synthesize_chinese', blocked)
        host.tts_generate_button.click()
        assert host.busy
        assert not host.tts_text.isEnabled() and not host.tts_speed_spin.isEnabled()
        assert not host.tts_save_button.isEnabled() and not host.tts_send_button.isEnabled()
        host.tts_text.setPlainText('程序化变化使旧合成失效。')
        release.set()
        wait_idle(app, host)
        assert host.tts_clip is None
        assert '丢弃' in host.message
    finally:
        release.set()
        cleanup(app, host)


def test_failed_regeneration_preserves_previous_valid_tts(app, monkeypatch):
    host, module = make_host(app, monkeypatch)
    try:
        host.tts_text.setPlainText('公开测试句。')
        generate(app, host)
        valid, source = host.tts_clip, host.clip
        def fail(*args):
            raise RuntimeError('公开引擎失败测试')
        monkeypatch.setattr(module, 'synthesize_chinese', fail)
        host.tts_generate_button.click()
        wait_idle(app, host)
        assert '引擎失败' in host.last_error
        assert host.tts_clip is valid and host.clip is source
        assert host.tts_play_button.isEnabled()
    finally:
        cleanup(app, host)


def test_preview_uses_separate_player_and_selected_output_stops_main(app, monkeypatch):
    host, _ = make_host(app, monkeypatch)
    try:
        host.tts_text.setPlainText('公开试听测试。')
        generate(app, host)
        fmt = audio_format(host.tts_clip.sample_rate)
        preview = DeviceOutput(fmt)
        main = DeviceOutput(fmt)
        host.tts_player._output_factory = lambda rate, parent: (preview, fmt)
        host.player._output_factory = lambda rate, parent: (main, fmt)
        host.player.set_clip(host.clip)
        host.player.play()
        host.tts_play_button.click()
        assert host.player.state == 'stopped'
        assert host.tts_player.state == 'playing'
        assert host.tts_player.output_device_name == 'test-output'
        assert preview.consume(30)
        assert host.tts_stop_button.isEnabled()
        host.tts_stop_button.click()
        assert host.tts_player.state == 'stopped'
    finally:
        cleanup(app, host)


def test_missing_engine_and_invalid_text_disable_dependent_actions(app, monkeypatch):
    host, _ = make_host(app, monkeypatch, available=False)
    try:
        host.tts_text.setPlainText('公开测试。')
        assert not host.tts_generate_button.isEnabled()
        assert not host.tts_play_button.isEnabled()
        assert 'cmn' in host.tts_engine_label.text()
    finally:
        cleanup(app, host)
    host, _ = make_host(app, monkeypatch)
    try:
        host.tts_text.setPlainText('中'*401)
        assert not host.tts_generate_button.isEnabled()
        host.start_tts()
        assert host.last_error and not host.busy
    finally:
        cleanup(app, host)
