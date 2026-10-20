"""设备边界使用可控 Qt 输出；播放器本身及 PCM 转换不替换。"""
import numpy as np
import pytest
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtMultimedia import QAudio, QAudioFormat

from voice_effect_lab.audio.data import AudioClip
from voice_effect_lab.audio.player import AudioPlayer, PlaybackError, pcm_bytes
from voice_effect_lab.audio import player as player_module


def audio_format(rate=8000, channels=1):
    fmt = QAudioFormat()
    fmt.setCodec('audio/pcm')
    fmt.setSampleRate(rate)
    fmt.setChannelCount(channels)
    fmt.setSampleSize(16)
    fmt.setSampleType(QAudioFormat.SignedInt)
    fmt.setByteOrder(QAudioFormat.LittleEndian)
    return fmt


class DeviceOutput(QObject):
    stateChanged = pyqtSignal(object)

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.device = None
        self.elapsed = 0

    def start(self, device):
        self.device = device
        self.elapsed = 0
        self.stateChanged.emit(QAudio.ActiveState)

    def suspend(self):
        self.stateChanged.emit(QAudio.SuspendedState)

    def resume(self):
        self.stateChanged.emit(QAudio.ActiveState)

    def stop(self):
        self.stateChanged.emit(QAudio.StoppedState)

    def processedUSecs(self):
        return self.elapsed

    def error(self):
        return QAudio.NoError

    def consume(self, frames):
        data = bytes(self.device.read(frames * self.fmt.channelCount() * 2))
        self.elapsed += int(len(data)/self.fmt.channelCount()/2/self.fmt.sampleRate()*1e6)
        return data


def make_player(app, rate=8000):
    output = DeviceOutput(audio_format(rate))
    player = AudioPlayer(output_factory=lambda rate, parent: (output, output.fmt))
    return player, output


def test_pause_resume_keeps_real_buffer_and_position(app):
    player, output = make_player(app)
    player.set_clip(AudioClip(np.full(8000, .25), 8000, 'one second'))
    player.play()
    first = output.consume(2000)
    assert np.frombuffer(first, '<i2').tolist() == [8192]*2000
    player.pause()
    assert player.state == 'paused'
    assert player.position_ms == 250
    player.play()
    output.consume(2000)
    assert player.state == 'playing'
    assert player.position_ms == 500
    player.close()


def test_stop_rewinds_and_replay_starts_at_beginning(app):
    player, output = make_player(app)
    player.set_clip(AudioClip(np.array([.25, .5, -.25, -.5]), 8000, 'four'))
    player.play()
    assert np.frombuffer(output.consume(2), '<i2').tolist() == [8192, 16384]
    player.stop()
    assert player.position_ms == 0
    assert player.state == 'stopped'
    player.play()
    assert np.frombuffer(output.consume(2), '<i2').tolist() == [8192, 16384]
    player.close()


def test_natural_finish_then_replay(app):
    player, output = make_player(app)
    player.set_clip(AudioClip(np.zeros(8000), 8000, 'one second'))
    player.play()
    output.consume(8000)
    output.stateChanged.emit(QAudio.IdleState)
    assert player.state == 'finished'
    assert player.position_ms == 1000
    player.play()
    assert player.position_ms == 0
    assert player.state == 'playing'
    player.close()


def test_new_clip_clears_paused_position(app):
    player, output = make_player(app)
    player.set_clip(AudioClip(np.zeros(8000), 8000, 'old'))
    player.play()
    output.consume(2000)
    player.pause()
    player.set_clip(AudioClip(np.ones(16000)*.1, 16000, 'new'))
    assert player.position_ms == 0
    assert player.duration_ms == 1000
    assert player.state == 'stopped'
    player.close()


def test_missing_device_does_not_claim_playing(app):
    def unavailable(rate, parent):
        raise PlaybackError('没有可用的音频输出设备')
    player = AudioPlayer(output_factory=unavailable)
    player.set_clip(AudioClip(np.zeros(8000), 8000, 'silent'))
    with pytest.raises(PlaybackError, match='输出设备'):
        player.play()
    assert player.state == 'stopped'
    assert player.position_ms == 0
    player.close()


def test_stereo_device_conversion_preserves_original():
    clip = AudioClip(np.array([-.5, .25, 0]), 8000, 'short')
    payload = pcm_bytes(clip, audio_format(channels=2))
    assert np.frombuffer(payload, '<i2').reshape(-1, 2).tolist() == [[-16384, -16384], [8192, 8192], [0, 0]]
    np.testing.assert_array_equal(clip.samples, [-.5, .25, 0])


def test_resampling_changes_output_length_not_original():
    clip = AudioClip(np.zeros(16000), 16000, 'one second')
    payload = pcm_bytes(clip, audio_format(rate=8000))
    assert len(payload) == 16000
    assert clip.sample_rate == 16000
    assert len(clip.samples) == 16000


def test_unsupported_device_sample_format_is_reported():
    fmt = audio_format()
    fmt.setSampleSize(24)
    with pytest.raises(PlaybackError):
        pcm_bytes(AudioClip(np.zeros(10), 8000, 'short'), fmt)


def test_very_short_audio_can_finish_during_output_start(app):
    class ImmediateOutput(DeviceOutput):
        def start(self, device):
            super().start(device)
            self.consume(1)
            self.stateChanged.emit(QAudio.IdleState)
    output = ImmediateOutput(audio_format())
    player = AudioPlayer(output_factory=lambda rate, parent: (output, output.fmt))
    player.set_clip(AudioClip(np.array([0]), 8000, 'one sample'))
    player.play()
    assert player.state == 'finished'
    player.close()


def test_rejected_format_before_signal_connection_remains_readable(app):
    fmt = audio_format()
    fmt.setSampleSize(24)
    output = DeviceOutput(fmt)
    player = AudioPlayer(output_factory=lambda rate, parent: (output, fmt))
    player.set_clip(AudioClip(np.zeros(10), 8000, 'short'))
    with pytest.raises(PlaybackError, match='24'):
        player.play()
    assert player.state == 'stopped'
    player.close()


@pytest.mark.parametrize('operation', ['pause', 'resume'])
def test_device_failure_during_pause_or_resume_cannot_claim_playing(app, operation):
    class FailingOutput(DeviceOutput):
        device_error = False
        def error(self):
            return QAudio.OpenError if self.device_error else QAudio.NoError
        def _fail(self):
            self.device_error = True
            self.stateChanged.emit(QAudio.StoppedState)
        def suspend(self):
            if operation == 'pause': self._fail()
            else: super().suspend()
        def resume(self):
            self._fail()
    output = FailingOutput(audio_format())
    player = AudioPlayer(output_factory=lambda rate, parent: (output, output.fmt))
    messages = []
    player.failed.connect(messages.append)
    player.set_clip(AudioClip(np.zeros(8000), 8000, 'one second'))
    player.play()
    player.pause()
    if operation == 'resume': player.play()
    assert player.state == 'stopped'
    assert player.position_ms == 0
    assert messages
    player.close()


def test_switching_output_device_stops_old_playback_without_losing_clip(app):
    player, output = make_player(app)
    clip = AudioClip(np.zeros(8000), 8000, 'one second')
    player.set_clip(clip)
    player.play()
    output.consume(2000)
    player.set_output_device('plughw:CARD=Generic,DEV=0')
    assert player.state == 'stopped'
    assert player.position_ms == 0
    assert player.duration_ms == 1000
    player.play()
    assert player.state == 'playing'
    assert output.consume(100)
    player.close()


def test_removed_selected_device_is_reported_without_silent_fallback(app):
    player = AudioPlayer()
    player.set_clip(AudioClip(np.zeros(10), 8000, 'short'))
    player.set_output_device('nonexistent-voice-lab-device')
    with pytest.raises(PlaybackError, match='选择的输出设备'):
        player.play()
    assert player.state == 'stopped'
    player.close()


def install_desktop_outputs(monkeypatch, failed_names=(), default_name='default'):
    """仅替换设备枚举与硬件 I/O，保留真实播放器、格式协商和 PCM 缓存。"""
    class Device:
        def __init__(self, name): self.name = name
        def deviceName(self): return self.name
        def isNull(self): return False
        def isFormatSupported(self, fmt): return True

    class Devices:
        @staticmethod
        def defaultOutputDevice(): return Device(default_name)
        @staticmethod
        def availableDevices(mode):
            return [Device(name) for name in [
                'default', 'plughw:CARD=HDMI,DEV=3', 'hdmi:CARD=HDMI,DEV=0',
                'plughw:CARD=Analog,DEV=0', 'sysdefault:CARD=Analog',
            ]]

    class DesktopOutput(DeviceOutput):
        def __init__(self, device, fmt, parent):
            super().__init__(fmt)
            self.name = device.deviceName()
            self.failed_start = False
        def start(self, buffer):
            if self.name in failed_names:
                self.failed_start = True
                self.stateChanged.emit(QAudio.StoppedState)
            else:
                super().start(buffer)
        def error(self):
            return QAudio.OpenError if self.failed_start else QAudio.NoError

    monkeypatch.setattr(player_module, 'QAudioDeviceInfo', Devices)
    monkeypatch.setattr(player_module, 'QAudioOutput', DesktopOutput)


def test_broken_system_default_falls_back_and_can_pause_resume_finish(app, monkeypatch):
    install_desktop_outputs(monkeypatch, failed_names={'default'})
    player = AudioPlayer()
    errors = []
    player.failed.connect(errors.append)
    clip = AudioClip(np.full(8000, .25), 8000, 'one second')
    player.set_clip(clip)
    try:
        player.play()
        assert player.state == 'playing'
        assert player.output_device_name == 'plughw:CARD=Analog,DEV=0'
        assert np.frombuffer(player._output.consume(2000), '<i2').tolist() == [8192]*2000
        player.pause()
        assert player.position_ms == 250
        player.play()
        player._output.consume(6000)
        player._output.stateChanged.emit(QAudio.IdleState)
        assert player.state == 'finished'
        assert player.position_ms == 1000
        assert not errors
        assert player._clip is clip
    finally:
        player.close()


def test_explicit_failed_output_does_not_switch_to_another_speaker(app, monkeypatch):
    install_desktop_outputs(monkeypatch, failed_names={'plughw:CARD=HDMI,DEV=3'})
    player = AudioPlayer()
    player.set_output_device('plughw:CARD=HDMI,DEV=3')
    player.set_clip(AudioClip(np.zeros(8000), 8000, 'one second'))
    try:
        with pytest.raises(PlaybackError, match='输出设备'):
            player.play()
        assert player.state == 'stopped'
        assert player.output_device_name == 'plughw:CARD=HDMI,DEV=3'
    finally:
        player.close()


def test_all_outputs_failed_has_one_actionable_error_and_no_live_buffer(app, monkeypatch):
    install_desktop_outputs(monkeypatch, failed_names={
        'default', 'plughw:CARD=Analog,DEV=0', 'sysdefault:CARD=Analog',
    })
    player = AudioPlayer()
    errors = []
    player.failed.connect(errors.append)
    player.set_clip(AudioClip(np.zeros(8000), 8000, 'one second'))
    try:
        with pytest.raises(PlaybackError, match='输出设备.*下拉'):
            player.play()
        assert player.state == 'stopped'
        assert player.position_ms == 0
        assert player.output_device_name is None
        assert player._output is None
        assert player._buffer is None
        assert not errors  # 同步启动失败只通过异常报告一次。
    finally:
        player.close()


def test_working_default_keeps_system_route(app, monkeypatch):
    install_desktop_outputs(monkeypatch)
    player = AudioPlayer()
    player.set_clip(AudioClip(np.zeros(8000), 8000, 'one second'))
    try:
        player.play()
        assert player._output.name == 'default'
        assert player.output_device_name is None
        assert player.state == 'playing'
    finally:
        player.close()
