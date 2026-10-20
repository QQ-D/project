"""使用已有 QtMultimedia 播放内存缓存，不依赖 PortAudio。"""
from math import gcd

import numpy as np
from scipy.signal import resample_poly
from PyQt5.QtCore import QByteArray, QBuffer, QIODevice, QObject, QTimer, pyqtSignal
from PyQt5.QtMultimedia import QAudio, QAudioDeviceInfo, QAudioFormat, QAudioOutput


class PlaybackError(RuntimeError):
    pass


def pcm_bytes(clip, fmt):
    """转换为设备接受的 PCM 格式；不修改原声与导出数据。"""
    rate, channels = fmt.sampleRate(), fmt.channelCount()
    if rate <= 0 or channels <= 0 or channels > 8 or fmt.codec() != 'audio/pcm':
        raise PlaybackError('音频输出设备的 PCM 格式无效。')
    samples = clip.samples
    if rate != clip.sample_rate:
        divisor = gcd(rate, clip.sample_rate)
        samples = resample_poly(samples, rate//divisor, clip.sample_rate//divisor)
    samples = np.clip(samples.astype(np.float64), -1, 1)
    size, kind = fmt.sampleSize(), fmt.sampleType()
    endian = '<' if fmt.byteOrder() == QAudioFormat.LittleEndian else '>'
    if kind == QAudioFormat.SignedInt and size in (16, 32):
        factor = 2**(size-1)
        data = np.clip(np.rint(samples*factor), -factor, factor-1).astype(endian+'i'+str(size//8))
    elif kind == QAudioFormat.UnSignedInt and size == 8:
        data = np.clip(np.rint(samples*128+128), 0, 255).astype('u1')
    elif kind == QAudioFormat.Float and size == 32:
        data = samples.astype(endian+'f4')
    else:
        raise PlaybackError(f'不支持输出设备的 {size} bit 音频格式。')
    if channels > 1:
        data = np.repeat(data[:, None], channels, axis=1)
    return data.tobytes()


def _default_output(rate, parent, device_name=None):
    if device_name:
        device = next((d for d in QAudioDeviceInfo.availableDevices(QAudio.AudioOutput)
                       if d.deviceName() == device_name), None)
        if device is None:
            raise PlaybackError('选择的输出设备已不可用，请重新选择设备。')
    else:
        device = QAudioDeviceInfo.defaultOutputDevice()
    if device.isNull():
        raise PlaybackError('没有可用的音频输出设备；请在桌面会话中检查扬声器与音频服务。')
    fmt = QAudioFormat()
    fmt.setCodec('audio/pcm')
    fmt.setSampleRate(rate)
    fmt.setChannelCount(1)
    fmt.setSampleSize(16)
    fmt.setSampleType(QAudioFormat.SignedInt)
    fmt.setByteOrder(QAudioFormat.LittleEndian)
    if not device.isFormatSupported(fmt):
        fmt = device.nearestFormat(fmt)
    return QAudioOutput(device, fmt, parent), fmt


def _fallback_output_names():
    """Qt ALSA 默认路由失效时，仅尝试有独立系统路由的声卡。

    不遍历 HDMI/环绕输出：它们经常未连接，且可能把声音送到别的屏幕。
    非 ALSA 后端没有这些名字，仍使用其系统默认路由与手动设备选择。
    """
    names = [d.deviceName() for d in QAudioDeviceInfo.availableDevices(QAudio.AudioOutput)]
    candidates = []
    default = QAudioDeviceInfo.defaultOutputDevice().deviceName()
    for name in names:
        if not name.startswith('sysdefault:CARD='):
            continue
        card = name.split('=', 1)[1]
        for candidate in (f'plughw:CARD={card},DEV=0', name):
            if candidate in names and candidate != default and candidate not in candidates:
                candidates.append(candidate)
    return candidates


class AudioPlayer(QObject):
    state_changed = pyqtSignal(str)
    position_changed = pyqtSignal(int)
    failed = pyqtSignal(str)
    output_device_changed = pyqtSignal(str)

    def __init__(self, parent=None, output_factory=None):
        super().__init__(parent)
        self.output_device_name = None
        self._output_factory = output_factory
        self._clip = None
        self._output = None
        self._output_connected = False
        self._buffer = None
        self._state = 'stopped'
        self._position = 0
        self._releasing = False
        self._starting = False
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)

    @property
    def state(self):
        return self._state

    @property
    def duration_ms(self):
        return round(self._clip.duration*1000) if self._clip else 0

    @property
    def position_ms(self):
        if self._output is not None and self._state in ('playing', 'paused'):
            return min(self.duration_ms, max(0, self._output.processedUSecs()//1000))
        return self._position

    def _set_state(self, state):
        self._state = state
        self.state_changed.emit(state)

    def set_clip(self, clip):
        self.stop()
        self._clip = clip

    def set_output_device(self, name):
        self.stop()
        self.output_device_name = name or None

    def play(self):
        if self._clip is None:
            raise PlaybackError('请先导入音频。')
        if self._state == 'playing':
            return
        if self._state == 'paused' and self._output is not None:
            output = self._output
            output.resume()
            if self._output is not output:
                return  # 同步错误信号已经释放设备并通知界面。
            if output.error() != QAudio.NoError:
                self.stop()
                raise PlaybackError('恢复播放失败，请检查音频输出设备。')
            self._set_state('playing')
            self._timer.start()
            return
        self.stop()
        candidates = [self.output_device_name]
        errors = []
        # 显式选择的设备不擅自切换；系统默认失败时才发现备用路由。
        automatic = self.output_device_name is None and self._output_factory is None
        for name in candidates:
            try:
                self._starting = True
                if self._output_factory is not None:
                    self._output, fmt = self._output_factory(self._clip.sample_rate, self)
                else:
                    self._output, fmt = _default_output(self._clip.sample_rate, self, name)
                payload = pcm_bytes(self._clip, fmt)
                self._buffer = QBuffer(self)
                self._buffer.setData(QByteArray(payload))
                self._buffer.open(QIODevice.ReadOnly)
                self._output.stateChanged.connect(self._output_state)
                self._output_connected = True
                self._set_state('playing')
                # 极短音频可能在 start() 内完成并触发释放，保留局部引用。
                output = self._output
                output.start(self._buffer)
                if output.error() != QAudio.NoError:
                    raise PlaybackError(f'输出设备启动失败（Qt 错误码 {int(output.error())}）。')
                if self._state == 'playing':
                    self._timer.start()
                if name != self.output_device_name:
                    self.output_device_name = name
                    self.output_device_changed.emit(name)
                return
            except Exception as exc:
                self.stop()
                errors.append(f'{name or "系统默认输出"}: {exc}')
                if not automatic:
                    if isinstance(exc, PlaybackError):
                        raise
                    raise PlaybackError(f'播放失败：{exc}') from exc
                if name is None:
                    candidates.extend(_fallback_output_names())
            finally:
                self._starting = False
        raise PlaybackError('输出设备无法打开，请在“输出设备”下拉框选择已连接的扬声器或耳机。'
                            + '；'.join(errors))

    def pause(self):
        if self._state == 'playing' and self._output is not None:
            output = self._output
            output.suspend()
            if self._output is not output:
                return
            if output.error() != QAudio.NoError:
                self.stop()
                self.failed.emit('暂停时音频输出中断，请检查设备。')
                return
            self._timer.stop()
            self._set_state('paused')

    def _release(self):
        self._releasing = True
        self._timer.stop()
        output, buffer = self._output, self._buffer
        self._output = self._buffer = None
        try:
            if output is not None:
                if self._output_connected:
                    output.stateChanged.disconnect(self._output_state)
                self._output_connected = False
                output.stop()
                # 原生输出对象属于本播放器；测试替身由测试持有。
                if output.parent() is self:
                    output.deleteLater()
            if buffer is not None:
                buffer.close()
                buffer.deleteLater()
        finally:
            self._releasing = False

    def stop(self):
        self._release()
        self._position = 0
        self._set_state('stopped')
        self.position_changed.emit(0)

    def _output_state(self, state):
        if self._releasing:
            return
        if state == QAudio.IdleState and self._state == 'playing':
            self._position = self.duration_ms
            self._release()
            self._set_state('finished')
            self.position_changed.emit(self._position)
        elif state == QAudio.StoppedState and self._output is not None and self._output.error() != QAudio.NoError:
            if self._starting:
                return  # start() 返回后统一处理，可回退且不重复发送错误。
            self.stop()
            self.failed.emit('音频输出中断，请检查设备连接或音频服务。')

    def _tick(self):
        self.position_changed.emit(self.position_ms)

    def close(self):
        self.stop()
        self._clip = None
