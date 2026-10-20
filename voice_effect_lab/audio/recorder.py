"""有界内存录音；设备回调只复制 PCM，停止后才转换音频。"""
import math

import numpy as np
from PyQt5.QtCore import QElapsedTimer, QIODevice, QObject, QTimer, pyqtSignal
from PyQt5.QtMultimedia import QAudio, QAudioDeviceInfo, QAudioFormat, QAudioInput

from .data import AudioClip, AudioError


class RecordingError(RuntimeError):
    pass


def _dtype(fmt):
    if (fmt.codec() != 'audio/pcm' or fmt.sampleRate() <= 0
            or not 1 <= fmt.channelCount() <= 8):
        raise RecordingError('输入设备的 PCM 格式无效。')
    size, kind = fmt.sampleSize(), fmt.sampleType()
    endian = '<' if fmt.byteOrder() == QAudioFormat.LittleEndian else '>'
    if kind == QAudioFormat.SignedInt and size in (16, 32):
        return np.dtype(endian+'i'+str(size//8))
    if kind == QAudioFormat.UnSignedInt and size == 8:
        return np.dtype('u1')
    if kind == QAudioFormat.Float and size == 32:
        return np.dtype(endian+'f4')
    raise RecordingError(f'暂不支持输入设备的 {size} bit PCM 格式。')


def decode_pcm(payload, fmt, name='麦克风录音.wav'):
    dtype = _dtype(fmt)
    frame_bytes = dtype.itemsize * fmt.channelCount()
    complete_bytes = len(payload)//frame_bytes * frame_bytes
    if not complete_bytes:
        raise RecordingError('没有采集到完整音频帧，请检查输入设备后重试。')
    data = np.frombuffer(payload[:complete_bytes], dtype=dtype).astype(np.float64)
    if fmt.sampleType() == QAudioFormat.SignedInt:
        data /= 2**(fmt.sampleSize()-1)
    elif fmt.sampleType() == QAudioFormat.UnSignedInt:
        data = (data-128)/128
    if not np.isfinite(data).all() or np.max(np.abs(data)) > 1:
        raise RecordingError('输入设备返回了非有限值或超出满幅的音频。')
    samples = data.reshape(-1, fmt.channelCount()).mean(axis=1)
    try:
        return AudioClip(samples, fmt.sampleRate(), name, fmt.channelCount())
    except AudioError as exc:
        raise RecordingError(str(exc)) from exc


class BoundedCapture(QIODevice):
    """达到上限仍确认接收数据，但不继续增长内存，并请求控制器停止。"""
    limit_reached = pyqtSignal(object)

    def __init__(self, capacity, parent=None):
        super().__init__(parent)
        self._data = bytearray(capacity)
        self.used = 0
        self._notified = False
        self.open(QIODevice.WriteOnly | QIODevice.Unbuffered)

    def writeData(self, data):
        count = min(len(data), len(self._data)-self.used)
        if count:
            self._data[self.used:self.used+count] = data[:count]
            self.used += count
        if self.used == len(self._data) and not self._notified:
            self._notified = True
            self.limit_reached.emit(self)
        return len(data)

    def readData(self, size):
        return b''

    def payload(self):
        return bytes(memoryview(self._data)[:self.used])


def _default_input(rate, parent, device_name):
    if device_name:
        device = next((d for d in QAudioDeviceInfo.availableDevices(QAudio.AudioInput)
                       if d.deviceName() == device_name), None)
        if device is None:
            raise RecordingError('选择的输入设备已不可用，请重新选择。')
    else:
        device = QAudioDeviceInfo.defaultInputDevice()
    if device.isNull():
        raise RecordingError('没有可用的音频输入设备，请检查麦克风和音频服务。')
    fmt = QAudioFormat()
    fmt.setCodec('audio/pcm')
    fmt.setSampleRate(rate)
    fmt.setChannelCount(1)
    fmt.setSampleSize(16)
    fmt.setSampleType(QAudioFormat.SignedInt)
    fmt.setByteOrder(QAudioFormat.LittleEndian)
    if not device.isFormatSupported(fmt):
        fmt = device.nearestFormat(fmt)
    _dtype(fmt)
    return QAudioInput(device, fmt, parent), fmt


class AudioRecorder(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    state_changed = pyqtSignal(str)
    duration_changed = pyqtSignal(int)

    def __init__(self, parent=None, input_factory=None):
        super().__init__(parent)
        self._factory = input_factory or _default_input
        self.input_device_name = None
        self._state = 'idle'
        self._device = self._sink = self._format = None
        self._connected = False
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.timeout.connect(self.stop)
        self._stopping = False

    @property
    def state(self):
        return self._state

    @property
    def duration_ms(self):
        if self._sink is None:
            return 0
        frame_bytes = self._format.channelCount()*self._format.sampleSize()//8
        return round(self._sink.used/frame_bytes/self._format.sampleRate()*1000)

    def _set_state(self, state):
        self._state = state
        self.state_changed.emit(state)

    def set_input_device(self, name):
        if self._state != 'idle':
            raise RecordingError('录音期间不能更换输入设备。')
        self.input_device_name = name or None

    def start(self, sample_rate=44100, limit_seconds=300, name='麦克风录音.wav'):
        if self._state != 'idle':
            raise RecordingError('正在采集音频，请先结束当前录音。')
        if (not isinstance(sample_rate, int) or sample_rate <= 0
                or not math.isfinite(limit_seconds) or not 0 < limit_seconds <= 300):
            raise RecordingError('采样率或录音时长不合法，最长支持 300 秒。')
        self._limit_seconds, self._name = limit_seconds, name
        try:
            self._device, self._format = self._factory(sample_rate, self, self.input_device_name)
            dtype = _dtype(self._format)
            capacity = max(1, round(limit_seconds*self._format.sampleRate())) * self._format.channelCount()*dtype.itemsize
            if capacity > 128*1024*1024:
                raise RecordingError('输入格式需要超过 128 MiB 的缓冲区，请降低采样率或录音时长。')
            self._sink = BoundedCapture(capacity, self)
            self._sink.limit_reached.connect(self._at_limit)
            self._device.stateChanged.connect(self._input_state)
            self._connected = True
            self._clock.start()
            self._set_state('recording')
            device = self._device
            device.start(self._sink)
            if device.error() != QAudio.NoError or self._device is not device:
                raise RecordingError('输入设备启动失败，请检查设备是否被占用或改选麦克风。')
            self._timer.start()
            self.duration_changed.emit(0)
        except Exception as exc:
            self.cancel()
            if isinstance(exc, RecordingError):
                raise
            raise RecordingError(f'无法开始录音：{exc}') from exc

    def _at_limit(self, sink):
        if sink is self._sink and self._state == 'recording' and not self._stopping:
            # 离开音频 writeData 回调后再停止；旧会话排队信号不会停止新会话。
            self._stop_timer.start(0)

    def _detach(self):
        self._timer.stop()
        self._stop_timer.stop()
        device, sink = self._device, self._sink
        self._device = self._sink = None
        close_error = None
        try:
            if device is not None:
                # Qt 的错误通常通过 error()/信号报告，而不是 Python 异常。
                # stop() 可能清除旧错误，因此必须在停止前先保留错误状态。
                if device.error() != QAudio.NoError:
                    close_error = '录音设备报告错误，本次采集已丢弃，原有数据保持不变。'
                if self._connected:
                    device.stateChanged.disconnect(self._input_state)
                device.stop()
                if device.error() != QAudio.NoError:
                    close_error = '停止录音时设备报告错误，本次采集已丢弃，原有数据保持不变。'
        except Exception as exc:
            close_error = f'录音设备关闭失败，本次采集已丢弃：{exc}'
        finally:
            self._connected = False
            if device is not None and device.parent() is self:
                device.deleteLater()
            if sink is not None:
                sink.close()
                sink.deleteLater()
        return sink, close_error

    def stop(self):
        if self._state != 'recording' or self._stopping:
            return
        self._stopping = True
        try:
            sink, close_error = self._detach()
            # 提取 payload 时设备已经停止，不与音频写入并发。
            payload = sink.payload() if sink is not None else b''
            fmt = self._format
        finally:
            self._stopping = False
            self._set_state('idle')
        if close_error:
            self.failed.emit(close_error)
            return
        try:
            clip = decode_pcm(payload, fmt, self._name)
        except RecordingError as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(clip)

    def cancel(self):
        self._stopping = True
        try:
            _, close_error = self._detach()
        finally:
            self._stopping = False
            self._set_state('idle')
        if close_error:
            self.failed.emit(close_error)
        return not close_error

    def _input_state(self, state):
        if self.sender() is not self._device or self._stopping:
            return
        if state == QAudio.StoppedState and self._device.error() != QAudio.NoError:
            self.cancel()
            self.failed.emit('录音设备中断，本次采集已丢弃，原有音频保持不变。')

    def _tick(self):
        if self._state != 'recording':
            return
        self.duration_changed.emit(self.duration_ms)
        if self._clock.elapsed() >= self._limit_seconds*1000:
            self.stop()
        elif self._clock.elapsed() >= 3000 and self._sink.used == 0:
            self.cancel()
            self.failed.emit('3 秒内未收到麦克风数据，请检查输入设备。')

    def close(self):
        self.cancel()
