"""使用真实 QIODevice 写入可控 PCM，只替代外部麦克风设备。"""
import numpy as np
import pytest
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtMultimedia import QAudio, QAudioFormat
from PyQt5.QtTest import QTest

from voice_effect_lab.audio.recorder import AudioRecorder, RecordingError, decode_pcm


def pcm_format(rate=8000, channels=1, bits=16, kind=QAudioFormat.SignedInt, endian=QAudioFormat.LittleEndian):
    f = QAudioFormat()
    f.setCodec('audio/pcm')
    f.setSampleRate(rate)
    f.setChannelCount(channels)
    f.setSampleSize(bits)
    f.setSampleType(kind)
    f.setByteOrder(endian)
    return f


class InputDevice(QObject):
    stateChanged = pyqtSignal(object)

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.sink = None
        self.error_code = QAudio.NoError

    def start(self, sink):
        self.sink = sink
        self.stateChanged.emit(QAudio.ActiveState)

    def write(self, payload):
        return self.sink.write(payload)

    def error(self):
        return self.error_code

    def stop(self):
        self.stateChanged.emit(QAudio.StoppedState)


def make_recorder(app, fmt=None):
    device = InputDevice(fmt or pcm_format())
    recorder = AudioRecorder(input_factory=lambda rate, parent, name: (device, device.fmt))
    return recorder, device


def test_decode_stereo_big_endian_uses_actual_format():
    fmt = pcm_format(48000, 2, endian=QAudioFormat.BigEndian)
    clip = decode_pcm(np.array([[16384, 0], [-16384, 16384]], dtype='>i2').tobytes(), fmt, 'test')
    np.testing.assert_allclose(clip.samples, [.25, 0])
    assert clip.sample_rate == 48000
    assert clip.source_channels == 2


@pytest.mark.parametrize('fmt,payload,want', [
    (pcm_format(bits=8, kind=QAudioFormat.UnSignedInt), bytes([0,128,255]), [-1,0,127/128]),
    (pcm_format(bits=32), np.array([-1073741824,536870912],dtype='<i4').tobytes(), [-.5,.25]),
    (pcm_format(bits=32, kind=QAudioFormat.Float), np.array([-.5,.25],dtype='<f4').tobytes(), [-.5,.25]),
])
def test_decode_supported_sample_types(fmt,payload,want):
    np.testing.assert_allclose(decode_pcm(payload,fmt,'test').samples,want)


def test_split_pcm_frames_are_reassembled_before_decode(app):
    recorder, device = make_recorder(app, pcm_format(48000,2))
    results=[]
    recorder.completed.connect(results.append)
    recorder.start()
    payload=np.array([[16384,0],[-16384,16384]],dtype='<i2').tobytes()
    device.write(payload[:3]); device.write(payload[3:])
    recorder.stop()
    assert recorder.state=='idle'
    np.testing.assert_allclose(results[0].samples,[.25,0])
    assert results[0].sample_rate==48000
    assert not device.sink.isOpen()


def test_cancel_discards_capture_without_completion(app):
    recorder,device=make_recorder(app)
    results=[]
    recorder.completed.connect(results.append)
    recorder.start()
    device.write(np.zeros(8000,dtype='<i2').tobytes())
    recorder.cancel()
    assert not results
    assert recorder.state=='idle'
    assert not device.sink.isOpen()


def test_empty_capture_reports_error_without_result(app):
    recorder,_=make_recorder(app)
    errors=[];results=[]
    recorder.failed.connect(errors.append);recorder.completed.connect(results.append)
    recorder.start();recorder.stop()
    assert errors and not results
    assert recorder.state=='idle'


def test_unsupported_format_startup_cleans_partial_initialization(app):
    recorder,_=make_recorder(app,pcm_format(bits=24))
    with pytest.raises(RecordingError): recorder.start()
    assert recorder.state=='idle'


def test_async_input_failure_does_not_return_partial_audio(app):
    recorder,device=make_recorder(app)
    errors=[];results=[]
    recorder.failed.connect(errors.append);recorder.completed.connect(results.append)
    recorder.start();device.write(b'\0'*100)
    device.error_code=QAudio.IOError
    device.stateChanged.emit(QAudio.StoppedState)
    assert recorder.state=='idle'
    assert errors and not results
    assert not device.sink.isOpen()


def test_capacity_stops_at_exact_sample_limit(app):
    recorder,device=make_recorder(app)
    results=[];recorder.completed.connect(results.append)
    recorder.start(limit_seconds=.001)
    device.write(np.array([1000]*20,dtype='<i2').tobytes())
    app.processEvents();QTest.qWait(1);app.processEvents()
    assert recorder.state=='idle'
    assert len(results)==1
    assert len(results[0].samples)==8


def test_queued_limit_from_cancelled_capture_cannot_stop_new_capture(app):
    recorder,device=make_recorder(app)
    recorder.start(limit_seconds=.001)
    old=device.sink
    device.write(b'\0'*40)
    recorder.cancel()
    recorder.start(limit_seconds=1)
    app.processEvents();QTest.qWait(1);app.processEvents()
    assert recorder.state=='recording'
    recorder.cancel()


def test_device_change_during_recording_is_rejected(app):
    recorder,_=make_recorder(app)
    recorder.start()
    with pytest.raises(RecordingError): recorder.set_input_device('another')
    recorder.cancel()


def test_missing_selected_input_has_readable_error(app):
    recorder=AudioRecorder()
    recorder.set_input_device('nonexistent-voice-lab-input')
    with pytest.raises(RecordingError,match='输入设备'): recorder.start()
    assert recorder.state=='idle'


def test_excessive_device_format_is_rejected_before_buffer_allocation(app,monkeypatch):
    import voice_effect_lab.audio.recorder as module
    recorder,_=make_recorder(app,pcm_format(192000,8,bits=32))
    def no_large_allocation(*args):
        raise AssertionError('不应分配巨型缓冲区')
    monkeypatch.setattr(module,'BoundedCapture',no_large_allocation)
    with pytest.raises(RecordingError,match='128 MiB'):
        recorder.start(limit_seconds=300)
    assert recorder.state=='idle'


def test_stop_device_exception_reports_error_and_closes_sink(app):
    class BrokenStop(InputDevice):
        def stop(self):
            raise RuntimeError('设备驱动关闭错误')
    device=BrokenStop(pcm_format())
    recorder=AudioRecorder(input_factory=lambda rate,parent,name:(device,device.fmt))
    errors=[];results=[]
    recorder.failed.connect(errors.append);recorder.completed.connect(results.append)
    recorder.start();device.write(b'\0'*200)
    recorder.stop()
    assert recorder.state=='idle'
    assert not device.sink.isOpen()
    assert errors and not results


def test_qt_error_reported_during_stop_cannot_emit_completed(app):
    class ErrorAtStop(InputDevice):
        def stop(self):
            self.error_code=QAudio.IOError
            self.stateChanged.emit(QAudio.StoppedState)
    device=ErrorAtStop(pcm_format())
    recorder=AudioRecorder(input_factory=lambda rate,parent,name:(device,device.fmt))
    results=[];errors=[]
    recorder.completed.connect(results.append);recorder.failed.connect(errors.append)
    recorder.start();device.write(b'\0'*200)
    recorder.stop()
    assert recorder.state=='idle'
    assert errors and not results
    assert not device.sink.isOpen()


def test_pending_input_error_is_kept_even_if_stop_resets_device_error(app):
    from threading import Thread
    class ErrorClearsOnStop(InputDevice):
        def stop(self):
            self.error_code=QAudio.NoError
            super().stop()
    device=ErrorClearsOnStop(pcm_format())
    recorder=AudioRecorder(input_factory=lambda rate,parent,name:(device,device.fmt))
    results=[];errors=[]
    recorder.completed.connect(results.append);recorder.failed.connect(errors.append)
    recorder.start();device.write(b'\0'*200)
    def interrupt():
        device.error_code=QAudio.IOError
        device.stateChanged.emit(QAudio.StoppedState)
    thread=Thread(target=interrupt);thread.start();thread.join()
    recorder.stop()
    app.processEvents()
    assert recorder.state=='idle'
    assert errors and not results
