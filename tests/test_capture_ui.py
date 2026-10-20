"""真实录音器/GUI，麦克风由确定的 PCM 输入设备代替。"""
import numpy as np
from PyQt5.QtTest import QTest
from scipy.io import wavfile

from test_recorder import make_recorder
from test_ui import wait_idle
from voice_effect_lab.ui.main_window import MainWindow


def make_window(app):
    recorder, device=make_recorder(app)
    return MainWindow(recorder=recorder), device


def load_original(app,window,tmp_path):
    path=tmp_path/'original.wav'
    wavfile.write(path,8000,np.zeros(8000,dtype=np.int16))
    window.load_path(path);wait_idle(app,window)
    return window.clip


def test_voice_capture_enters_real_analysis_and_save_flow(app,tmp_path):
    window,device=make_window(app)
    window.start_recording()
    assert not window.import_button.isEnabled()
    assert not window.play_button.isEnabled()
    assert window.end_record_button.isEnabled()
    samples=np.tile(np.array([8192,-8192],dtype='<i2'),4000)
    device.write(samples.tobytes())
    window.stop_recording();wait_idle(app,window)
    assert window.clip.sample_rate==8000
    np.testing.assert_allclose(window.clip.samples[:2],[.25,-.25])
    assert window.play_button.isEnabled()
    assert window.save_button.isEnabled()
    assert window.analysis is not None
    path=tmp_path/'录音导出.wav'
    window.save_to(path);wait_idle(app,window)
    rate,exported=wavfile.read(path)
    assert rate==8000
    assert exported[:2].tolist()==[8192,-8192]
    window.close()


def test_noise_capture_auto_stops_without_overwriting_original(app,tmp_path):
    window,device=make_window(app)
    original=load_original(app,window,tmp_path)
    window.start_recording('noise')
    device.write(np.full(12000,100,dtype='<i2').tobytes())
    app.processEvents();QTest.qWait(1);app.processEvents()
    assert window.recorder.state=='idle'
    assert window.clip is original
    assert window.noise_clip.duration==1.5
    assert window.noise_confirmed
    assert not window.denoise_button.isEnabled()
    window.close()


def test_empty_recording_and_cancel_preserve_original(app,tmp_path):
    window,device=make_window(app)
    original=load_original(app,window,tmp_path)
    window.start_recording();window.stop_recording()
    assert window.last_error
    assert window.clip is original
    window.start_recording();device.write(b'\0'*200)
    window.cancel_recording()
    assert window.clip is original
    assert window.recorder.state=='idle'
    window.close()


def test_source_noise_selection_requires_confirmation_and_invalidates_on_import(app,tmp_path):
    window,_=make_window(app)
    original=load_original(app,window,tmp_path)
    assert not window.set_noise_segment(.1,.4,confirmed=False)
    assert window.noise_clip is None
    assert window.set_noise_segment(.1,.4,confirmed=True)
    assert window.noise_clip.duration==.3
    assert window.clip is original
    window.load_path(tmp_path/'original.wav');wait_idle(app,window)
    assert window.noise_clip is None
    window.close()


def test_recorded_noise_is_retained_but_requires_confirmation_for_import(app,tmp_path):
    window,device=make_window(app)
    window.start_recording('noise')
    device.write(np.full(12000,100,dtype='<i2').tobytes())
    app.processEvents();QTest.qWait(1);app.processEvents()
    old_noise=window.noise_clip
    load_original(app,window,tmp_path)
    assert window.noise_clip is old_noise
    assert not window.noise_confirmed
    assert window.confirm_noise_button.isEnabled()
    window.confirm_noise()
    assert window.noise_confirmed
    window.close()


def test_same_input_voice_recording_can_reuse_recorded_noise(app):
    window,device=make_window(app)
    window.start_recording('noise')
    device.write(b'\0'*24000)
    app.processEvents();QTest.qWait(1);app.processEvents()
    noise=window.noise_clip
    window.start_recording('voice')
    device.write(b'\0'*16000)
    window.stop_recording();wait_idle(app,window)
    assert window.noise_clip is noise
    assert window.noise_confirmed
    window.close()


def test_input_change_marks_noise_unconfirmed(app):
    window,device=make_window(app)
    window.start_recording('noise');device.write(b'\0'*24000)
    app.processEvents();QTest.qWait(1);app.processEvents()
    window.input_combo.addItem('另一个麦克风','other')
    window.input_combo.setCurrentIndex(window.input_combo.count()-1)
    assert not window.noise_confirmed
    assert '待确认' in window.noise_status_label.text()
    window.close()


def test_close_while_recording_discards_capture_and_releases_input(app):
    window,device=make_window(app)
    window.show();window.start_recording();device.write(b'\0'*200)
    window.close()
    assert window.recorder.state=='idle'
    assert not device.sink.isOpen()
    assert not window.isVisible()
    assert not window.busy


def test_import_and_second_capture_are_ignored_while_recording(app,tmp_path):
    window,device=make_window(app)
    original=load_original(app,window,tmp_path)
    window.start_recording('voice')
    window.load_path(tmp_path/'original.wav')
    window.start_recording('noise')
    assert window.recorder.state=='recording'
    assert window._recording_target=='voice'
    assert window.clip is original
    assert not window.noise_button.isEnabled()
    window.cancel_recording();window.close()


def test_qt_stop_failure_cannot_replace_original_or_noise(app,tmp_path):
    from PyQt5.QtMultimedia import QAudio
    window,device=make_window(app)
    original=load_original(app,window,tmp_path)
    window.set_noise_segment(.1,.4,confirmed=True)
    noise=window.noise_clip
    def bad_stop():
        device.error_code=QAudio.IOError
        device.stateChanged.emit(QAudio.StoppedState)
    device.stop=bad_stop
    window.start_recording('noise');device.write(b'\0'*8000)
    window.stop_recording();app.processEvents()
    assert window.clip is original
    assert window.noise_clip is noise
    assert window.last_error
    window.close()


def test_cancelled_noise_dialog_is_destroyed_after_modal_returns(app,tmp_path):
    from PyQt5.QtCore import QTimer,QCoreApplication,QEvent
    from voice_effect_lab.ui.noise_dialog import NoiseSelectionDialog
    window,_=make_window(app)
    load_original(app,window,tmp_path)
    for _ in range(3):
        QTimer.singleShot(0,lambda:window.findChildren(NoiseSelectionDialog)[-1].reject())
        window._choose_noise()
    QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
    assert not window.findChildren(NoiseSelectionDialog)
    window.close()
