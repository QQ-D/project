"""实际 GUI + 实际文件，验证用户可见流程和任务收尾。"""
import time
import numpy as np
import pytest
from scipy.io import wavfile
from PyQt5.QtTest import QTest

from voice_effect_lab.ui.main_window import MainWindow


def wait_idle(app, window, timeout=10):
    end = time.monotonic()+timeout
    while window.busy and time.monotonic()<end:
        app.processEvents()
        QTest.qWait(10)
    app.processEvents()
    assert not window.busy, '后台任务没有结束'


def test_initial_controls_do_not_offer_unimplemented_actions(app):
    window = MainWindow()
    assert not window.play_button.isEnabled()
    assert not window.save_button.isEnabled()
    assert not window.record_button.isEnabled()
    assert not window.process_button.isEnabled()
    assert window.import_button.isEnabled()
    window.close()


def test_real_import_updates_metadata_plot_and_export(app, tmp_path):
    source = tmp_path/'测试.wav'
    t = np.arange(8000)/8000
    wavfile.write(source, 8000, (.25*np.sin(2*np.pi*440*t)).astype(np.float32))
    window = MainWindow()
    window.load_path(source)
    wait_idle(app, window)
    assert window.clip.sample_rate == 8000
    assert '8000' in window.info_label.text()
    assert '1.00' in window.info_label.text()
    assert window.play_button.isEnabled()
    assert window.save_button.isEnabled()
    assert len(window.plots.axes[0, 0].lines[0].get_xdata()) > 1
    assert len(window.plots.axes[2, 0].images) == 1
    dest = tmp_path/'导出.wav'
    window.save_to(dest)
    wait_idle(app, window)
    rate, samples = wavfile.read(dest)
    assert rate == 8000
    np.testing.assert_allclose(samples.astype(float)/32768, window.clip.samples, atol=1/32768)
    window.close()


@pytest.mark.parametrize('kind', ['garbage', 'truncated'])
def test_bad_import_keeps_previous_audio_and_reports_error(app, tmp_path, kind):
    good = tmp_path/'good.wav'
    bad = tmp_path/'bad.wav'
    wavfile.write(good, 8000, np.zeros(8000, dtype=np.int16))
    bad.write_bytes(b'bad file' if kind == 'garbage' else good.read_bytes()[:8044])
    window = MainWindow()
    window.load_path(good)
    wait_idle(app, window)
    original = window.clip
    window.load_path(bad)
    wait_idle(app, window)
    assert window.clip is original
    assert window.last_error
    assert window.play_button.isEnabled()
    window.close()


def test_import_cannot_reenter_and_closing_waits_for_worker(app, tmp_path):
    source = tmp_path/'one.wav'
    other = tmp_path/'two.wav'
    wavfile.write(source, 8000, np.zeros(8000, dtype=np.int16))
    wavfile.write(other, 16000, np.zeros(16000, dtype=np.int16))
    window = MainWindow()
    window.show()
    window.load_path(source)
    window.load_path(other)
    window.close()
    wait_idle(app, window)
    assert window.clip.name == 'one.wav'
    assert not window.isVisible()
    assert window.player.state == 'stopped'


def test_saving_error_is_visible_and_original_remains(app, tmp_path):
    source = tmp_path/'one.wav'
    wavfile.write(source, 8000, np.zeros(8, dtype=np.int16))
    window = MainWindow()
    window.load_path(source)
    wait_idle(app, window)
    window.save_to(tmp_path/'missing'/'out.wav')
    wait_idle(app, window)
    assert window.last_error
    assert window.clip.name == 'one.wav'
    assert window.save_button.isEnabled()
    window.close()


def test_canvas_renders_chinese_without_missing_glyphs(app):
    import warnings
    window = MainWindow()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        window.plots.draw()
    assert not [w for w in caught if 'Glyph' in str(w.message)]
    window.close()
