"""浅色工作台的新入口和真实信息反馈，避免只断言样式字符串。"""
import numpy as np
from scipy.io import wavfile

from test_ui import wait_idle
from voice_effect_lab.ui.main_window import MainWindow


def test_header_navigation_tracks_programmatic_return_to_workbench(app):
    window = MainWindow()
    try:
        assert window.main_navigation.checkedId() == 0
        window.main_navigation.button(1).click()
        assert window.tabs.currentIndex() == 1
        window.tabs.setCurrentIndex(0)
        assert window.main_navigation.checkedId() == 0
        window.tabs.setCurrentIndex(1)
        assert window.main_navigation.checkedId() == 1
    finally:
        window.close()


def test_device_drawer_preserves_sample_rate_and_output_selection(app):
    window = MainWindow()
    window.show()
    try:
        assert window.device_details_panel.isHidden()
        window.device_details_button.click()
        assert not window.device_details_panel.isHidden()
        assert window.sample_rate_combo.isVisible() and window.output_combo.isVisible()
        window.sample_rate_combo.setCurrentIndex(window.sample_rate_combo.findData(48000))
        output = window.output_combo.currentData()
        window.device_details_button.click()
        assert window.device_details_panel.isHidden()
        window.device_details_button.click()
        assert window.sample_rate_combo.currentData() == 48000
        assert window.output_combo.currentData() == output
    finally:
        window.close()


def test_metadata_badges_show_imported_values_and_survive_failed_import(app, tmp_path):
    path = tmp_path/'公开立体声.wav'
    wavfile.write(path, 16000, np.zeros((4000,2),dtype=np.int16))
    window = MainWindow()
    try:
        window.load_path(path)
        wait_idle(app,window)
        assert window.rate_badge.text() == '16 kHz'
        assert window.duration_badge.text() == '0.25 s'
        assert '2' in window.channel_badge.text() and '1' in window.channel_badge.text()
        assert '16000' in window.info_label.text()
        source = window.clip
        broken = tmp_path/'bad.wav'
        broken.write_bytes(b'invalid')
        window.load_path(broken)
        wait_idle(app,window)
        assert window.last_error and window.clip is source
        assert window.duration_badge.text() == '0.25 s'
        assert window.status_label.property('tone') == 'error'
        window._status('已恢复')
        assert window.status_label.property('tone') == 'normal'
    finally:
        window.close()


def test_manual_noise_details_show_without_hiding_collection_or_import(app):
    window = MainWindow()
    window.show()
    try:
        assert window.noise_details_panel.isHidden()
        assert window.noise_button.isVisible() and window.import_noise_button.isVisible()
        window.auto_noise_check.setChecked(False)
        assert not window.noise_details_panel.isHidden()
        assert window.confirm_noise_button.isVisible() and window.clear_noise_button.isVisible()
        window.auto_noise_check.setChecked(True)
        assert window.noise_details_panel.isHidden()
        assert window.noise_button.isVisible() and window.import_noise_button.isVisible()
    finally:
        window.close()


def test_extra_effect_controls_remain_available_in_compact_card(app,tmp_path):
    from test_processing_ui import prepare_window
    window,_=prepare_window(app,tmp_path)
    window.show()
    try:
        assert window.effects_details_panel.isHidden()
        window.effects_details_button.click()
        assert not window.effects_details_panel.isHidden()
        assert window.telephone_check.isVisible() and window.echo_delay_spin.isVisible()
        window.echo_delay_spin.setValue(.45)
        window.effects_details_button.click()
        window.effects_details_button.click()
        assert window.echo_delay_spin.value() == .45
    finally:
        window.close()


def test_first_capture_shows_live_duration_without_another_control_refresh(app):
    from PyQt5.QtTest import QTest
    from test_capture_ui import make_window
    window, device = make_window(app)
    window.show()
    try:
        window.start_recording()
        device.write(np.zeros(8000, dtype='<i2').tobytes())
        QTest.qWait(150)
        app.processEvents()
        assert '1.00 s' in window.capture_label.text()
        assert window.capture_label.isVisible()
        assert window.cancel_capture_button.isVisible()
    finally:
        window.cancel_recording()
        window.close()


def test_normalization_cannot_change_during_enhancement_and_result_is_adopted(app, tmp_path, monkeypatch):
    from test_processing_ui import prepare_window
    from test_enhancement_ui import block_enhance
    window, _ = prepare_window(app, tmp_path)
    release = block_enhance(monkeypatch)
    try:
        window.start_enhance()
        before = window.normalize_check.isChecked()
        assert not window.normalize_check.isEnabled()
        window.normalize_check.click()
        assert window.normalize_check.isChecked() == before
        release.set()
        wait_idle(app, window)
        assert window.enhanced_clip is not None
        assert window.normalize_check.isEnabled()
    finally:
        release.set()
        wait_idle(app, window)
        window.close()
