"""扩展页多重回声的共享参数、真实输出与后台快照。"""
from threading import Event
import numpy as np
from scipy.io import wavfile
from test_processing_ui import prepare_window
from test_ui import wait_idle


def test_extension_apply_has_multiple_tail_and_matching_player_plot_export(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        raw=window.clip
        window.tabs.setCurrentIndex(1)
        assert not window.multiecho_check.isChecked()
        window.multiecho_delay_spin.setValue(.2)
        assert window.echo_delay_spin.value()==.2
        window.echo_strength_spin.setValue(.4)
        assert window.multiecho_strength_spin.value()==.4
        window.multiecho_count_spin.setValue(4)
        window.multiecho_button.click();wait_idle(app,window)
        assert not window.last_error
        assert window.tabs.currentIndex()==0
        assert window.multiecho_check.isChecked() and window.effects_check.isChecked()
        assert window.current_clip is window.effected_clip and window.player._clip is window.effected_clip
        assert len(window.current_clip.samples)==len(raw.samples)+4*round(.2*raw.sample_rate)
        np.testing.assert_array_equal(window.plots.axes[0,1].lines[0].get_ydata(),window.effected_analysis.wave_values)
        assert '4次' in window.effects_status_label.text()
        path=tmp_path/'多重回声.wav';result=window.current_clip
        window.save_to(path);wait_idle(app,window)
        rate,values=wavfile.read(path)
        assert rate==raw.sample_rate
        np.testing.assert_allclose(values/32768,result.samples,atol=1/32768)
        assert window.clip is raw
    finally:window.close()


def test_echo_count_change_only_clears_effect_and_repeat_does_not_accumulate(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        window.use_denoise_check.setChecked(True)
        window.enhance_check.setChecked(True)
        window.multiecho_button.click();wait_idle(app,window)
        first=window.effected_clip;noise,enhanced=window.denoised_clip,window.enhanced_clip
        assert window.echo_strength_spin.value()==.45
        window.multiecho_button.click();wait_idle(app,window)
        np.testing.assert_array_equal(window.effected_clip.samples,first.samples)
        window.multiecho_count_spin.setValue(8)
        assert window.effected_clip is None
        assert window.enhanced_clip is enhanced and window.denoised_clip is noise
        assert window.current_clip is enhanced and window.preset_name=='自定义'
    finally:window.close()


def test_single_echo_bypass_and_preset_reset_multiecho_parameters(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        window.multiecho_button.click();wait_idle(app,window)
        window.multiecho_check.setChecked(False)
        assert window._effect_settings().echo_count==1
        window.start_effects();wait_idle(app,window)
        assert len(window.current_clip.samples)==len(window.clip.samples)+round(.3*window.clip.sample_rate)
        window.multiecho_check.setChecked(True)
        window.multiecho_count_spin.setValue(6)
        window.preset_buttons['自定义'].click()
        assert window.multiecho_check.isChecked() and window.multiecho_count_spin.value()==6
        window.preset_buttons['山谷回声'].click()
        assert not window.multiecho_check.isChecked()
        assert window.multiecho_count_spin.value()==3
        assert window._effect_settings().echo_count==1
    finally:window.close()


def test_multiecho_reset_synchronizes_both_pages(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        source=window.clip
        window.multiecho_check.setChecked(True)
        window.multiecho_count_spin.setValue(8)
        window.multiecho_delay_spin.setValue(.8)
        window.multiecho_strength_spin.setValue(.7)
        window.reset_processing()
        assert window.clip is source
        assert not window.multiecho_check.isChecked() and window._effect_settings().echo_count==1
        assert window.multiecho_count_spin.value()==3 and window.multiecho_count_slider.value()==3
        assert window.multiecho_delay_spin.value()==.3 and window.multiecho_delay_slider.value()==30
        assert window.multiecho_strength_spin.value()==0 and window.multiecho_strength_slider.value()==0
    finally:window.close()


def test_pending_multiecho_count_change_discards_result_and_controls_are_locked(app,tmp_path,monkeypatch):
    from voice_effect_lab.ui.effects_controller import apply_effects
    entered,release=Event(),Event()
    def blocked(*args):
        entered.set();assert release.wait(10);return apply_effects(*args)
    monkeypatch.setattr('voice_effect_lab.ui.effects_controller.apply_effects',blocked)
    window,_=prepare_window(app,tmp_path)
    try:
        window.multiecho_button.click();assert entered.wait(5)
        assert not window.multiecho_count_spin.isEnabled()
        assert not window.multiecho_button.isEnabled()
        assert not window.multiecho_delay_spin.isEnabled()
        window.multiecho_count_spin.setValue(8)
        release.set();wait_idle(app,window)
        assert window.effected_clip is None
    finally:release.set();wait_idle(app,window);window.close()


def test_enable_multiecho_also_enables_main_processing_route(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        window.multiecho_check.setChecked(True)
        window.multiecho_strength_spin.setValue(.4)
        assert window.effects_check.isChecked()
        window.process_button.click();wait_idle(app,window)
        assert not window.last_error and window.effected_clip is not None
        assert window.current_clip is window.effected_clip
        assert len(window.current_clip.samples)==len(window.clip.samples)+3*round(.3*window.clip.sample_rate)
    finally:window.close()
