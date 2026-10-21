"""真实Qt四阶段链路、参数依赖及后台失败保护。"""
from threading import Event
import numpy as np
from scipy.io import wavfile
from test_processing_ui import prepare_window
from test_enhancement_ui import run_enhance
from test_ui import wait_idle


def run_effects(app,window):
    window.effects_button.click();wait_idle(app,window)
    assert not window.last_error,window.last_error
    assert window.effected_clip is not None


def test_raw_effect_updates_duration_plots_player_and_export(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        raw=window.clip
        window.effects_check.setChecked(True)
        window.rate_spin.setValue(2)
        run_effects(app,window)
        assert window.current_clip is window.effected_clip
        assert window.player._clip is window.effected_clip
        assert window.stages.checkedId()==3
        assert window.clip is raw
        assert window.enhanced_clip is None and window.denoised_clip is None
        assert abs(window.current_clip.duration-raw.duration/2)<=1/raw.sample_rate
        np.testing.assert_array_equal(window.plots.axes[0,1].lines[0].get_ydata(),window.effected_analysis.wave_values)
        assert f'{window.current_clip.duration:.2f}' in window.progress_label.text()
        dest=tmp_path/'音效.wav';selected=window.current_clip
        window.save_to(dest);wait_idle(app,window)
        rate,values=wavfile.read(dest)
        assert rate==raw.sample_rate
        np.testing.assert_allclose(values/32768,selected.samples,atol=1/32768)
    finally:window.close()


def test_presets_sync_parameters_and_manual_edit_marks_custom(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        for name,speed,low in [('磁性',1,4),('娃娃音',1.35,-3),('大叔音',.8,4)]:
            window.preset_buttons[name].click()
            assert window.preset_name==name
            assert window.enhance_check.isChecked()
            assert window.effects_check.isChecked()
            assert window.rate_spin.value()==speed
            assert window.eq_low_spin.value()==low
            assert window.rate_slider.value()==round(speed*100)
        window.preset_buttons['电话音'].click()
        assert window.telephone_check.isChecked()
        assert window.enhance_check.isChecked()
        window.preset_buttons['山谷回声'].click()
        assert not window.telephone_check.isChecked()
        assert window.echo_strength_spin.value()==.45
        run_effects(app,window)
        assert len(window.effected_clip.samples)==len(window.clip.samples)+round(.3*window.clip.sample_rate)
        window.rate_spin.setValue(1.1)
        assert window.preset_name=='自定义'
        assert window.effected_clip is None
        window.preset_buttons['原声'].click()
        assert not window.effects_check.isChecked()
        assert window.enhance_check.isChecked()
        assert window.echo_strength_spin.value()==0
    finally:window.close()


def test_complete_chain_builds_four_stages_and_repeats_from_upstream(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        window.use_denoise_check.setChecked(True)
        window.preset_buttons['娃娃音'].click()
        window.process_button.click();wait_idle(app,window)
        assert not window.last_error
        assert window.denoised_clip is not None and window.enhanced_clip is not None
        first=window.effected_clip
        assert first is not None
        assert window._effects_from_denoise and window._effects_from_enhancement
        for i,value in enumerate([window.clip,window.denoised_clip,window.enhanced_clip,first]):
            window.stages.button(i).click()
            assert window.current_clip is value and window.player._clip is value
        run_effects(app,window)
        np.testing.assert_array_equal(window.effected_clip.samples,first.samples)
        window.echo_delay_spin.setValue(.8)
        assert window.effected_clip is None
        assert window.enhanced_clip is not None and window.denoised_clip is not None
        assert window.current_clip is window.enhanced_clip
    finally:window.close()


def test_unrelated_denoise_does_not_clear_raw_effect_but_source_policy_does(app,tmp_path):
    window,path=prepare_window(app,tmp_path)
    try:
        run_effects(app,window);output=window.effected_clip
        window.denoise_spin.setValue(70)
        assert window.effected_clip is output and window.current_clip is output
        window.gain_spin.setValue(.5)  # 增强未启用，独立原声音效仍有效。
        assert window.effected_clip is output
        window.use_denoise_check.setChecked(True)
        assert window.effected_clip is None
        run_effects(app,window)
        assert window._effects_from_denoise and not window._effects_from_enhancement
        window.denoise_spin.setValue(80)
        assert window.effected_clip is None
        run_effects(app,window)
        window.load_path(path);wait_idle(app,window)
        assert window.effected_clip is None and window.current_clip is window.clip
    finally:window.close()


def test_enhancement_toggle_and_recompute_invalidate_dependent_effect(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        run_effects(app,window)
        window.enhance_check.setChecked(True)
        assert window.effected_clip is None
        run_effects(app,window)
        window.gain_spin.setValue(.5)
        assert window.effected_clip is None
        run_effects(app,window)
        run_enhance(app,window)
        assert window.effected_clip is None
        run_effects(app,window)
        window.enhance_check.setChecked(False)
        assert window.effected_clip is None
    finally:window.close()


def test_effects_work_without_noise_unless_strategy_requires_it(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        window.auto_noise_check.setChecked(False)
        run_effects(app,window)
        window.use_denoise_check.setChecked(True)
        assert not window.effects_button.isEnabled()
        window.start_effects()
        assert window.last_error and not window.busy
    finally:window.close()


def test_final_step_failure_preserves_old_effect_and_no_partial_upstream(app,tmp_path,monkeypatch):
    window,_=prepare_window(app,tmp_path)
    try:
        run_effects(app,window);old=window.effected_clip
        def fail(*args):raise ValueError('合成测试故障')
        monkeypatch.setattr('voice_effect_lab.ui.effects_controller.apply_effects',fail)
        window.start_effects();wait_idle(app,window)
        assert window.last_error and window.effected_clip is old
        window.enhance_check.setChecked(True)
        window.use_denoise_check.setChecked(True)
        window.start_effects();wait_idle(app,window)
        assert window.last_error
        assert window.denoised_clip is None and window.enhanced_clip is None
        assert window.effected_clip is None
    finally:window.close()


def test_stale_result_discard_and_busy_close_wait(app,tmp_path,monkeypatch):
    from voice_effect_lab.ui.effects_controller import apply_effects
    window,_=prepare_window(app,tmp_path);entered,release=Event(),Event()
    def blocked(*args):
        entered.set();assert release.wait(10);return apply_effects(*args)
    monkeypatch.setattr('voice_effect_lab.ui.effects_controller.apply_effects',blocked)
    try:
        window.show();window.start_effects();assert entered.wait(5)
        assert not window.effects_button.isEnabled() and not window.rate_spin.isEnabled()
        assert not window.preset_buttons['娃娃音'].isEnabled()
        window.rate_spin.setValue(1.5)
        window.close();assert window.isVisible()
        release.set();wait_idle(app,window)
        assert window.effected_clip is None
        assert not window.isVisible()
    finally:release.set();window.close()


def test_reset_restores_effect_spins_sliders_and_preserves_source(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        source=window.clip
        window.preset_buttons['山谷回声'].click()
        window.rate_spin.setValue(.5);window.echo_delay_spin.setValue(.8)
        run_effects(app,window)
        window.reset_processing()
        assert window.clip is source and window.current_clip is source
        assert window.effected_clip is None and not window.effects_check.isChecked()
        assert window.rate_spin.value()==1 and window.rate_slider.value()==100
        assert window.echo_delay_spin.value()==.3 and window.echo_delay_slider.value()==30
        assert window.echo_strength_spin.value()==0 and window.echo_strength_slider.value()==0
        assert window.preset_name=='原声'
    finally:window.close()
