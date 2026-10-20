"""实际增强、处理链、缓存依赖与三阶段导出。"""
from threading import Event
import numpy as np
from scipy.io import wavfile
from test_processing_ui import prepare_window, process
from test_ui import wait_idle
from voice_effect_lab.audio.enhance import enhance_audio


def run_enhance(app,window):
    window.enhance_button.click()
    wait_idle(app,window)
    assert not window.last_error,window.last_error
    assert window.enhanced_clip is not None


def test_one_click_enhance_uses_original_and_matches_plot_player_export(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        source=window.clip
        assert not window.enhance_check.isChecked()
        assert not window.gain_spin.isEnabled()
        assert window.enhance_button.isEnabled()
        run_enhance(app,window)
        assert window.enhance_check.isChecked()
        assert window.stages.checkedId()==2
        assert window.current_clip is window.enhanced_clip
        assert window.player._clip is window.enhanced_clip
        assert window.clip is source
        assert window.denoised_clip is None
        assert np.max(np.abs(window.enhanced_clip.samples))==np.float32(.95)
        np.testing.assert_array_equal(window.plots.axes[0,1].lines[0].get_ydata(),window.enhanced_analysis.wave_values)
        path=tmp_path/'增强.wav';expected=window.enhanced_clip
        window.save_to(path);wait_idle(app,window)
        rate,samples=wavfile.read(path)
        assert rate==source.sample_rate
        np.testing.assert_allclose(samples/32768,expected.samples,atol=1/32768)
        assert '增强' in window.save_button.text()
        window.select_stage(0)
        assert window.player._clip is source
        window.select_stage(2)
        assert window.player._clip is expected
        assert not window.stages.button(3).isEnabled()
    finally:window.close()


def test_gain_eq_and_repeated_enhance_apply_to_source_not_previous_result(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        window.enhance_check.setChecked(True)
        window.gain_spin.setValue(.5)
        window.clarity_spin.setValue(40)
        window.eq_low_spin.setValue(-2)
        run_enhance(app,window)
        first=window.enhanced_clip.samples.copy()
        assert np.max(np.abs(first))==np.float32(.475)
        run_enhance(app,window)
        np.testing.assert_array_equal(window.enhanced_clip.samples,first)
        window.gain_spin.setValue(.8)
        assert window.enhanced_clip is None
        assert window.stages.checkedId()==0
        assert not window.stages.button(2).isEnabled()
    finally:window.close()


def test_use_denoised_input_generates_whole_chain_then_selects_enhanced(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        window.use_denoise_check.setChecked(True)
        window.enhance_check.setChecked(True)
        window.process_button.click();wait_idle(app,window)
        assert not window.last_error
        assert window.denoised_clip is not None
        expected=enhance_audio(window.denoised_clip,window._enhancement_settings()).clip
        np.testing.assert_array_equal(window.enhanced_clip.samples,expected.samples)
        assert window._enhanced_from_denoise
        assert window.stages.button(1).isEnabled() and window.stages.button(2).isEnabled()
        window.select_stage(1)
        assert window.player._clip is window.denoised_clip
        window.select_stage(2)
        assert window.player._clip is window.enhanced_clip
        assert '去噪' in window.enhance_status_label.text()
    finally:window.close()


def test_parameter_dependency_keeps_unaffected_caches_and_invalidates_dependents(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        process(app,window)
        noise_result=window.denoised_clip
        run_enhance(app,window)  # 默认直接增强原声。
        raw_enhanced=window.enhanced_clip
        window.denoise_spin.setValue(80)
        assert window.denoised_clip is None
        assert window.enhanced_clip is raw_enhanced
        window.start_denoise();wait_idle(app,window)
        noise_result=window.denoised_clip
        window.clarity_spin.setValue(30)
        assert window.enhanced_clip is None
        assert window.denoised_clip is noise_result
        window.use_denoise_check.setChecked(True)
        run_enhance(app,window)
        window.denoise_spin.setValue(70)
        assert window.denoised_clip is None and window.enhanced_clip is None
        assert window.stages.checkedId()==0
    finally:window.close()


def test_disabled_reference_confirmation_only_blocks_chains_that_need_denoise(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        window.auto_noise_check.setChecked(False)
        run_enhance(app,window)  # 无底噪仍能独立增强。
        window.use_denoise_check.setChecked(True)
        assert not window.enhance_button.isEnabled()
        assert not window.process_button.isEnabled()
        window.start_enhance()
        assert window.last_error and not window.busy
    finally:window.close()


def test_bad_import_and_failed_enhance_preserve_valid_output(app,tmp_path,monkeypatch):
    from voice_effect_lab.ui import enhancement_controller
    window,path=prepare_window(app,tmp_path)
    try:
        run_enhance(app,window)
        result=window.enhanced_clip
        bad=tmp_path/'bad.wav';bad.write_bytes(b'bad')
        window.load_path(bad);wait_idle(app,window)
        assert window.enhanced_clip is result
        def fail(*args):raise RuntimeError('增强失败测试')
        monkeypatch.setattr(enhancement_controller,'enhance_audio',fail)
        window.start_enhance();wait_idle(app,window)
        assert '增强失败测试' in window.last_error
        assert window.enhanced_clip is result and window.current_clip is result
        window.load_path(path);wait_idle(app,window)
        assert window.enhanced_clip is None and window.denoised_clip is None
    finally:window.close()


def block_enhance(monkeypatch):
    from voice_effect_lab.ui import enhancement_controller
    release=Event();original=enhancement_controller.enhance_audio
    def block(*args):
        assert release.wait(10)
        return original(*args)
    monkeypatch.setattr(enhancement_controller,'enhance_audio',block)
    return release


def test_processing_snapshot_and_controls_prevent_stale_enhancement(app,tmp_path,monkeypatch):
    window,_=prepare_window(app,tmp_path);release=block_enhance(monkeypatch)
    try:
        window.start_enhance()
        for control in (window.gain_spin,window.enhance_check,window.use_denoise_check,
                        window.enhance_button,window.record_button,window.import_button):
            assert not control.isEnabled()
        window.gain_spin.setValue(.6)  # 程序化变化仍必须使快照失效。
        release.set();wait_idle(app,window)
        assert window.enhanced_clip is None
    finally:
        release.set();wait_idle(app,window);window.close()


def test_failed_chain_does_not_publish_partial_new_denoised_cache(app,tmp_path,monkeypatch):
    from voice_effect_lab.ui import enhancement_controller
    window,_=prepare_window(app,tmp_path)
    try:
        window.use_denoise_check.setChecked(True)
        def fail(*args):raise RuntimeError('链路增强失败')
        monkeypatch.setattr(enhancement_controller,'enhance_audio',fail)
        window.start_enhance();wait_idle(app,window)
        assert window.denoised_clip is None and window.enhanced_clip is None
        assert window.last_error
    finally:window.close()


def test_closing_waits_for_enhancement_and_reset_restores_bypass(app,tmp_path,monkeypatch):
    window,_=prepare_window(app,tmp_path)
    run_enhance(app,window)
    window.use_denoise_check.setChecked(True)
    window.reset_button.click()
    assert not window.enhance_check.isChecked() and not window.use_denoise_check.isChecked()
    assert window.enhanced_clip is None and window.denoised_clip is None
    assert window.gain_spin.value()==1 and window.clarity_spin.value()==0
    release=block_enhance(monkeypatch)
    try:
        window.show();window.start_enhance();window.close()
        assert window._close_pending and window.isVisible()
        release.set();wait_idle(app,window)
        assert not window.isVisible()
        assert window.player.state=='stopped'
    finally:
        release.set();wait_idle(app,window);window.close()


def test_reset_keeps_sliders_and_numeric_values_synchronized(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        window.enhance_check.setChecked(True)
        window.gain_spin.setValue(.4)
        window.clarity_spin.setValue(80)
        window.eq_low_spin.setValue(7)
        window.reset_button.click()
        for slider,spin in [(window.gain_slider,window.gain_spin),
                            (window.clarity_slider,window.clarity_spin),
                            (window.eq_low_slider,window.eq_low_spin)]:
            assert slider.value()==round(spin.value()*10**spin.decimals())
    finally:window.close()
