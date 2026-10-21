"""方法选择必须贯穿去噪/增强/音效，切换不能保留旧结果。"""
from threading import Event
import numpy as np
from test_processing_ui import prepare_window,block_processing
from test_ui import wait_idle


def test_default_wiener_real_output_and_reset(app,tmp_path):
    from voice_effect_lab.audio.wiener import wiener_denoise
    window,_=prepare_window(app,tmp_path)
    try:
        assert window.denoise_method_combo.currentData()=='wiener'
        window.start_denoise();wait_idle(app,window)
        assert not window.last_error
        np.testing.assert_array_equal(window.current_clip.samples,wiener_denoise(window.clip).clip.samples)
        assert '维纳' in window.denoise_status_label.text()
        assert 'SNR' not in window.denoise_status_label.text()
        window.denoise_method_combo.setCurrentIndex(1)
        assert window.denoised_clip is None
        window.reset_processing()
        assert window.denoise_method_combo.currentData()=='wiener'
    finally:window.close()


def test_switch_method_invalidates_all_dependent_caches_and_chain_uses_choice(app,tmp_path):
    from voice_effect_lab.audio.denoising import denoise_audio
    from voice_effect_lab.audio.enhance import enhance_audio
    from voice_effect_lab.audio.effects import apply_effects
    window,_=prepare_window(app,tmp_path)
    try:
        window.use_denoise_check.setChecked(True)
        window.enhance_check.setChecked(True)
        window.start_effects();wait_idle(app,window)
        assert window.denoised_clip and window.enhanced_clip and window.effected_clip
        window.denoise_method_combo.setCurrentIndex(1)
        assert window.denoised_clip is None and window.enhanced_clip is None and window.effected_clip is None
        assert window.current_clip is window.clip
        window.start_effects();wait_idle(app,window)
        expected_noise=denoise_audio(window.clip,method='spectral').clip
        np.testing.assert_array_equal(window.denoised_clip.samples,expected_noise.samples)
        expected=apply_effects(enhance_audio(expected_noise,window._enhancement_settings()).clip,window._effect_settings()).clip
        np.testing.assert_array_equal(window.effected_clip.samples,expected.samples)
        assert '谱减' in window.denoise_status_label.text()
    finally:window.close()


def test_method_change_preserves_independent_original_enhancement(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        window.start_enhance();wait_idle(app,window)
        enhanced=window.enhanced_clip
        window.denoise_method_combo.setCurrentIndex(1)
        assert window.enhanced_clip is enhanced and window.current_clip is enhanced
    finally:window.close()


def test_background_method_switch_discards_old_result(app,tmp_path,monkeypatch):
    window,_=prepare_window(app,tmp_path);release=block_processing(monkeypatch)
    try:
        window.start_denoise()
        assert not window.denoise_method_combo.isEnabled()
        window.denoise_method_combo.setCurrentIndex(1)
        release.set();wait_idle(app,window)
        assert window.denoised_clip is None and window.current_clip is window.clip
    finally:release.set();wait_idle(app,window);window.close()


def test_enhancement_chain_captures_method_and_rejects_stale_commit(app,tmp_path,monkeypatch):
    from voice_effect_lab.ui.enhancement_controller import enhance_audio
    entered,release=Event(),Event()
    def blocked(*args):
        entered.set();assert release.wait(10);return enhance_audio(*args)
    monkeypatch.setattr('voice_effect_lab.ui.enhancement_controller.enhance_audio',blocked)
    window,_=prepare_window(app,tmp_path)
    try:
        window.use_denoise_check.setChecked(True)
        window.start_enhance();assert entered.wait(5)
        window.denoise_method_combo.setCurrentIndex(1)
        release.set();wait_idle(app,window)
        assert window.denoised_clip is None and window.enhanced_clip is None
    finally:release.set();wait_idle(app,window);window.close()
