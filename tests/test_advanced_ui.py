"""高级音效、试听音量与中文TTS在真实主窗口中的整合。"""
import numpy as np
from scipy.io import wavfile
from test_processing_ui import prepare_window
from test_player import DeviceOutput,audio_format
from test_ui import wait_idle


def test_listen_volume_and_mute_are_live_and_do_not_change_export(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        source=window.clip;output=DeviceOutput(audio_format(source.sample_rate))
        window.player._output_factory=lambda rate,parent:(output,output.fmt)
        window.listen_volume_spin.setValue(40);window.play_button.click()
        assert output.volume==.4 and window.player.volume==40
        buffer=window.player._buffer;output.consume(800)
        position=window.player.position_ms
        window.listen_volume_slider.setValue(65)
        assert window.player._buffer is buffer and window.player.position_ms==position
        window.mute_button.setChecked(True)
        assert output.volume==0 and window.player.volume==65
        window.mute_button.setChecked(False)
        assert output.volume==.65
        path=tmp_path/'原声.wav';window.save_to(path);wait_idle(app,window)
        rate,values=wavfile.read(path)
        np.testing.assert_allclose(values/32768,source.samples,atol=1/32768)
        assert window.clip is source
    finally:window.close()


def test_advanced_controls_enable_pipeline_and_export_real_changed_duration(app,tmp_path):
    from voice_effect_lab.audio.effects import apply_effects
    window,_=prepare_window(app,tmp_path)
    try:
        raw=window.clip
        window.independent_check.setChecked(True)
        window.pitch_spin.setValue(5);window.tempo_spin.setValue(1.25)
        window.reverb_check.setChecked(True);window.reverb_decay_spin.setValue(.4)
        window.modulation_check.setChecked(True);window.compression_check.setChecked(True)
        assert window.effects_check.isChecked()
        window.advanced_apply_button.click();wait_idle(app,window)
        assert not window.last_error
        out=window.effected_clip
        assert out is window.current_clip and window.player._clip is out
        assert len(out.samples)==round(len(raw.samples)/1.25)+round(.4*raw.sample_rate)-1
        np.testing.assert_array_equal(out.samples,apply_effects(raw,window._effect_settings()).clip.samples)
        assert window.clip is raw
        path=tmp_path/'高级音效.wav';window.save_to(path);wait_idle(app,window)
        rate,values=wavfile.read(path)
        assert rate==raw.sample_rate
        np.testing.assert_allclose(values/32768,out.samples,atol=1/32768)
        window.reverb_wet_spin.setValue(.5)
        assert window.effected_clip is None
    finally:window.close()


def test_chain_order_and_all_advanced_resets_are_consistent(app,tmp_path):
    from voice_effect_lab.audio.effects import DEFAULT_CHAIN_ORDER
    window,_=prepare_window(app,tmp_path)
    try:
        window.chain_list.setCurrentRow(0);window.chain_down_button.click()
        assert window._effect_settings().chain_order[1]=='pitch_tempo'
        window.independent_check.setChecked(True);window.pitch_spin.setValue(7)
        window.tempo_spin.setValue(.75);window.reverb_check.setChecked(True)
        window.modulation_check.setChecked(True);window.compression_check.setChecked(True)
        window.preset_buttons['自定义'].click()
        assert window.independent_check.isChecked()
        window.preset_buttons['电话音'].click()
        assert not any(w.isChecked() for w in [window.independent_check,window.reverb_check,window.modulation_check,window.compression_check])
        assert window._effect_settings().chain_order==DEFAULT_CHAIN_ORDER
        assert window.pitch_spin.value()==0 and window.pitch_slider.value()==0
        assert window.tempo_spin.value()==1 and window.tempo_slider.value()==100
        window.reverb_check.setChecked(True);window.reverb_wet_spin.setValue(.7)
        window.reset_processing()
        assert not window.reverb_check.isChecked() and window.reverb_wet_spin.value()==.25
        assert window.reverb_wet_slider.value()==25
    finally:window.close()


def test_tts_stays_independent_until_send_and_volume_preview_is_shared(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        source=window.clip
        window.tts_text.setPlainText('你好，这是公开编写的语音功能测试。')
        window.tts_generate_button.click();wait_idle(app,window)
        assert not window.last_error and window.tts_clip is not None
        assert window.clip is source
        dest=tmp_path/'中文.wav';window.save_tts_to(dest);wait_idle(app,window)
        rate,data=wavfile.read(dest)
        assert rate==window.tts_clip.sample_rate and len(data)==len(window.tts_clip.samples)
        output=DeviceOutput(audio_format(rate))
        window.tts_player._output_factory=lambda rate,parent:(output,output.fmt)
        window.listen_volume_spin.setValue(35);window.play_tts()
        assert window.tts_player.state=='playing' and output.volume==.35
        window.mute_button.setChecked(True);assert output.volume==0
        window.start_denoise();assert window.tts_player.state=='stopped';wait_idle(app,window)
        generated=window.tts_clip
        window.send_tts_to_workbench()
        assert window.clip is generated and window.denoised_clip is None
        assert window.current_clip is generated
    finally:window.close()


def test_tts_volume_controls_sync_with_workbench_without_changing_audio(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        original=window.clip
        window.tts_volume_spin.setValue(45)
        assert window.listen_volume_spin.value()==45 and window.tts_player.volume==45
        window.listen_volume_slider.setValue(80)
        assert window.tts_volume_spin.value()==80 and window.tts_volume_slider.value()==80
        window.tts_mute_button.setChecked(True)
        assert window.mute_button.isChecked() and window.player.muted and window.tts_player.muted
        window.mute_button.setChecked(False)
        assert not window.tts_mute_button.isChecked()
        assert window.clip is original
    finally:window.close()


def test_leaving_tts_page_stops_preview_before_other_audio_actions(app,tmp_path):
    window,_=prepare_window(app,tmp_path)
    try:
        window.tts_text.setPlainText('你好，正在测试页面切换。');window.start_tts();wait_idle(app,window)
        output=DeviceOutput(audio_format(window.tts_clip.sample_rate))
        window.tts_player._output_factory=lambda rate,parent:(output,output.fmt)
        window.tabs.setCurrentIndex(1);window.play_tts()
        assert window.tts_player.state=='playing'
        window.tabs.setCurrentIndex(0)
        assert window.tts_player.state=='stopped'
    finally:window.close()
