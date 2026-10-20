"""防止选段边界/确认遗漏，以及对话框取消时泄露播放资源。"""
import numpy as np
import pytest
from voice_effect_lab.audio.data import AudioClip, AudioError
from voice_effect_lab.audio.noise import extract_noise
from voice_effect_lab.ui.noise_dialog import NoiseSelectionDialog


def source_clip():
    return AudioClip(np.linspace(-.8,.8,8000),8000,'source.wav')


def test_extract_noise_has_correct_samples_and_keeps_source():
    clip=source_clip(); original=clip.samples.copy()
    noise=extract_noise(clip,.1,.3)
    np.testing.assert_array_equal(noise.samples,clip.samples[800:2400])
    assert noise.sample_rate==8000
    assert not np.shares_memory(noise.samples,clip.samples)
    np.testing.assert_array_equal(clip.samples,original)


@pytest.mark.parametrize('start,end',[(-.1,.3),(.5,.4),(0,1.1),(0,.099),(np.nan,.5),(0,np.inf)])
def test_bad_noise_interval_is_rejected(start,end):
    with pytest.raises(AudioError): extract_noise(source_clip(),start,end)


def test_dialog_requires_explicit_pure_noise_confirmation(app):
    dialog=NoiseSelectionDialog(source_clip())
    dialog.set_range(.1,.3)
    assert not dialog.use_button.isEnabled()
    dialog.accept()
    assert dialog.selected_clip is None
    dialog.confirm_check.setChecked(True)
    assert dialog.use_button.isEnabled()
    dialog.accept()
    np.testing.assert_array_equal(dialog.selected_clip.samples,source_clip().samples[800:2400])


def test_dialog_reversed_range_cannot_be_accepted(app):
    dialog=NoiseSelectionDialog(source_clip())
    dialog.set_range(.8,.2)
    dialog.confirm_check.setChecked(True)
    assert not dialog.use_button.isEnabled()
    dialog.accept()
    assert dialog.selected_clip is None
    dialog.reject()


def test_plot_selection_updates_numeric_controls(app):
    dialog=NoiseSelectionDialog(source_clip())
    dialog.set_range(.25,.75)
    assert dialog.start_spin.value()==.25
    assert dialog.end_spin.value()==.75
    assert dialog.selector.extents==(.25,.75)
    dialog.reject()


def test_cancel_leaves_no_selected_clip_or_player_data(app):
    dialog=NoiseSelectionDialog(source_clip())
    dialog.confirm_check.setChecked(True)
    dialog.reject()
    assert dialog.selected_clip is None
    assert dialog.player.state=='stopped'
    assert dialog.player.duration_ms==0


def test_preview_device_error_is_reported_in_dialog(app,monkeypatch):
    from voice_effect_lab.audio.player import PlaybackError
    dialog=NoiseSelectionDialog(source_clip())
    def unavailable(): raise PlaybackError('没有输出设备')
    monkeypatch.setattr(dialog.player,'play',unavailable)
    dialog.preview()
    assert '没有输出设备' in dialog.message.text()
    assert dialog.selected_clip is None
    dialog.reject()
