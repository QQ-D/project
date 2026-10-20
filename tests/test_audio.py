"""防止声道/整数缩放错误、导出改变音量、静音/短片段分析崩溃。"""
import numpy as np
import pytest
from scipy.io import wavfile

from voice_effect_lab.audio.data import AudioClip, AudioError, load_wav, save_wav
from voice_effect_lab.analysis import analyze


def test_stereo_import_averages_channels_and_preserves_rate(tmp_path):
    path = tmp_path / '立体声.wav'
    wavfile.write(path, 8000, np.array([[16384, 0], [-16384, 16384]], dtype=np.int16))
    clip = load_wav(path)
    np.testing.assert_allclose(clip.samples, [.25, 0], atol=1e-7)
    assert clip.sample_rate == 8000
    assert clip.source_channels == 2
    assert clip.duration == .00025


def test_unsigned_pcm_has_zero_at_128(tmp_path):
    path = tmp_path / '8bit.wav'
    wavfile.write(path, 8000, np.array([0, 128, 255], dtype=np.uint8))
    np.testing.assert_allclose(load_wav(path).samples, [-1, 0, 127/128])


def test_pcm16_roundtrip_does_not_normalize(tmp_path):
    clip = AudioClip(np.array([-.4, -.1, 0, .1, .4]), 16000, 'quiet')
    path = tmp_path / '结果.wav'
    save_wav(path, clip)
    result = load_wav(path)
    np.testing.assert_allclose(result.samples, clip.samples, atol=1/32768)
    assert result.sample_rate == 16000


@pytest.mark.parametrize('samples', [[], [np.nan], [np.inf], [1.1]])
def test_invalid_samples_are_rejected(samples):
    with pytest.raises(AudioError):
        AudioClip(np.array(samples), 8000, 'bad')


def test_empty_wav_is_rejected(tmp_path):
    path = tmp_path / 'empty.wav'
    wavfile.write(path, 8000, np.array([], dtype=np.int16))
    with pytest.raises(AudioError):
        load_wav(path)


def test_corrupted_file_has_readable_error(tmp_path):
    path = tmp_path / 'bad.wav'
    path.write_bytes(b'not a WAV')
    with pytest.raises(AudioError, match='WAV'):
        load_wav(path)


def test_long_audio_is_rejected(tmp_path):
    path = tmp_path / 'long.wav'
    wavfile.write(path, 8000, np.zeros(8000*301, dtype=np.int16))
    with pytest.raises(AudioError, match='300'):
        load_wav(path)


def test_sine_analysis_finds_frequency_and_amplitude():
    t = np.arange(8000)/8000
    clip = AudioClip(.5*np.sin(2*np.pi*440*t), 8000, 'sine')
    result = analyze(clip)
    peak = np.argmax(result.spectrum_db)
    assert result.frequencies[peak] == pytest.approx(440, abs=1)
    assert result.spectrum_db[peak] == pytest.approx(-6.0206, abs=.05)
    assert result.spectrogram_db.shape == (len(result.spec_frequencies), len(result.spec_times))
    assert np.isfinite(result.spectrogram_db).all()


@pytest.mark.parametrize('samples', [[0], [0, 0, 0], [.25]])
def test_short_and_silent_analysis_is_finite(samples):
    result = analyze(AudioClip(np.array(samples), 8000, 'short'))
    assert np.isfinite(result.spectrum_db).all()
    assert np.isfinite(result.spectrogram_db).all()
    assert len(result.wave_times) == len(result.wave_values)


def test_waveform_downsample_preserves_impulse():
    samples = np.zeros(100000, dtype=np.float32)
    samples[12345] = .9
    result = analyze(AudioClip(samples, 16000, 'impulse'))
    assert max(result.wave_values) == pytest.approx(.9)
    assert len(result.wave_values) <= 6000


def test_truncated_wav_is_rejected_instead_of_silently_shortened(tmp_path):
    path = tmp_path/'truncated.wav'
    wavfile.write(path, 8000, np.zeros(8000, dtype=np.int16))
    path.write_bytes(path.read_bytes()[:8044])
    with pytest.raises(AudioError, match='WAV'):
        load_wav(path)
