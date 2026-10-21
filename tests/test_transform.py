"""Catch coupled tempo/pitch, silent output, incorrect lengths and unsafe allocations."""
import importlib

import numpy as np
import pytest

from voice_effect_lab.audio.data import AudioError


def api():
    return importlib.import_module('voice_effect_lab.audio.transform')


def tone(hz=440, size=16000, sample_rate=8000):
    return .2 * np.sin(2 * np.pi * hz * np.arange(size) / sample_rate)


def dominant_frequency(values, sample_rate=8000):
    # Ignore the phase vocoder's boundary transients before measuring real audio.
    middle = values[len(values) // 5:len(values) * 4 // 5]
    spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle))))
    return np.fft.rfftfreq(len(middle), 1 / sample_rate)[np.argmax(spectrum)]


@pytest.mark.parametrize('rate,want_length', [(.5, 32000), (2, 8000), (1.3, 12308)])
def test_tempo_changes_duration_without_changing_pitch(rate, want_length):
    source = tone()
    original = source.copy()
    result = api().time_stretch(source, rate, 8000)
    assert len(result) == want_length
    assert abs(dominant_frequency(result) - 440) < 2
    assert .08 < np.std(result[len(result)//5:-len(result)//5]) < .2
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize('semitones,want_hz', [(12, 880), (-12, 220), (7, 659.255)])
def test_pitch_shift_changes_frequency_without_changing_duration(semitones, want_hz):
    source = tone()
    result = api().pitch_shift(source, semitones, 8000)
    assert len(result) == 16000
    assert abs(dominant_frequency(result) - want_hz) < 2
    assert .07 < np.std(result[3200:-3200]) < .2
    np.testing.assert_array_equal(source, tone())


@pytest.mark.parametrize('function,parameter', [('time_stretch', 1), ('pitch_shift', 0)])
def test_neutral_is_exact_independent_copy_without_clipping(function, parameter):
    source = np.array([1.2, -.3, 0, 2], dtype=np.float32)
    result = getattr(api(), function)(source, parameter, 8000)
    np.testing.assert_array_equal(result, source)
    assert not np.shares_memory(result, source)


@pytest.mark.parametrize('size', [1, 2, 17, 2048])
@pytest.mark.parametrize('sample_rate', [1, 10, 8000])
def test_short_low_rate_and_silent_audio_stay_finite(size, sample_rate):
    module = api()
    for source in (np.zeros(size), np.full(size, .2)):
        for rate in (.5, 2):
            result = module.time_stretch(source, rate, sample_rate)
            assert len(result) == max(1, round(size / rate))
            assert np.isfinite(result).all()
            if not source.any():
                assert not result.any()
        for semitones in (-12, 12):
            result = module.pitch_shift(source, semitones, sample_rate)
            assert len(result) == size
            assert np.isfinite(result).all()
            if not source.any():
                assert not result.any()


@pytest.mark.parametrize('function,parameter', [
    ('time_stretch', 0), ('time_stretch', .49), ('time_stretch', 2.01),
    ('time_stretch', True), ('time_stretch', np.nan), ('time_stretch', '1'),
    ('pitch_shift', -12.1), ('pitch_shift', 12.1), ('pitch_shift', np.inf),
    ('pitch_shift', True), ('pitch_shift', None),
])
def test_invalid_parameters_raise_audio_error(function, parameter):
    with pytest.raises(AudioError):
        getattr(api(), function)(np.zeros(32), parameter, 8000)


@pytest.mark.parametrize('source', [[], [[0, 1]], [np.nan], [np.inf], [1j], ['a']])
@pytest.mark.parametrize('function,parameter', [('time_stretch', 1), ('pitch_shift', 0)])
def test_invalid_samples_rejected_even_in_neutral_mode(source, function, parameter):
    with pytest.raises(AudioError):
        getattr(api(), function)(source, parameter, 8000)


@pytest.mark.parametrize('sample_rate', [0, -1, True, 1.5, np.inf])
def test_invalid_sample_rate_rejected(sample_rate):
    with pytest.raises(AudioError):
        api().time_stretch(np.zeros(32), 1, sample_rate)


def test_long_intermediate_duration_is_allowed_within_sample_budget():
    result = api().time_stretch(np.zeros(301), .5, 1)
    assert len(result) == 602
    assert not result.any()


@pytest.mark.parametrize('function,parameter', [('time_stretch', .5), ('pitch_shift', 12)])
def test_intermediate_limit_checked_before_allocation(monkeypatch, function, parameter):
    module = api()
    monkeypatch.setattr(module, 'MAX_OUTPUT_SAMPLES', 63)
    with pytest.raises(AudioError):
        getattr(module, function)(np.zeros(32), parameter, 8000)


def test_input_limit_checked_even_when_neutral(monkeypatch):
    module = api()
    monkeypatch.setattr(module, 'MAX_OUTPUT_SAMPLES', 31)
    with pytest.raises(AudioError):
        module.pitch_shift(np.zeros(32), 0, 8000)
