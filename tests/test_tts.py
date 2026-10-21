"""Offline Mandarin TTS uses only public, self-authored text and never plays it."""
from concurrent.futures import ThreadPoolExecutor
import importlib

import numpy as np
import pytest

from voice_effect_lab.audio.data import AudioClip, AudioError


def tts_module():
    spec = importlib.util.find_spec('voice_effect_lab.audio.tts')
    assert spec is not None, 'Offline Mandarin synthesis is not implemented'
    return importlib.import_module('voice_effect_lab.audio.tts')


@pytest.mark.parametrize('text,speed', [('', 1), ('  \n', 1), (None, 1), ('中'*401, 1),
    ('测试\x00文字', 1), ('公开测试', .49), ('公开测试', 2.01), ('公开测试', float('nan')),
    ('公开测试', float('inf')), ('公开测试', '快'), ('公开测试', True)])
def test_invalid_text_or_speed_is_user_facing_error(text, speed):
    with pytest.raises(AudioError):
        tts_module().synthesize_chinese(text, speed)


def test_actual_mandarin_returns_finite_nonzero_audio_and_speed_changes_duration():
    tts = tts_module()
    available, reason = tts.tts_available()
    if not available:
        pytest.skip(reason)
    slow = tts.synthesize_chinese('这是公开编写的中文语音测试。', .75)
    fast = tts.synthesize_chinese('这是公开编写的中文语音测试。', 1.5)
    assert isinstance(slow, AudioClip)
    assert slow.sample_rate >= 8000 and slow.source_channels == 1
    assert slow.samples.dtype == np.float32
    assert not slow.samples.flags.writeable
    assert np.isfinite(slow.samples).all()
    assert 0 < np.max(np.abs(slow.samples)) <= 1
    assert 0 < fast.duration < slow.duration <= 300


def test_parallel_requests_do_not_share_engine_state():
    tts = tts_module()
    if not tts.tts_available()[0]:
        pytest.skip('Mandarin engine unavailable')
    with ThreadPoolExecutor(max_workers=2) as executor:
        clips = list(executor.map(tts.synthesize_chinese, ['公开测试甲。', '公开测试乙。']))
    assert all(clip.samples.size and np.any(clip.samples) for clip in clips)


def test_missing_worker_engine_and_worker_timeout_are_audio_errors(monkeypatch):
    tts = tts_module()
    def unavailable(*args, **kwargs):
        raise OSError('engine unavailable')
    monkeypatch.setattr(tts.subprocess, 'run', unavailable)
    assert not tts.tts_available()[0]
    with pytest.raises(AudioError):
        tts.synthesize_chinese('公开测试。')
    def timeout(*args, **kwargs):
        raise tts.subprocess.TimeoutExpired('worker', 60)
    monkeypatch.setattr(tts.subprocess, 'run', timeout)
    with pytest.raises(AudioError, match='超时'):
        tts.synthesize_chinese('公开测试。')


def test_worker_failures_and_bad_payloads_are_not_published(monkeypatch):
    import subprocess
    tts = tts_module()
    for output in [subprocess.CompletedProcess([], 1, b'', b'missing cmn'),
                   subprocess.CompletedProcess([], 0, b'bad', b''),
                   subprocess.CompletedProcess([], 0, b'\x00'*8, b'')]:
        monkeypatch.setattr(tts.subprocess, 'run', lambda *a, _out=output, **k: _out)
        with pytest.raises(AudioError):
            tts.synthesize_chinese('公开测试。')


def test_pcm_resource_limit_is_rejected_before_audio_copy(monkeypatch):
    import struct
    import subprocess
    tts = tts_module()
    monkeypatch.setattr(tts, 'MAX_SECONDS', .001)
    output = subprocess.CompletedProcess([], 0, struct.pack('<I', 22050) + b'\x01\x00'*100, b'')
    monkeypatch.setattr(tts.subprocess, 'run', lambda *a, **k: output)
    with pytest.raises(AudioError, match='上限'):
        tts.synthesize_chinese('公开测试。')


def test_callback_stays_alive_until_native_engine_termination():
    """Native engines retain callback pointers beyond the Python synth call."""
    import ctypes
    import gc
    import weakref
    from voice_effect_lab.audio import tts_worker

    class Function:
        def __init__(self, function):
            self.function = function
        def __call__(self, *args):
            return self.function(*args)

    class Engine:
        def __init__(self):
            self.callback_ref = None
            self.espeak_SetSynthCallback = Function(self.set_callback)
            self.espeak_SetParameter = Function(lambda *args: 0)
            self.espeak_Synth = Function(self.synth)
            self.espeak_Synchronize = Function(lambda: 0)
        def set_callback(self, callback):
            self.callback_ref = weakref.ref(callback)
        def synth(self, *args):
            samples = (ctypes.c_short*2)(100, -100)
            self.callback_ref()(samples, 2, None)
            return 0

    engine = Engine()
    output = tts_worker._synthesize(engine, 22050, '公开测试', 1)
    gc.collect()
    assert engine.callback_ref() is not None, 'Callback pointer freed before espeak_Terminate'
    assert output[-4:] == b'\x64\x00\x9c\xff'


def test_worker_callback_aborts_at_resource_limit():
    import ctypes
    import pytest
    from voice_effect_lab.audio import tts_worker

    class Function:
        def __init__(self, function):
            self.function = function
        def __call__(self, *args):
            return self.function(*args)

    class Engine:
        def __init__(self):
            self.espeak_SetSynthCallback = Function(lambda callback: setattr(self, 'callback', callback))
            self.espeak_SetParameter = Function(lambda *args: 0)
            self.espeak_Synth = Function(self.synth)
            self.espeak_Synchronize = Function(lambda: 0)
        def synth(self, *args):
            samples = (ctypes.c_short*2)(1, 2)
            # Low sample rate makes limit 300; count is rejected before dereference.
            assert self.callback(samples, 301, None) == 1
            return 0

    with pytest.raises(RuntimeError, match='上限'):
        tts_worker._synthesize(Engine(), 1, '公开测试', 1)
