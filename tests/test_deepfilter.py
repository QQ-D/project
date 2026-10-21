"""DF3 contract: real dbaf inference; synthetic signals only, no user recordings."""
from pathlib import Path
import importlib
import os
import subprocess
import sys

import numpy as np
import pytest

from voice_effect_lab.audio.data import AudioClip, AudioError


def backend():
    return importlib.import_module('voice_effect_lab.audio.deepfilter')


def signal(rate=48000, seconds=.15):
    rng = np.random.default_rng(17)
    t = np.arange(max(1, round(rate*seconds)))/rate
    return AudioClip(.2*np.sin(2*np.pi*170*t)+.025*rng.normal(size=len(t)), rate, '公开合成.wav')


def test_backend_is_available_without_importing_torch():
    module = backend()
    available, message = module.deepfilter_available()
    assert available, message
    result = subprocess.run([sys.executable, '-c',
        'import sys; from voice_effect_lab.audio.deepfilter import deepfilter_available; '
        'assert deepfilter_available()[0]; assert "torch" not in sys.modules'],
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_zero_strength_bypasses_missing_runtime_exactly(monkeypatch):
    module = backend()
    monkeypatch.setenv('VOICE_LAB_DEEPFILTER_PYTHON', '/missing/dbaf/python')
    clip = signal()
    result = module.deepfilter_denoise(clip, strength=0)
    np.testing.assert_array_equal(result.clip.samples, clip.samples)
    assert result.algorithm == 'deepfilter' and result.noise_mode == 'model'
    assert result.estimated_frames == 0 and result.clipped_samples == 0


@pytest.mark.parametrize('rate,count', [(48000, 7200), (16000, 2401), (44100, 3001), (1, 1), (48000, 1)])
def test_real_inference_restores_exact_shape_and_rate(rate, count):
    module = backend()
    source = signal(rate, count/rate)
    before = source.samples.copy()
    result = module.deepfilter_denoise(source)
    assert result.clip.sample_rate == rate
    assert result.clip.samples.shape == source.samples.shape
    assert np.isfinite(result.clip.samples).all()
    assert np.max(np.abs(result.clip.samples)) <= 1
    assert result.elapsed_seconds > 0
    np.testing.assert_array_equal(source.samples, before)


def test_real_inference_matches_direct_external_reference(tmp_path):
    module = backend()
    upstream = Path('/media/rwj/ZX1 2TB/project/DeepFilterNet')
    if not upstream.is_dir():
        pytest.skip('External reference checkout is optional after vendoring')
    source = signal(seconds=.31)
    input_path, output_path = tmp_path/'in.npy', tmp_path/'ref.npy'
    np.save(input_path, source.samples, allow_pickle=False)
    script = '''import sys, numpy as np, torch
torch.set_num_threads(2)
from df.enhance import init_df, enhance
m,s,_,_ = init_df(sys.argv[1], log_file=None, log_level='ERROR')
x=np.load(sys.argv[2],allow_pickle=False)
y=enhance(m,s,torch.from_numpy(x[None].copy()),pad=True)
np.save(sys.argv[3],y[0].numpy(),allow_pickle=False)
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(upstream/'DeepFilterNet'), OMP_NUM_THREADS='2')
    reference = subprocess.run(['/home/rwj/anaconda3/envs/dbaf/bin/python', '-c', script,
        str(upstream/'models/DeepFilterNet3'), str(input_path), str(output_path)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert reference.returncode == 0, reference.stderr
    result = module.deepfilter_denoise(source)
    np.testing.assert_allclose(result.clip.samples, np.clip(np.load(output_path, allow_pickle=False), -1, 1),
                               rtol=1e-5, atol=1e-7)
    assert not np.allclose(result.clip.samples, source.samples)


def test_real_model_preserves_digital_silence():
    result = backend().deepfilter_denoise(AudioClip(np.zeros(997), 48000, 'silence'))
    np.testing.assert_array_equal(result.clip.samples, np.zeros(997, np.float32))


def test_unusual_coprime_sample_rate_has_bounded_resampling():
    source = signal(47999, .025)
    result = backend().deepfilter_denoise(source)
    assert result.clip.sample_rate == 47999
    assert result.clip.samples.shape == source.samples.shape
    assert np.isfinite(result.clip.samples).all()


def test_missing_bundled_model_reports_unavailable(monkeypatch, tmp_path):
    module = backend()
    monkeypatch.setattr(module, '_VENDOR', tmp_path)
    available, message = module.deepfilter_available()
    assert not available
    with pytest.raises(AudioError, match='资源缺失'):
        module.deepfilter_denoise(signal())


@pytest.mark.parametrize('strength', [-1, 101, True, float('nan'), '100'])
def test_invalid_strength_is_audioerror(strength):
    with pytest.raises(AudioError):
        backend().deepfilter_denoise(signal(), strength=strength)


def test_duration_and_memory_limits_precede_worker(monkeypatch):
    module = backend()
    monkeypatch.setenv('VOICE_LAB_DEEPFILTER_PYTHON', '/missing')
    with pytest.raises(AudioError, match='300'):
        module.deepfilter_denoise(AudioClip(np.zeros(301), 1, 'long'))
    clip = signal()
    # Large allocation belongs to the production limit; avoid 128 MiB in tests.
    monkeypatch.setattr(module, '_MAX_SAMPLE_BYTES', 16)
    with pytest.raises(AudioError, match='128'):
        module.deepfilter_denoise(clip)


def test_missing_runtime_is_clear_audioerror(monkeypatch):
    module = backend()
    monkeypatch.setenv('VOICE_LAB_DEEPFILTER_PYTHON', '/missing/dbaf/python')
    assert not module.deepfilter_available()[0]
    with pytest.raises(AudioError, match='dbaf|解释器'):
        module.deepfilter_denoise(signal())


@pytest.mark.parametrize('failure', ['timeout', 'exit', 'missing', 'shape', 'nan', 'object'])
def test_worker_failure_or_bad_output_is_audioerror(monkeypatch, failure):
    module = backend()
    def run(command, **kwargs):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        if failure == 'exit':
            return subprocess.CompletedProcess(command, 1, '', 'missing torch dependency')
        if failure != 'missing':
            value = {'shape': np.zeros(7, np.float32), 'nan': np.full(7200, np.nan, np.float32),
                     'object': np.array([object()], dtype=object)}[failure]
            np.save(command[-1], value)
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(module.subprocess, 'run', run)
    with pytest.raises(AudioError):
        module.deepfilter_denoise(signal())


@pytest.mark.parametrize('shape,dtype', [((2**50,), '<f4'), ((7200,), '<f8'), ((7200,), '|O')])
def test_malformed_npy_header_is_rejected_before_numpy_allocation(monkeypatch, shape, dtype):
    module = backend()
    def run(command, **kwargs):
        with open(command[-1], 'wb') as output:
            np.lib.format.write_array_header_1_0(output, dict(shape=shape, fortran_order=False, descr=dtype))
        return subprocess.CompletedProcess(command, 0, '', '')
    def forbidden_load(*args, **kwargs):
        pytest.fail('Unvalidated output header reached np.load allocation')
    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setattr(module.np, 'load', forbidden_load)
    with pytest.raises(AudioError):
        module.deepfilter_denoise(signal())


def test_truncated_npy_payload_is_rejected_before_allocation(monkeypatch):
    module = backend()
    def run(command, **kwargs):
        with open(command[-1], 'wb') as output:
            np.lib.format.write_array_header_1_0(output, dict(shape=(7200,), fortran_order=False, descr='<f4'))
            output.write(b'partial payload')
        return subprocess.CompletedProcess(command, 0, '', '')
    def forbidden_load(*args, **kwargs):
        pytest.fail('Truncated NPY reached np.load allocation')
    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setattr(module.np, 'load', forbidden_load)
    with pytest.raises(AudioError):
        module.deepfilter_denoise(signal())


def test_wet_mix_and_clipping_are_applied_after_resampling(monkeypatch):
    module = backend()
    def run(command, **kwargs):
        value = np.load(command[-2], allow_pickle=False)
        np.save(command[-1], np.full(value.shape, 3, np.float32), allow_pickle=False)
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(module.subprocess, 'run', run)
    source = signal()
    result = module.deepfilter_denoise(source, strength=25)
    np.testing.assert_allclose(result.clip.samples, source.samples*.75+.75)
    full = module.deepfilter_denoise(source)
    np.testing.assert_array_equal(full.clip.samples, np.ones(7200, np.float32))
    assert full.clipped_samples == 7200


def test_corrupt_bundled_checkpoint_is_rejected_before_torch_import(tmp_path):
    worker = importlib.import_module('voice_effect_lab.audio.deepfilter_worker')
    bad = tmp_path/'model_120.ckpt.best'
    bad.write_bytes(b'invalid pickle')
    with pytest.raises(ValueError, match='SHA256'):
        worker._trusted_checkpoint(bad)


def test_long_audio_keeps_nonzero_tail_and_does_not_leak_segment_state():
    module = backend()
    source = signal(seconds=10.13)
    result = module.deepfilter_denoise(source)
    assert result.clip.samples.shape == source.samples.shape
    assert np.isfinite(result.clip.samples).all()
    assert np.max(np.abs(result.clip.samples[-4800:])) > 1e-5
    # Fresh context state makes the last core equal to a standalone inference
    # over the independently chosen [9.75s, 10.13s] context.
    context = AudioClip(source.samples[468000:], 48000, 'tail-context.wav')
    tail = module.deepfilter_denoise(context)
    np.testing.assert_allclose(result.clip.samples[480000:], tail.clip.samples[12000:],
                               rtol=1e-5, atol=1e-7)
