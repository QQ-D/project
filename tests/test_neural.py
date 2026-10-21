"""BP使用公开合成数据；无硬件、下载或私人音频。"""
import importlib
import zipfile

import numpy as np
import pytest

from voice_effect_lab.audio.data import AudioClip, AudioError


def backend():
    return importlib.import_module('voice_effect_lab.audio.neural')


@pytest.fixture(scope='module')
def trained():
    return backend().train_model(epochs=8)


def test_real_training_reduces_heldout_loss_and_reports_each_epoch():
    logs = []
    result = backend().train_model(epochs=8, progress=logs.append)
    assert len(logs) == len(result.train_losses) == len(result.validation_losses) == 8
    assert all(isinstance(line, str) and line for line in logs)
    assert np.isfinite(result.train_losses).all()
    assert result.train_losses[-1] < result.train_losses[0] * .90
    assert result.validation_losses[-1] < result.validation_losses[0]
    assert result.train_samples > 1000 and result.validation_samples > 1000
    assert result.elapsed_seconds > 0 and '合成' in result.model.training_label


def test_model_frozen_and_reproducible(trained):
    again = backend().train_model(epochs=8)
    np.testing.assert_array_equal(trained.train_losses, again.train_losses)
    with pytest.raises(ValueError):
        trained.model.w1[0, 0] = 1
    with pytest.raises(ValueError):
        trained.model.w1.setflags(write=True)


def test_demo_independent_and_actual_output_changes(trained):
    clean, noisy = backend().make_demo_pair()
    clean2, noisy2 = backend().make_demo_pair()
    np.testing.assert_array_equal(noisy.samples, noisy2.samples)
    assert not np.array_equal(clean.samples, noisy.samples)
    out = backend().denoise_neural(noisy, trained.model)
    assert out.clip.sample_rate == noisy.sample_rate
    assert out.clip.samples.shape == noisy.samples.shape
    assert not np.array_equal(out.clip.samples, noisy.samples)
    assert np.isfinite(out.clip.samples).all() and out.elapsed_seconds > 0
    np.testing.assert_array_equal(clean.samples, clean2.samples)


def test_custom_training_uses_disjoint_time_segments():
    # 前80%纯零、后20%谐波：随机混帧会让两组损失相同。
    rate = 8000
    values = np.zeros(rate * 3, np.float32)
    values[int(len(values)*.8):] = .3*np.sin(2*np.pi*370*np.arange(len(values)-int(len(values)*.8))/rate)
    result = backend().train_model(AudioClip(values, rate, '自定义.wav'), epochs=2)
    assert '自定义' in result.model.training_label
    assert result.validation_losses[-1] > 2 * result.train_losses[-1]


@pytest.mark.parametrize('rate,length', [(1,1), (100,7), (8000,511), (44100,9001), (192000,3)])
def test_tiny_low_rate_silence_and_exact_bypass(trained, rate, length):
    zero = AudioClip(np.zeros(length), rate, '静音.wav')
    out = backend().denoise_neural(zero, trained.model)
    np.testing.assert_array_equal(out.clip.samples, zero.samples)
    source = AudioClip(np.full(length,.15), rate, '短.wav')
    out = backend().denoise_neural(source, trained.model)
    assert out.clip.sample_rate == rate and len(out.clip.samples) == length
    assert np.isfinite(out.clip.samples).all()
    bypass = backend().denoise_neural(source, trained.model, strength=0)
    np.testing.assert_array_equal(bypass.clip.samples, source.samples)


def test_save_reload_preserves_exact_inference(trained, tmp_path):
    path = tmp_path / '网络.npz'
    backend().save_model(path, trained.model)
    model = backend().load_model(path)
    assert model.training_label == trained.model.training_label
    _, clip = backend().make_demo_pair(seconds=.1)
    np.testing.assert_array_equal(backend().denoise_neural(clip, model).clip.samples,
                                  backend().denoise_neural(clip, trained.model).clip.samples)


@pytest.mark.parametrize('bad', ['object', 'shape', 'nan', 'version', 'extra', 'huge', 'notzip'])
def test_untrusted_models_fail_as_audio_error(trained, tmp_path, bad):
    path = tmp_path / 'bad.npz'
    backend().save_model(path, trained.model)
    with np.load(path, allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    if bad == 'object': fields['w1'] = np.array([object()], dtype=object)
    if bad == 'shape': fields['w1'] = np.zeros((8, 17), np.float32)
    if bad == 'nan': fields['w1'][0,0] = np.nan
    if bad == 'version': fields['version'] = np.array(999)
    if bad == 'extra': fields['unknown'] = np.zeros(1)
    if bad == 'huge':
        fields['w1'] = np.zeros((1024,1024),np.float32)
        np.savez_compressed(path,**fields)
    elif bad == 'notzip': path.write_bytes(b'not a model')
    else: np.savez_compressed(path, **fields)
    with pytest.raises(AudioError): backend().load_model(path)


@pytest.mark.parametrize('kwargs', [{'epochs':0}, {'epochs':101}, {'epochs':True}, {'epochs':1.5},
    {'batch_size':31}, {'batch_size':513}, {'batch_size':True}])
def test_invalid_training_parameters_fail(kwargs):
    with pytest.raises(AudioError): backend().train_model(**kwargs)


@pytest.mark.parametrize('strength', [-1,101,np.nan,np.inf,True,'75'])
def test_invalid_strength_rejected(trained, strength):
    with pytest.raises(AudioError):
        backend().denoise_neural(AudioClip([.1],8000,'短.wav'),trained.model,strength=strength)


def test_duration_limits_apply_before_processing(trained):
    long = AudioClip(np.zeros(301), 1, '过长.wav')
    with pytest.raises(AudioError): backend().denoise_neural(long, trained.model)
    with pytest.raises(AudioError): backend().train_model(AudioClip(np.zeros(61),1,'过长训练.wav'))
    with pytest.raises(AudioError): backend().train_model(AudioClip(np.zeros(1),8000,'过短训练.wav'))


def test_sample_byte_limit_without_allocating_large_backing(trained):
    source = object.__new__(AudioClip)
    object.__setattr__(source,'samples',np.broadcast_to(np.zeros(1,np.float32),(33554433,)))
    object.__setattr__(source,'sample_rate',200000)
    object.__setattr__(source,'name','超限.wav')
    with pytest.raises(AudioError): backend().denoise_neural(source,trained.model,strength=0)


def test_strength_controls_linear_blend(trained):
    _,clip = backend().make_demo_pair(seconds=.1)
    full = backend().denoise_neural(clip,trained.model,strength=100).clip.samples
    half = backend().denoise_neural(clip,trained.model,strength=50).clip.samples
    np.testing.assert_allclose(half,(clip.samples+full)/2,atol=1e-7)


def test_corrupt_deflate_is_presented_as_audio_error(trained,tmp_path):
    path = tmp_path/'corrupt.npz'
    backend().save_model(path,trained.model)
    with zipfile.ZipFile(path) as archive:
        entry = archive.getinfo('w1.npy')
    data = bytearray(path.read_bytes())
    # Local headers may contain a ZIP64 extra field; read its actual length.
    header = entry.header_offset
    name_length = int.from_bytes(data[header+26:header+28],'little')
    extra_length = int.from_bytes(data[header+28:header+30],'little')
    offset = header+30+name_length+extra_length
    data[offset:offset+8] = b'\xff'*8
    path.write_bytes(data)
    with pytest.raises(AudioError): backend().load_model(path)
