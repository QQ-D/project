"""DeepFilterNet3 CPU bridge; torch stays in the isolated dbaf process.

The bundled pretrained model processes 48 kHz mono samples. Parent-side
resampling restores the exact original sample rate and count. Each job loads
the model once; long jobs use 10 s cores with 0.25 s context on each side.
"""
from math import gcd
from io import BytesIO
from numbers import Real
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.signal import resample, resample_poly

from .data import AudioClip, AudioError
from .denoise import DenoiseResult


_ROOT = Path(__file__).resolve().parents[2]
_VENDOR = _ROOT/'third_party/DeepFilterNet'
_WORKER = Path(__file__).with_name('deepfilter_worker.py')
_DEFAULT_PYTHON = '/home/rwj/anaconda3/envs/dbaf/bin/python'
_MAX_SAMPLE_BYTES = 128*1024*1024
_RATE = 48000
_TIMEOUT_SECONDS = 600


def _python():
    return Path(os.environ.get('VOICE_LAB_DEEPFILTER_PYTHON', _DEFAULT_PYTHON))


def deepfilter_available() -> tuple[bool, str]:
    """Cheap resource check, without importing torch or launching Python.

    Dependency and integrity errors are reported by the worker at run time.
    VOICE_LAB_DEEPFILTER_PYTHON may select an existing compatible environment.
    """
    interpreter = _python()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        return False, f'DeepFilterNet3 dbaf 解释器不存在或不可执行：{interpreter}'
    for path in (_WORKER, _VENDOR/'DeepFilterNet/df/enhance.py',
                 _VENDOR/'models/DeepFilterNet3/config.ini',
                 _VENDOR/'models/DeepFilterNet3/checkpoints/model_120.ckpt.best'):
        if not path.is_file():
            return False, f'DeepFilterNet3 内置资源缺失：{path.name}'
    return True, 'DeepFilterNet3 · dbaf CPU（2线程）'


def _convert(samples, source_rate, target_rate, count):
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    factor = gcd(source_rate, target_rate)
    up, down = target_rate//factor, source_rate//factor
    # Coprime / very low rates otherwise create enormous polyphase filters.
    value = resample(samples, count) if max(up, down) > 10000 else resample_poly(samples, up, down)
    if len(value) < count:
        value = np.pad(value, (0, count-len(value)))
    return np.asarray(value[:count], dtype=np.float32)


def deepfilter_denoise(clip, noise_clip=None, strength=100) -> DenoiseResult:
    """Pretrained DF3; noise_clip is unused; strength is a 0–100 dry/wet mix."""
    started = perf_counter()
    if not isinstance(clip, AudioClip):
        raise AudioError('请选择有效音频。')
    if clip.duration > 300 or clip.samples.nbytes > _MAX_SAMPLE_BYTES:
        raise AudioError('DeepFilterNet3 支持 300 秒、128 MiB 以内的音频。')
    if (isinstance(strength, bool) or not isinstance(strength, Real)
            or not np.isfinite(strength) or not 0 <= strength <= 100):
        raise AudioError('DeepFilterNet3 降噪强度必须是 0–100 的有限数值。')
    name = Path(clip.name).stem+'_DeepFilterNet3降噪.wav'
    if strength == 0:
        return DenoiseResult(AudioClip(clip.samples, clip.sample_rate, name),
                             perf_counter()-started, 0, 'model', 0, 'deepfilter')
    available, message = deepfilter_available()
    if not available:
        raise AudioError(message)
    count = len(clip.samples)
    model_count = max(1, round(count*_RATE/clip.sample_rate))
    try:
        samples = _convert(clip.samples, clip.sample_rate, _RATE, model_count)
        with TemporaryDirectory(prefix='voice-lab-df3-') as directory:
            source, target = Path(directory)/'input.npy', Path(directory)/'output.npy'
            np.save(source, samples, allow_pickle=False)
            command = [str(_python()), str(_WORKER), str(source), str(target)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       PYTHONDONTWRITEBYTECODE='1')
            # A clean source search path also avoids importing an unrelated df.
            env.pop('PYTHONPATH', None)
            process = subprocess.run(command, cwd=directory, env=env,
                                     capture_output=True, text=True, timeout=_TIMEOUT_SECONDS)
            if process.returncode:
                detail = (process.stderr or process.stdout or '未知错误')[-2000:].strip()
                raise AudioError(f'DeepFilterNet3 dbaf CPU 处理失败：{detail}')
            if not target.is_file():
                raise AudioError('DeepFilterNet3 输出文件缺失或大小无效。')
            with target.open('rb') as output_file:
                payload = output_file.read(samples.nbytes+4097)
            if len(payload) > samples.nbytes+4096:
                raise AudioError('DeepFilterNet3 输出文件大小无效。')
            # A tiny NPY can claim a huge array. Inspect the header before
            # np.load can allocate it, using the same immutable bytes snapshot.
            snapshot = BytesIO(payload)
            version = np.lib.format.read_magic(snapshot)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(snapshot)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(snapshot)
            else:
                raise AudioError('DeepFilterNet3 输出数组格式无效。')
            if (shape != samples.shape or fortran or dtype != np.dtype('float32')
                    or len(payload)-snapshot.tell() != samples.nbytes):
                raise AudioError('DeepFilterNet3 输出数组形状、类型或数据大小无效。')
            snapshot.seek(0)
            restored = np.load(snapshot, allow_pickle=False)
            if not np.isfinite(restored).all():
                raise AudioError('DeepFilterNet3 输出长度、类型或有限数值无效。')
        restored = _convert(restored, _RATE, clip.sample_rate, count)
        wet = float(strength)/100
        output = clip.samples*(1-wet)+restored*wet
        if output.shape != clip.samples.shape or not np.isfinite(output).all():
            raise AudioError('DeepFilterNet3 重采样或混合输出无效。')
        clipped = int(np.count_nonzero(np.abs(output)>1))
        np.clip(output, -1, 1, out=output)
        return DenoiseResult(AudioClip(output, clip.sample_rate, name),
                             perf_counter()-started, clipped, 'model', 0, 'deepfilter')
    except AudioError:
        raise
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f'DeepFilterNet3 dbaf CPU 处理超时（{_TIMEOUT_SECONDS}秒）。') from exc
    except (OSError, ValueError, TypeError, EOFError, OverflowError) as exc:
        raise AudioError(f'DeepFilterNet3 处理失败：{exc}') from exc
