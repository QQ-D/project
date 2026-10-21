"""可解释的教学BP频谱掩膜：8输入 → tanh(16) → sigmoid(1)。

只从含噪谱提取局部谱峰、邻频、相邻帧、频率和帧能量特征。
目标为 clean_power/(clean_power+noise_power)，以软标签交叉熵训练，
NumPy显式反向传播与Adam更新；每轮报告真实训练/独立验证损失。
默认数据是程序生成的谐波与白色/有色噪声，不是通用语音预训练模型。
512点周期Hann、128点hop；最高16kHz内部采样，256帧分批推理。
"""
from dataclasses import dataclass
from io import BytesIO
from math import gcd
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter
import zipfile
import zlib

import numpy as np
from scipy.fft import rfft, irfft
from scipy.ndimage import median_filter
from scipy.signal import resample, resample_poly
from scipy.signal.windows import hann

from .data import AudioClip, AudioError


_RATE = 16000
_SIZE = 512
_HOP = 128
_BATCH_FRAMES = 256
_VERSION = 1
_SHAPES = {'w1': (8, 16), 'b1': (16,), 'w2': (16, 1), 'b2': (1,),
           'mean': (8,), 'scale': (8,)}
_KEYS = set(_SHAPES) | {'version', 'training_label'}
_WINDOW = hann(_SIZE, sym=False).astype(np.float32)
_MAX_FILE = 1024 * 1024
_MAX_UNPACKED = 128 * 1024


def _frozen(value):
    contiguous = np.asarray(value, dtype=np.float32)
    # bytes backing also prevents callers from re-enabling WRITEABLE.
    return np.frombuffer(contiguous.tobytes(), dtype=np.float32).reshape(contiguous.shape)


@dataclass(frozen=True)
class NeuralModel:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    training_label: str

    def __post_init__(self):
        for name, shape in _SHAPES.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
                raise AudioError('BP模型参数形状或有限数值无效。')
            if np.max(np.abs(value)) > 1e6:
                raise AudioError('BP模型参数超出安全范围。')
            if name == 'scale' and np.any(value < 1e-5):
                raise AudioError('BP归一化尺度无效。')
            object.__setattr__(self, name, _frozen(value))
        if (not isinstance(self.training_label, str) or not self.training_label
                or len(self.training_label) > 256):
            raise AudioError('BP模型训练标签无效。')


@dataclass(frozen=True)
class TrainingResult:
    model: NeuralModel
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    elapsed_seconds: float
    train_samples: int
    validation_samples: int


@dataclass(frozen=True)
class NeuralResult:
    clip: AudioClip
    elapsed_seconds: float
    clipped_samples: int


def _check_clip(clip):
    if not isinstance(clip, AudioClip):
        raise AudioError('请选择有效音频。')
    if clip.duration > 300 or clip.samples.nbytes > 128 * 1024 * 1024:
        raise AudioError('BP处理支持 300 秒、128 MiB 以内的音频。')


def _convert(samples, source_rate, target_rate, target_length=None):
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    if target_length is None:
        target_length = max(1, round(len(samples) * target_rate / source_rate))
    factor = gcd(source_rate, target_rate)
    up, down = target_rate // factor, source_rate // factor
    # Avoid pathological polyphase filters for unusual coprime sample rates.
    if max(up, down) > 10000:
        result = resample(samples, target_length)
    else:
        result = resample_poly(samples, up, down)
    if len(result) < target_length:
        result = np.pad(result, (0, target_length-len(result)))
    return np.asarray(result[:target_length], dtype=np.float32)


def _frames(samples):
    padded = np.pad(samples, (_SIZE, _SIZE))
    return np.lib.stride_tricks.sliding_window_view(padded, _SIZE)[::_HOP]


def _spectrum(frames):
    return rfft(frames * _WINDOW, axis=-1)


def _features(power, rate):
    """power includes optional neighboring frames; no reference enters here.

    Inputs: peak/noise, left/noise, right/noise, peak/frame mean,
    physical frequency/8kHz, log frame energy, previous/noise, next/noise.
    Frequency-median noise estimate is local to the observed noisy frame.
    """
    noise = median_filter(power, size=(1, 15), mode='nearest') + 1e-10
    previous = np.concatenate((power[:1], power[:-1]), axis=0)
    following = np.concatenate((power[1:], power[-1:]), axis=0)
    left = np.concatenate((power[:, :2], power[:, :-2]), axis=1)
    right = np.concatenate((power[:, 2:], power[:, -2:]), axis=1)
    energy = power.mean(axis=1, keepdims=True) + 1e-10
    frequency = np.broadcast_to(np.arange(power.shape[1], dtype=np.float32)
                               * (rate / _SIZE / 8000), power.shape)
    log = lambda values: np.clip(np.log(np.maximum(values, 1e-10)), -16, 16)
    return np.stack((log((power+1e-10)/noise), log((left+1e-10)/noise),
                     log((right+1e-10)/noise), log((power+1e-10)/energy),
                     frequency, np.broadcast_to(log(energy), power.shape),
                     log((previous+1e-10)/noise), log((following+1e-10)/noise)),
                    axis=-1).astype(np.float32)


def _harmonics(rng, seconds, rate=_RATE):
    t = np.arange(max(1, round(seconds*rate)), dtype=np.float64)/rate
    base = rng.uniform(95, 330)
    phase = 2*np.pi*(base*t + rng.uniform(-12,12)*t*t/max(seconds, .1))
    sound = np.zeros(len(t))
    for harmonic in range(1, rng.integers(5, 12)):
        sound += np.sin(harmonic*phase+rng.uniform(0, 2*np.pi))/harmonic**1.2
    envelope = (.3+.7*np.sin(np.pi*(rng.uniform(1.5,3.5)*t+.1))**2)
    envelope *= np.minimum(t/.025, 1)*np.minimum((seconds-t)/.025, 1)
    envelope *= np.where(np.sin(2*np.pi*.7*t) < -.7, .02, 1)
    return (sound*envelope*.28/max(np.max(np.abs(sound)), 1e-9)).astype(np.float32)


def _mix(clean, rng, snr_db, colored=False):
    noise = rng.normal(size=len(clean)).astype(np.float32)
    if colored:
        noise = np.convolve(noise, np.array([.2,.6,.2], np.float32), mode='same')
        noise = noise[:len(clean)]
    desired = max(float(np.sqrt(np.mean(clean.astype(np.float64)**2))), .01)
    noise *= desired * 10**(-snr_db/20) / max(float(np.std(noise)), 1e-10)
    noisy = clean + noise
    gain = min(1., .95/max(float(np.max(np.abs(noisy))), 1e-10))
    return clean*gain, noisy*gain


def make_demo_pair(seed=20261022, seconds=3, snr_db=5):
    """独立演示参考；默认seed与训练数据生成器不同。"""
    if (isinstance(seconds, bool) or not isinstance(seconds, Real)
            or not np.isfinite(seconds) or not .01 <= seconds <= 60):
        raise AudioError('演示时长必须为 0.01–60 秒。')
    if (isinstance(snr_db, bool) or not isinstance(snr_db, Real)
            or not np.isfinite(snr_db) or not -20 <= snr_db <= 60):
        raise AudioError('演示信噪比必须为 −20–60 dB。')
    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError) as exc:
        raise AudioError('演示随机种子无效。') from exc
    clean, noisy = _mix(_harmonics(rng, seconds), rng, snr_db)
    return AudioClip(clean, _RATE, '公开合成_干净参考.wav'), AudioClip(noisy, _RATE, '公开合成_加噪演示.wav')


def _dataset(pairs, rng, limit):
    features, targets = [], []
    # Per-pair bounded sampling prevents allocating all frames × bins × 8.
    quota = max(1, limit//len(pairs))
    for clean, noisy in pairs:
        clean_frames, noisy_frames = _frames(clean), _frames(noisy)
        selected = rng.choice(len(noisy_frames)*257,
                              size=min(quota, len(noisy_frames)*257), replace=False)
        selected.sort()
        for start in range(0, len(noisy_frames), _BATCH_FRAMES):
            count = min(_BATCH_FRAMES, len(noisy_frames)-start)
            indices = selected[(selected >= start*257) & (selected < (start+count)*257)]-start*257
            if not len(indices):
                continue
            lo, hi = max(0,start-1), min(len(noisy_frames),start+count+1)
            observed = _spectrum(noisy_frames[lo:hi])
            power = np.abs(observed)**2
            local = _features(power, _RATE)[start-lo:start-lo+count].reshape(-1,8)
            clean_spectrum = _spectrum(clean_frames[start:start+count])
            noisy_spectrum = observed[start-lo:start-lo+count]
            clean_power = np.abs(clean_spectrum)**2
            noise_power = np.abs(noisy_spectrum-clean_spectrum)**2
            target = clean_power/(clean_power+noise_power+1e-10)
            features.append(local[indices])
            targets.append(target.reshape(-1,1)[indices])
    return np.concatenate(features), np.concatenate(targets).astype(np.float32)


def _sigmoid(logits):
    return 1/(1+np.exp(-np.clip(logits, -40, 40)))


def _loss(x, target, parameters):
    w1,b1,w2,b2 = parameters
    total = 0.
    for start in range(0, len(x), 2048):
        logits = np.tanh(x[start:start+2048]@w1+b1)@w2+b2
        # Stable BCE for continuous ideal-ratio labels, not rounded classes.
        total += float(np.sum(np.logaddexp(0,logits)-target[start:start+2048]*logits,
                              dtype=np.float64))
    return total/len(x)


def train_model(clean_clip=None, *, epochs=20, batch_size=128, progress=None):
    started = perf_counter()
    for value, low, high, label in ((epochs,1,100,'训练轮数'),(batch_size,32,512,'批大小')):
        if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
            raise AudioError(f'{label}必须是 {low}–{high} 的整数。')
    if progress is not None and not callable(progress):
        raise AudioError('训练日志回调无效。')
    rng = np.random.default_rng(20261021)
    if clean_clip is None:
        train_pairs = [_mix(_harmonics(rng, 1.5),rng,rng.uniform(-2,15),bool(i%2)) for i in range(8)]
        # New clean waveforms and independent noise; no training-frame reuse.
        valid_pairs = [_mix(_harmonics(rng, 1.5),rng,rng.uniform(-2,15),bool(i%2)) for i in range(3)]
        label = '公开合成谐波/白色与有色噪声（教学BP）'
    else:
        _check_clip(clean_clip)
        if not 2 <= clean_clip.duration <= 60:
            raise AudioError('干净训练音频需要 2–60 秒。')
        split = int(len(clean_clip.samples)*.8)
        # Split raw time first; STFT and independent noise never cross boundary.
        segments = [_convert(part,clean_clip.sample_rate,_RATE)
                    for part in (clean_clip.samples[:split], clean_clip.samples[split:])]
        train_pairs = [_mix(segments[0],rng,snr,bool(i%2)) for i,snr in enumerate((0,5,12))]
        valid_pairs = [_mix(segments[1],rng,snr,bool(i%2)) for i,snr in enumerate((2,8))]
        label = '自定义干净音频（前80%训练/后20%验证）'
    x,y = _dataset(train_pairs,rng,24000)
    vx,vy = _dataset(valid_pairs,rng,6000)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .05)
    x = np.clip((x-mean)/scale,-12,12)
    vx = np.clip((vx-mean)/scale,-12,12)
    parameters = [rng.normal(0,.25,(8,16)).astype(np.float32),np.zeros(16,np.float32),
                  rng.normal(0,.25,(16,1)).astype(np.float32),np.zeros(1,np.float32)]
    moments = [np.zeros_like(p) for p in parameters]
    variances = [np.zeros_like(p) for p in parameters]
    train_losses, validation_losses, step = [], [], 0
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0,len(x),batch_size):
            indices = order[start:start+batch_size]
            batch,target = x[indices], y[indices]
            w1,b1,w2,b2 = parameters
            hidden = np.tanh(batch@w1+b1)
            delta = (_sigmoid(hidden@w2+b2)-target)/len(batch)
            back = (delta@w2.T)*(1-hidden*hidden)
            gradients = (batch.T@back,back.sum(axis=0),hidden.T@delta,delta.sum(axis=0))
            step += 1
            for p,m,v,gradient in zip(parameters,moments,variances,gradients):
                m *= .9; m += .1*gradient
                v *= .999; v += .001*gradient*gradient
                p -= .006*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        train_losses.append(_loss(x,y,parameters))
        validation_losses.append(_loss(vx,vy,parameters))
        if progress:
            progress(f'第 {epoch+1}/{epochs} 轮：训练损失 {train_losses[-1]:.6f}，'
                     f'验证损失 {validation_losses[-1]:.6f}')
    model = NeuralModel(*parameters,mean,scale,label)
    return TrainingResult(model,tuple(train_losses),tuple(validation_losses),
                          perf_counter()-started,len(x),len(vx))


def denoise_neural(clip, model, strength=75):
    started = perf_counter()
    _check_clip(clip)
    if not isinstance(model, NeuralModel):
        raise AudioError('请先训练或加载有效BP模型。')
    if (isinstance(strength,bool) or not isinstance(strength,Real)
            or not np.isfinite(strength) or not 0 <= strength <= 100):
        raise AudioError('BP降噪强度必须是 0–100 的有限数值。')
    name = Path(clip.name).stem + '_BP降噪.wav'
    if strength == 0 or not np.any(clip.samples):
        return NeuralResult(AudioClip(clip.samples,clip.sample_rate,name),perf_counter()-started,0)
    rate = min(_RATE,clip.sample_rate)
    samples = _convert(clip.samples,clip.sample_rate,rate)
    frames = _frames(samples)
    output = np.zeros(len(samples)+2*_SIZE,np.float32)
    for start in range(0,len(frames),_BATCH_FRAMES):
        count = min(_BATCH_FRAMES,len(frames)-start)
        lo,hi = max(0,start-1), min(len(frames),start+count+1)
        observed = _spectrum(frames[lo:hi])
        features = _features(np.abs(observed)**2,rate)[start-lo:start-lo+count]
        x = np.clip((features.reshape(-1,8)-model.mean)/model.scale,-12,12)
        mask = _sigmoid(np.tanh(x@model.w1+model.b1)@model.w2+model.b2).reshape(count,257)
        gain = 1-float(strength)/100*(1-mask)
        restored = irfft(observed[start-lo:start-lo+count]*gain,n=_SIZE,axis=-1)*_WINDOW
        for offset,frame in enumerate(restored):
            first = (start+offset)*_HOP
            output[first:first+_SIZE] += frame
    result = output[_SIZE:_SIZE+len(samples)]/1.5
    result = _convert(result,rate,clip.sample_rate,len(clip.samples))
    clipped = int(np.count_nonzero(np.abs(result)>1))
    np.clip(result,-1,1,out=result)
    return NeuralResult(AudioClip(result,clip.sample_rate,name),perf_counter()-started,clipped)


def save_model(path, model):
    if not isinstance(model, NeuralModel):
        raise AudioError('只能保存本项目的BP模型。')
    fields = {name:getattr(model,name) for name in _SHAPES}
    fields.update(version=np.array(_VERSION,dtype=np.int64),training_label=np.array(model.training_label))
    try:
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream,**fields)
    except (OSError,ValueError,TypeError) as exc:
        raise AudioError(f'无法保存BP模型：{exc}') from exc


def load_model(path):
    """Only bounded numeric project NPZ; reject pickle, unknown keys and versions."""
    try:
        # One bounded immutable snapshot is used for inspection and loading.
        # A path replacement between ZIP validation and np.load cannot bypass it.
        with Path(path).open('rb') as source:
            payload = source.read(_MAX_FILE+1)
        if len(payload) > _MAX_FILE:
            raise AudioError('BP模型压缩文件过大。')
        snapshot = BytesIO(payload)
        with zipfile.ZipFile(snapshot) as archive:
            entries = archive.infolist()
            if (len(entries) != len(_KEYS) or {entry.filename for entry in entries} !=
                    {key+'.npy' for key in _KEYS}):
                raise AudioError('BP模型字段不匹配。')
            if sum(entry.file_size for entry in entries) > _MAX_UNPACKED:
                raise AudioError('BP模型解压大小超限。')
            # Inspect NPY headers before np.load can allocate claimed dimensions.
            for entry in entries:
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1,0):
                        shape,fortran,dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2,0):
                        shape,fortran,dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise AudioError('BP模型数组格式无效。')
                    key = entry.filename[:-4]
                    expected = _SHAPES.get(key,())
                    if shape != expected or dtype.hasobject or fortran:
                        raise AudioError('BP模型数组形状或类型无效。')
                    if key in _SHAPES and dtype != np.dtype('float32'):
                        raise AudioError('BP模型参数类型无效。')
                    if key == 'version' and dtype != np.dtype('int64'):
                        raise AudioError('BP模型版本类型无效。')
                    if key == 'training_label' and (dtype.kind != 'U' or dtype.itemsize > 1024):
                        raise AudioError('BP模型标签类型无效。')
                    expected_bytes = int(np.prod(shape,dtype=np.int64))*dtype.itemsize
                    if expected_bytes != entry.file_size-stream.tell():
                        raise AudioError('BP模型数组大小无效。')
        snapshot.seek(0)
        with np.load(snapshot,allow_pickle=False) as data:
            if int(data['version']) != _VERSION:
                raise AudioError('不支持此BP模型版本。')
            return NeuralModel(**{name:data[name] for name in _SHAPES},
                               training_label=str(data['training_label'].item()))
    except AudioError:
        raise
    except (OSError,ValueError,TypeError,EOFError,KeyError,OverflowError,
            zipfile.BadZipFile,zlib.error,RuntimeError,NotImplementedError) as exc:
        raise AudioError(f'无法加载BP模型：{exc}') from exc
