"""Standalone dbaf worker: only the fixed bundled model is trusted.

Run by deepfilter.py, never import torch in the GUI process. A job initializes
one model; each bounded segment uses a fresh DF analysis/synthesis state.
"""
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import sys


_VENDOR = Path(__file__).resolve().parents[2]/'third_party/DeepFilterNet'
_MODEL = _VENDOR/'models/DeepFilterNet3'
_CHECKPOINT = _MODEL/'checkpoints/model_120.ckpt.best'
_CHECKPOINT_SHA256 = '23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003'
_CONFIG_SHA256 = '415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290'
_CORE = 480000
_CONTEXT = 12000


def _trusted_checkpoint(path):
    """Snapshot bounded trusted bytes before importing torch or unpickling."""
    with Path(path).open('rb') as source:
        payload = source.read(8714073+1)
    if len(payload) != 8714073 or sha256(payload).hexdigest() != _CHECKPOINT_SHA256:
        raise ValueError('DeepFilterNet3 checkpoint SHA256 完整性校验失败。')
    return payload


def main(input_path, output_path):
    payload = _trusted_checkpoint(_CHECKPOINT)
    if sha256((_MODEL/'config.ini').read_bytes()).hexdigest() != _CONFIG_SHA256:
        raise ValueError('DeepFilterNet3 config SHA256 完整性校验失败。')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(_VENDOR/'DeepFilterNet'))
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    # Upstream reads a pickled checkpoint. Supply the exact verified snapshot
    # (rather than reopening the path) to close the hash/load replacement gap.
    original_load = torch.load
    def trusted_load(path, *args, **kwargs):
        if not isinstance(path, (str, os.PathLike)) or Path(path).resolve() != _CHECKPOINT.resolve():
            raise ValueError('Only the bundled DeepFilterNet3 checkpoint is allowed')
        return original_load(BytesIO(payload), *args, **kwargs)
    torch.load = trusted_load
    from df.enhance import enhance, init_df
    from df.model import ModelParams
    from libdf import DF
    samples = np.load(input_path, allow_pickle=False)
    if (samples.ndim != 1 or not 0 < samples.size <= 14400000
            or samples.dtype != np.dtype('float32') or not np.isfinite(samples).all()):
        raise ValueError('Invalid bounded mono float32 input')
    model, state, _, _ = init_df(str(_MODEL), log_file=None, log_level='ERROR')
    p = ModelParams()
    output = np.empty_like(samples)
    for first in range(0, len(samples), _CORE):
        end = min(first+_CORE, len(samples))
        low, high = max(0, first-_CONTEXT), min(len(samples), end+_CONTEXT)
        # Reusing libdf analysis/synthesis buffers carries previous audio into
        # the next overlapping context. Create clean state, reuse model only.
        if first:
            state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size,
                       nb_bands=p.nb_erb, min_nb_erb_freqs=p.min_nb_freqs)
        enhanced = enhance(model, state, torch.from_numpy(samples[None, low:high].copy()), pad=True)
        value = enhanced[0].cpu().numpy()
        if value.shape != (high-low,) or not np.isfinite(value).all():
            raise ValueError('Invalid model output')
        output[first:end] = value[first-low:end-low]
    np.save(output_path, output, allow_pickle=False)


if __name__ == '__main__':
    try:
        if len(sys.argv) != 3:
            raise ValueError('Expected input.npy and output.npy')
        main(sys.argv[1], sys.argv[2])
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
