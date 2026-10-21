"""Offline Mandarin synthesis, isolated from Qt and audio devices.

Only stdin carries text to a local worker. The worker returns raw mono PCM16;
no shell, cloud API, model download, microphone, or playback is involved.
"""
import json
import math
from numbers import Real
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from .data import AudioClip, AudioError

MAX_CHARACTERS = 400
MAX_SECONDS = 300
MAX_BYTES = 128 * 1024 * 1024
_WORKER = Path(__file__).with_name('tts_worker.py')


def _run_worker(payload=None):
    command = [sys.executable, str(_WORKER)]
    if payload is None:
        command.append('--check')
    try:
        result = subprocess.run(command, input=(json.dumps(payload, ensure_ascii=False).encode('utf-8')
                                               if payload is not None else b''),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=60 if payload is not None else 10,
                                check=False)
    except subprocess.TimeoutExpired as exc:
        raise AudioError('离线中文合成超时，请缩短文字后重试。') from exc
    except (OSError, ValueError) as exc:
        raise AudioError(f'无法启动离线中文语音引擎：{exc}') from exc
    if result.returncode:
        detail = result.stderr.decode('utf-8', errors='replace').strip()[-600:]
        raise AudioError('离线中文语音引擎不可用或合成失败：' + (detail or f'worker {result.returncode}'))
    return result.stdout


def tts_available():
    """Return actual local engine/voice availability and a user-facing reason."""
    try:
        _run_worker()
    except AudioError as exc:
        return False, str(exc)
    return True, '离线基础普通话 · eSpeak NG；机器感较强，非自然人声或身份转换。'


def synthesize_chinese(text, speed=1.0):
    """Return a new immutable mono clip; speed is a 0.5–2.0 speech-rate factor."""
    if not isinstance(text, str) or not text.strip():
        raise AudioError('请输入需要合成的中文文字。')
    if len(text) > MAX_CHARACTERS:
        raise AudioError(f'中文合成最多支持 {MAX_CHARACTERS} 个字符。')
    if any(ord(char) < 32 and char not in '\n\r\t' for char in text):
        raise AudioError('合成文字含不支持的控制字符。')
    if (isinstance(speed, bool) or not isinstance(speed, Real)
            or not math.isfinite(speed) or not .5 <= speed <= 2):
        raise AudioError('合成语速必须在 0.5–2.0 倍之间。')
    try:
        text.encode('utf-8')
    except UnicodeError as exc:
        raise AudioError('合成文字含无效的 Unicode 字符。') from exc
    output = _run_worker({'text': text.strip(), 'speed': float(speed)})
    if len(output) < 6 or (len(output)-4) % 2:
        raise AudioError('离线引擎没有返回有效音频。')
    rate = struct.unpack('<I', output[:4])[0]
    count = (len(output)-4) // 2
    if not 8000 <= rate <= 192000:
        raise AudioError('离线引擎返回了无效采样率。')
    if count > rate*MAX_SECONDS or count*4 > MAX_BYTES:
        raise AudioError('合成音频超过 300 秒或 128 MiB 上限，请缩短文字。')
    samples = np.frombuffer(output, dtype='<i2', offset=4).astype(np.float32) / 32768
    if not np.any(samples):
        raise AudioError('离线引擎未生成可用语音，请检查文字内容。')
    return AudioClip(samples, rate, '中文合成.wav')
