"""Private isolated eSpeak NG worker. stdout: little-endian rate + mono PCM16.

AUDIO_OUTPUT_SYNCHRONOUS sends samples solely to the callback. All C objects,
text buffers and callbacks remain alive until synthesis and termination finish.
A native failure therefore cannot corrupt the GUI process or another request.
"""
import ctypes as ct
import ctypes.util
import json
import math
import struct
import sys


MAX_SAMPLES_BYTES = 128 * 1024 * 1024


def _engine():
    name = ctypes.util.find_library('espeak-ng')
    if not name:
        raise RuntimeError('未找到 libespeak-ng，请安装系统 eSpeak NG 及 cmn 普通话语音数据。')
    lib = ct.CDLL(name)
    lib.espeak_Initialize.argtypes = [ct.c_int, ct.c_int, ct.c_char_p, ct.c_int]
    lib.espeak_Initialize.restype = ct.c_int
    lib.espeak_SetVoiceByName.argtypes = [ct.c_char_p]
    lib.espeak_SetVoiceByName.restype = ct.c_int
    lib.espeak_SetParameter.argtypes = [ct.c_int, ct.c_int, ct.c_int]
    lib.espeak_SetParameter.restype = ct.c_int
    lib.espeak_Synth.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_uint, ct.c_int,
                                ct.c_uint, ct.c_uint, ct.c_void_p, ct.c_void_p]
    lib.espeak_Synth.restype = ct.c_int
    lib.espeak_Synchronize.argtypes = []
    lib.espeak_Synchronize.restype = ct.c_int
    lib.espeak_Terminate.argtypes = []
    lib.espeak_Terminate.restype = ct.c_int
    # 2 = AUDIO_OUTPUT_SYNCHRONOUS, 0x8000 = DONT_EXIT on missing data.
    rate = lib.espeak_Initialize(2, 40, None, 0x8000)
    if rate <= 0:
        raise RuntimeError('eSpeak NG 初始化失败，缺少本地语音数据。')
    if lib.espeak_SetVoiceByName(b'cmn'):
        lib.espeak_Terminate()
        raise RuntimeError('未找到 cmn 普通话语音数据。')
    return lib, rate


def _synthesize(lib, rate, text, speed):
    chunks = []
    total = 0
    callback_error = []
    limit = min(rate*300, MAX_SAMPLES_BYTES//4)
    callback_type = ct.CFUNCTYPE(ct.c_int, ct.POINTER(ct.c_short), ct.c_int, ct.c_void_p)

    @callback_type
    def callback(samples, count, events):
        nonlocal total
        try:
            if count <= 0 or not samples:
                return 0
            if total+count > limit:
                callback_error.append('合成音频超过 300 秒或 128 MiB 上限。')
                return 1
            chunks.append(ct.string_at(samples, count*2))
            total += count
            return 0
        except BaseException:
            callback_error.append('无法接收离线合成音频。')
            return 1

    lib.espeak_SetSynthCallback.argtypes = [callback_type]
    lib.espeak_SetSynthCallback.restype = None
    # C stores only a raw pointer. Retain the trampoline on the library object
    # through the caller's final espeak_Terminate(), including error paths.
    lib._tts_callback = callback
    lib.espeak_SetSynthCallback(callback)
    # espeakRATE=1, nominal 175 words/min. Speech-rate only; no resampling.
    if lib.espeak_SetParameter(1, round(175*speed), 0):
        raise RuntimeError('无法设置离线语速。')
    buffer = ct.create_string_buffer(text.encode('utf-8'))
    # POS_CHARACTER=1; espeakCHARS_UTF8=1 and espeakENDPAUSE=0x1000.
    code = lib.espeak_Synth(buffer, len(buffer), 0, 1, 0, 1 | 0x1000, None, None)
    synced = lib.espeak_Synchronize()
    if callback_error:
        raise RuntimeError(callback_error[0])
    if code or synced or not total:
        raise RuntimeError('eSpeak NG 未能生成普通话音频。')
    return struct.pack('<I', rate) + b''.join(chunks)


def main():
    lib = None
    try:
        checking = len(sys.argv) == 2 and sys.argv[1] == '--check'
        if not checking:
            payload = json.loads(sys.stdin.buffer.read(8192))
            text, speed = payload['text'], payload['speed']
            if (not isinstance(text, str) or not text.strip() or len(text) > 400
                    or not isinstance(speed, (int, float)) or isinstance(speed, bool)
                    or not math.isfinite(speed) or not .5 <= speed <= 2
                    or any(ord(c) < 32 and c not in '\n\r\t' for c in text)):
                raise RuntimeError('离线合成参数无效。')
        lib, rate = _engine()
        if not checking:
            sys.stdout.buffer.write(_synthesize(lib, rate, text, speed))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if lib is not None:
            lib.espeak_Terminate()


if __name__ == '__main__':
    sys.exit(main())
