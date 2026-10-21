"""GUI和处理链共用的去噪方法分派。"""
from .data import AudioError
from .denoise import spectral_subtract
from .wiener import wiener_denoise

METHOD_NAMES={'wiener':'自适应维纳','spectral':'功率谱减'}


def denoise_audio(clip, noise_clip=None, strength=60, method='wiener'):
    if method=='wiener':
        return wiener_denoise(clip,noise_clip,strength)
    if method=='spectral':
        return spectral_subtract(clip,noise_clip,strength)
    raise AudioError('未知去噪方法，请选择维纳或谱减。')
