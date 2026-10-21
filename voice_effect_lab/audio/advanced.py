"""可解释的统计混响、环形调制与平滑RMS压缩；不修改调用者数据。"""
import numpy as np
from scipy.signal import lfilter, oaconvolve


def room_reverb(samples, sample_rate, wet=.25, decay=1.2):
    values=np.asarray(samples,dtype=np.float64)
    if wet==0:
        return values.copy()
    length=max(1,int(np.ceil(sample_rate*decay)))
    time=np.arange(length,dtype=np.float64)/sample_rate
    impulse=.12*np.random.default_rng(20261021).standard_normal(length)*np.exp(-np.log(1000)*time/decay)
    predelay=min(length-1,round(.015*sample_rate))
    impulse[:predelay]=0
    for seconds,gain in [(.021,.6),(.037,.4),(.053,.25)]:
        index=round(seconds*sample_rate)
        if index<length:
            impulse[index]+=gain
    norm=np.sqrt(np.dot(impulse,impulse))
    if norm:
        impulse/=norm
    reverberated=oaconvolve(values,impulse,mode='full')
    reverberated[:predelay]=0  # IR前导严格为零，消除FFT在前延迟区的舍入残留。
    reverberated *= wet
    reverberated[:len(values)] += (1-wet)*values
    return reverberated


def ring_modulate(samples, sample_rate, frequency=40, mix=.6):
    values=np.asarray(samples,dtype=np.float64)
    if mix==0:
        return values.copy()
    frequency=min(frequency,.45*sample_rate)
    output=np.empty_like(values)
    for start in range(0,len(values),65536):
        stop=min(start+65536,len(values))
        carrier=np.cos(2*np.pi*frequency*np.arange(start,stop)/sample_rate)
        output[start:stop]=values[start:stop]*((1-mix)+mix*carrier)
    return output


def compress(samples, sample_rate, threshold_db=-18, ratio=3, makeup_db=3):
    values=np.asarray(samples,dtype=np.float64)
    if ratio==1 and makeup_db==0:
        return values.copy()
    alpha=np.exp(-1/(sample_rate*.01))
    power=lfilter([1-alpha],[1,-alpha],values*values)
    level=10*np.log10(np.maximum(power,1e-20))
    gain=10**(-np.maximum(level-threshold_db,0)*(1-1/ratio)/20)
    smoothed,_=lfilter([1-alpha],[1,-alpha],gain,zi=[alpha])
    return values*smoothed*10**(makeup_db/20)
