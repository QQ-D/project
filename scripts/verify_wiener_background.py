"""维纳旧版/新版36组公开中文背景对照；需要含基线提交的Git副本和本机eSpeak NG。"""
import hashlib
import json
from pathlib import Path
import subprocess
import types
import numpy as np
import time
from scipy.signal import butter, sosfilt, resample_poly
from voice_effect_lab.audio.data import AudioClip, save_wav
from voice_effect_lab.audio.tts import synthesize_chinese
from voice_effect_lab.audio.wiener import wiener_denoise

BASELINE='ed299b371b3fba9d7c6988b02daec2e7dfc36021'


def main():
    project=Path(__file__).resolve().parents[1]
    old_source=subprocess.check_output(['git','show',BASELINE+':voice_effect_lab/audio/wiener.py'],cwd=project)
    baseline=types.ModuleType('voice_effect_lab.audio._baseline_wiener')
    baseline.__package__='voice_effect_lab.audio'
    exec(compile(old_source,BASELINE,'exec'),baseline.__dict__)
    root=Path(__file__).resolve().parents[1]/'artifacts';rows=[];rate=16000
    texts=['今天我们测试风扇背景噪声。请注意听人声是否清楚，停顿时是否还存在嗡嗡声。', '打开窗户以后，房间里传来持续的气流声。声音处理应该减少背景，同时保留说话的细节。']
    for text_id,text in enumerate(texts):
        clip=synthesize_chinese(text)
        x=resample_poly(clip.samples,rate,clip.sample_rate).astype(float)*.8
        # No artificial leading pause: estimator must not assume the first 0.5 s is noise.
        t=np.arange(len(x))/rate
        speech=np.convolve(x*x,np.ones(320)/320,'same')>1e-5
        pause=np.convolve(x*x,np.ones(640)/640,'same')<1e-9
        base_values=x.copy()
        for seed in [42031,90557]:
          for kind in ['hiss','fan','varying']:
           for input_snr in [0,5,10]:
            x=base_values.copy()
            rng=np.random.default_rng(seed);n=rng.normal(size=len(x))
            if kind=='hiss':n=sosfilt(butter(2,600,fs=rate,btype='high',output='sos'),n)
            if kind=='fan':n=sosfilt(butter(3,1100,fs=rate,output='sos'),n)+.7*np.sin(2*np.pi*93*t)+.25*np.sin(2*np.pi*186*t)
            if kind=='varying':n*=np.where(t<len(t)/rate*.55,.6,1.8)
            n*=np.sqrt(np.mean(x*x)/np.mean(n*n)/10**(input_snr/10))
            common_scale=min(1.,.98/max(np.max(np.abs(x+n)),1e-20))
            x=x*common_scale;n=n*common_scale
            source=AudioClip(x+n,rate,'独立中文_'+kind+'.wav')
            for name,fn in [('current',baseline.wiener_denoise),('candidate',wiener_denoise)]:
                start=time.perf_counter();result=fn(source);y=result.clip.samples.astype(float);err=y-x
                a=x-x.mean();b=y-y.mean();projection=a*np.dot(a,b)/np.dot(a,a)
                row={'text_id':text_id,'seed':seed,'noise':kind,'input_snr_db':input_snr,'version':name,
                     'output_snr_db':float(10*np.log10(np.dot(x,x)/np.dot(err,err))),
                     'si_sdr_db':float(10*np.log10(np.dot(projection,projection)/np.dot(b-projection,b-projection))),
                     'speech_projection':float(np.dot(y[speech],x[speech])/np.dot(x[speech],x[speech])),
                     'pause_attenuation_db':float(10*np.log10(np.dot(n[pause],n[pause])/max(np.dot(y[pause],y[pause]),1e-30))) if pause.any() else None,
                     'seconds':time.perf_counter()-start}
                assert len(y)==len(x) and result.clip.sample_rate==rate and np.isfinite(y).all()
                rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
                if text_id==0 and seed==42031 and input_snr==5:
                    save_wav(root/f'维纳优化_{kind}_{name}.wav',result.clip)
                    if name=='current':
                        save_wav(root/f'维纳优化_{kind}_input.wav',source)
                        save_wav(root/f'维纳优化_{kind}_clean.wav',AudioClip(x,rate,'干净参考.wav'))

    pairs={}
    for row in rows:
        key=(row['text_id'],row['seed'],row['noise'],row['input_snr_db'])
        pairs.setdefault(key,{})[row['version']]=row
    summary={}
    for kind in ['hiss','fan','varying']:
        cases=[v for k,v in pairs.items() if k[2]==kind]
        entry={}
        for metric in ['output_snr_db','si_sdr_db','pause_attenuation_db','speech_projection','seconds']:
            before=[v['current'][metric] for v in cases]
            after=[v['candidate'][metric] for v in cases]
            entry[metric]={'before_mean':float(np.mean(before)),'after_mean':float(np.mean(after)),
                           'worst_delta':float(np.min(np.array(after)-before)),
                           'best_delta':float(np.max(np.array(after)-before))}
        summary[kind]=entry
    report={'baseline_commit':BASELINE,'baseline_sha256':hashlib.sha256(old_source).hexdigest(),
            'data':'2段公开自写离线中文TTS，2种种子，3类噪声，输入0/5/10dB，共36组；未假设开头为底噪。',
            'evaluation':'clean只用于测量；输出不归一化。防溢出时输入/噪声/参考同倍率缩放。',
            'method':'本项目维纳改进；借鉴开源思路，不是完整WebRTC/IMCRA移植。',
            'scope':'不访问voice/，不启动麦克风/扬声器，不安装依赖；SNR不代表真人听感或可懂度。',
            'summary':summary,'cases':rows}
    (root/'维纳背景优化验收数据.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
