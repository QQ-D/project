"""合成样本的两方法对照、真实Qt工作流与导出核对；无硬件采集/播放。"""
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.io import wavfile


def main():
    root=Path(__file__).resolve().parents[1]
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.runtime'/'matplotlib'))
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from voice_effect_lab.audio.data import load_wav
    from voice_effect_lab.audio.denoising import denoise_audio
    from voice_effect_lab.ui.main_window import MainWindow
    app=QApplication.instance() or QApplication([])
    app.setStyle('Fusion');app.setFont(QFont('Noto Sans CJK JP',10))
    artifacts=root/'artifacts'
    window=MainWindow();window.show()

    def wait_idle():
        deadline=time.monotonic()+30
        while window.busy and time.monotonic()<deadline:
            app.processEvents();QTest.qWait(10)
        app.processEvents()
        assert not window.busy and not window.last_error,window.last_error

    try:
        window.load_path(root/'samples/去噪测试_人工加噪.wav');wait_idle()
        source=window.clip
        reference=load_wav(root/'samples/去噪测试_干净参考.wav').samples.astype(float)
        def snr(samples):
            return float(10*np.log10(np.sum(reference**2)/np.sum((samples-reference)**2)))
        report={'source':'仓库谐波+白噪声合成样本，非真人录音','input_snr_db':snr(source.samples),'methods':{}}
        assert window.denoise_method_combo.currentData()=='wiener'
        for method,index in [('wiener',0),('spectral',1)]:
            window.denoise_method_combo.setCurrentIndex(index)
            assert window.denoised_clip is None
            window.denoise_button.click();wait_idle()
            selected=window.current_clip
            expected=denoise_audio(source,None,60,method)
            np.testing.assert_array_equal(selected.samples,expected.clip.samples)
            assert window.player._clip is selected and window.clip is source
            path=artifacts/f'维纳验收_合成{method}输出.wav'
            window.save_to(path);wait_idle()
            rate,values=wavfile.read(path)
            assert rate==source.sample_rate and len(values)==len(source.samples)
            np.testing.assert_allclose(values/32768,selected.samples,atol=1/32768)
            report['methods'][method]={'reference_snr_db':snr(selected.samples),'seconds':expected.elapsed_seconds}
        assert report['methods']['wiener']['reference_snr_db']>report['methods']['spectral']['reference_snr_db']
        window.denoise_method_combo.setCurrentIndex(0)
        window.use_denoise_check.setChecked(True)
        window.enhance_check.setChecked(True)
        window.gain_spin.setValue(.8)
        window.effects_check.setChecked(True)
        window.rate_spin.setValue(1.1)
        window.process_button.click();wait_idle()
        np.testing.assert_array_equal(window.denoised_clip.samples,denoise_audio(source).clip.samples)
        assert window.enhanced_clip is not None and window.effected_clip is not None
        window.stages.button(1).click()
        window.control_scroll.ensureWidgetVisible(window.auto_noise_check,10,10)
        window.plots.draw();QTest.qWait(150)
        assert window.grab().save(str(artifacts/'维纳去噪实际界面.png'))
        report['gui_flow']='默认维纳→谱减切换失效→两方法导出核对→维纳串联增强/音效→切回去噪截图：通过'
        report['scope']='无麦克风/扬声器启动；不将合成SNR当作真人听感'
        (artifacts/'维纳降噪验收数据.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:window.close()


if __name__=='__main__':main()
