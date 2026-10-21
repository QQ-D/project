"""真实Qt扩展页多重回声与工作台导出验收，只使用仓库合成样本。"""
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
    from voice_effect_lab.ui.main_window import MainWindow
    from voice_effect_lab.audio.effects import apply_effects
    app=QApplication.instance() or QApplication([])
    app.setStyle('Fusion');app.setFont(QFont('Noto Sans CJK JP',10))
    artifacts=root/'artifacts';artifacts.mkdir(exist_ok=True)
    window=MainWindow();window.show()

    def wait_idle():
        deadline=time.monotonic()+30
        while window.busy and time.monotonic()<deadline:
            app.processEvents();QTest.qWait(10)
        app.processEvents()
        assert not window.busy and not window.last_error,window.last_error

    try:
        window.load_demo();wait_idle();source=window.clip
        window.tabs.setCurrentIndex(1)
        window.multiecho_check.setChecked(True)
        window.multiecho_count_spin.setValue(4)
        window.multiecho_delay_spin.setValue(.25)
        window.multiecho_strength_spin.setValue(.4)
        QTest.qWait(100)
        assert window.grab().save(str(artifacts/'多重回声扩展页.png'))
        window.multiecho_button.click();wait_idle()
        assert window.tabs.currentIndex()==0
        result=window.effected_clip
        assert result is window.current_clip and window.player._clip is result
        assert window.clip is source
        assert result.sample_rate==source.sample_rate
        assert len(result.samples)==len(source.samples)+4*round(.25*source.sample_rate)
        np.testing.assert_array_equal(result.samples,apply_effects(source,window._effect_settings()).clip.samples)
        path=artifacts/'多重回声_合成输出.wav'
        window.save_to(path);wait_idle()
        rate,values=wavfile.read(path)
        assert rate==source.sample_rate and len(values)==len(result.samples)
        np.testing.assert_allclose(values/32768,result.samples,atol=1/32768)
        window.plots.draw();QTest.qWait(100)
        assert window.grab().save(str(artifacts/'多重回声工作台.png'))
        window.multiecho_check.setChecked(False)
        assert window.effected_clip is None
        window.start_effects();wait_idle()
        assert len(window.effected_clip.samples)==len(source.samples)+round(.25*source.sample_rate)
        report={'source':'仓库合成测试音，非真人录音','sample_rate':source.sample_rate,
                'source_samples':len(source.samples),'source_seconds':source.duration,
                'echo_count':4,'delay_seconds':.25,'strength_and_decay':.4,
                'last_echo_gain':.4**4,'output_samples':len(result.samples),'output_seconds':result.duration,
                'single_echo_seconds':window.effected_clip.duration,
                'flow':'扩展页共享参数→应用返回工作台→图/播放缓存/导出一致→关闭恢复单次：通过',
                'scope':'未启动麦克风或扬声器，全部产物为合成音'}
        (artifacts/'多重回声验收数据.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:window.close()


if __name__=='__main__':main()
