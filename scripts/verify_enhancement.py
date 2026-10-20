"""真实GUI增强/去噪链路与三阶段导出验收；只使用仓库合成样本。

QT_QPA_PLATFORM=offscreen .venv-run/bin/python -m scripts.verify_enhancement
"""
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
    app=QApplication.instance() or QApplication([])
    app.setStyle('Fusion');app.setFont(QFont('Noto Sans CJK JP',10))
    window=MainWindow();window.show()
    artifacts=root/'artifacts';artifacts.mkdir(exist_ok=True)

    def wait_idle():
        deadline=time.monotonic()+20
        while window.busy and time.monotonic()<deadline:
            app.processEvents();QTest.qWait(10)
        app.processEvents()
        assert not window.busy and not window.last_error,window.last_error

    try:
        window.load_demo();wait_idle()
        source=window.clip
        window.enhance_check.setChecked(True)
        window.gain_spin.setValue(.75)
        window.clarity_spin.setValue(40)
        window.eq_low_spin.setValue(-2)
        window.eq_mid_spin.setValue(1)
        window.eq_high_spin.setValue(-1)
        window.enhance_button.click();wait_idle()
        assert window.denoised_clip is None
        assert window.current_clip is window.enhanced_clip
        raw_peak=float(np.max(np.abs(window.enhanced_clip.samples)))
        assert abs(raw_peak-.7125)<1e-6
        window.use_denoise_check.setChecked(True)
        window.process_button.click();wait_idle()
        assert window.clip is source
        assert window.denoised_clip is not None
        assert window.enhanced_clip is not None
        assert window._enhanced_from_denoise
        assert abs(float(np.max(np.abs(window.enhanced_clip.samples)))-.7125)<1e-6
        peaks={}
        for stage,label in [(0,'原声'),(1,'去噪'),(2,'增强')]:
            window.stages.button(stage).click()
            selected=window.current_clip
            assert window.player._clip is selected
            assert selected.sample_rate==source.sample_rate
            assert len(selected.samples)==len(source.samples)
            path=artifacts/f'第四阶段_合成{label}输出.wav'
            window.save_to(path);wait_idle()
            rate,values=wavfile.read(path)
            assert rate==selected.sample_rate
            np.testing.assert_allclose(values/32768,selected.samples,atol=1/32768)
            peaks[label]=float(np.max(np.abs(selected.samples)))
        window.control_scroll.ensureWidgetVisible(window.enhance_group,10,10)
        window.plots.draw();QTest.qWait(150)
        assert window.grab().save(str(artifacts/'第四阶段实际界面.png'))
        report={'source':'仓库合成测试音，非真人录音','sample_rate':source.sample_rate,
                'duration_seconds':source.duration,'peak_by_stage':peaks,
                'normalization_target':.95,'gain':.75,'expected_output_peak':.7125,
                'clarity_percent':40,'extra_eq_db':{'low':-2,'mid':1,'high':-1},
                'raw_input_enhancement_peak':raw_peak,
                'flow':'原声增强→勾选去噪输入→生成去噪+增强→三阶段切换及导出核对→截图：通过',
                'scope':'未启动麦克风或扬声器；不将响度/EQ变化当作语音质量提升证明'}
        (artifacts/'第四阶段验收数据.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:window.close()


if __name__=='__main__':main()
