"""仅用仓库合成音验收五种预设、四阶段实际GUI与PCM16导出。"""
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
        deadline=time.monotonic()+30
        while window.busy and time.monotonic()<deadline:
            app.processEvents();QTest.qWait(10)
        app.processEvents()
        assert not window.busy and not window.last_error,window.last_error

    try:
        window.load_demo();wait_idle();source=window.clip
        presets={}
        for name in ['磁性','娃娃音','大叔音','电话音','山谷回声']:
            window.preset_buttons[name].click()
            window.process_button.click();wait_idle()
            result=window.effected_clip
            assert result is not None and window.current_clip is result
            assert source is window.clip
            assert result.sample_rate==source.sample_rate
            expected=int(np.ceil(len(source.samples)/window.rate_spin.value()))
            if window.echo_strength_spin.value():
                expected+=round(window.echo_delay_spin.value()*source.sample_rate)
            assert len(result.samples)==expected
            presets[name]={'rate':window.rate_spin.value(),'samples':len(result.samples),
                           'duration_seconds':result.duration,
                           'peak':float(np.max(np.abs(result.samples)))}
            window.save_to(artifacts/f'第五阶段_合成{name}.wav');wait_idle()
        window.use_denoise_check.setChecked(True)
        window.rate_spin.setValue(1.25)
        window.process_button.click();wait_idle()
        assert window.denoised_clip is not None and window.enhanced_clip is not None
        assert window._effects_from_denoise and window._effects_from_enhancement
        stages={}
        for stage,label in [(0,'原声'),(1,'去噪'),(2,'增强'),(3,'音效')]:
            window.stages.button(stage).click()
            selected=window.current_clip
            assert window.player._clip is selected
            path=artifacts/f'第五阶段_合成{label}输出.wav'
            window.save_to(path);wait_idle()
            rate,values=wavfile.read(path)
            assert rate==selected.sample_rate
            np.testing.assert_allclose(values/32768,selected.samples,atol=1/32768)
            assert len(values)==len(selected.samples)
            stages[label]={'samples':len(values),'duration_seconds':selected.duration}
        window.control_scroll.ensureWidgetVisible(window.effects_group,10,10)
        window.plots.draw();QTest.qWait(150)
        assert window.grab().save(str(artifacts/'第五阶段实际界面.png'))
        report={'source':'仓库合成测试音，非真人录音','sample_rate':source.sample_rate,
                'presets':presets,'four_stage_exports':stages,
                'final_parameters':{'rate':1.25,'echo_delay':.3,'echo_strength':.45},
                'flow':'五种预设→去噪+增强+音效链路→四阶段试听缓存及导出核对→截图：通过',
                'scope':'未启动麦克风或扬声器；风格参数验收不替代真人语音试听'}
        (artifacts/'第五阶段验收数据.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:window.close()


if __name__=='__main__':main()
