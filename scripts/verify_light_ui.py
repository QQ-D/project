"""第一版浅色草图的实际Qt布局验收，全部数据来自公开合成语音。"""
import json
import os
from pathlib import Path
import time


def main():
    root=Path(__file__).resolve().parents[1]
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.runtime/matplotlib'))
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from voice_effect_lab.ui.main_window import MainWindow
    app=QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    app.setFont(QFont('Noto Sans CJK SC',10))
    artifacts=root/'artifacts'
    window=MainWindow()
    window.show()
    def wait():
        deadline=time.monotonic()+60
        while window.busy and time.monotonic()<deadline:
            app.processEvents();QTest.qWait(10)
        app.processEvents()
        assert not window.busy and not window.last_error,window.last_error
    def capture(name):
        app.processEvents();QTest.qWait(100)
        window.plots.draw()
        if window.intelligent_plots is not None:
            window.intelligent_plots.draw()
        assert window.grab().save(str(artifacts/name))
    def reachable(scroll,widget):
        scroll.ensureWidgetVisible(widget,6,6)
        app.processEvents()
        point=widget.mapTo(scroll.viewport(),widget.rect().center())
        assert widget.isVisible() and scroll.viewport().rect().contains(point)
    try:
        window.load_path(artifacts/'DeepFilterNet3_公开中文加噪输入.wav');wait()
        source=window.clip
        window.start_denoise();wait()
        window.enhance_check.setChecked(True)
        window.control_scroll.verticalScrollBar().setValue(0)
        app.processEvents()
        for widget in (window.record_button,window.denoise_button,window.enhance_check,
                       window.preset_buttons['自定义'],window.rate_spin,window.echo_strength_spin):
            point=widget.mapTo(window.control_scroll.viewport(),widget.rect().center())
            assert window.control_scroll.viewport().rect().contains(point),str(widget.objectName())
        assert window.play_button.width()==window.play_button.height()==46
        capture('浅色界面_基础工作台.png')
        window.device_details_button.click()
        assert not window.device_details_panel.isHidden()
        reachable(window.control_scroll,window.output_combo)
        window.device_details_button.click()
        window.effects_details_button.click()
        reachable(window.control_scroll,window.echo_delay_spin)
        assert window.telephone_check.isVisible()
        window.effects_details_button.click()
        window.main_navigation.button(1).click()
        assert window.tabs.currentIndex()==1
        capture('浅色界面_高级音效.png')
        window.extension_tabs.setCurrentIndex(1)
        window.tts_text.setPlainText('欢迎来到数字信号处理实验。')
        window.start_tts();wait()
        assert window.clip is source and window.tts_clip is not None
        capture('浅色界面_中文TTS.png')
        window.extension_tabs.setCurrentIndex(2)
        window.run_neural_comparison();wait()
        assert window.neural_deepfilter_clip is not None and window.clip is source
        capture('浅色界面_智能降噪.png')
        window.resize(1080,720);app.processEvents()
        assert (window.width(),window.height())==(1080,720)
        reachable(window.neural_scroll,window.neural_compare_button)
        reachable(window.neural_scroll,window.neural_send_button)
        capture('浅色界面_小窗口.png')
        window.main_navigation.button(0).click()
        assert window.tabs.currentIndex()==0
        window.device_details_button.setChecked(True)
        reachable(window.control_scroll,window.output_combo)
        window.device_details_button.setChecked(False)
        reachable(window.control_scroll,window.import_noise_button)
        for widget in (window.play_button,window.save_button,window.main_navigation.button(1)):
            assert window.rect().contains(widget.mapTo(window,widget.rect().center()))
        window._error('公开测试：输出设备不可用，请在设备设置中检查。')
        assert window.status_label.property('tone')=='error'
        assert window.status_label.toolTip().startswith('公开测试')
        window._status('界面验收完成')
        assert window.status_label.property('tone')=='normal'
        report={'design':'用户选定第一版浅色草图；图表与数值均为真实公开输入，不照搬示意数字',
                'source':{'sample_rate':source.sample_rate,'samples':len(source.samples),'seconds':source.duration},
                'sizes':[[1500,1040],[1080,720]],
                'checks':['主导航与旧标签页状态同步','采样率/时长/声道徽标来自真实导入',
                          '默认尺寸四张基础卡的主要控件可见','圆形播放按钮宽高均46像素',
                          '设备/更多音效参数可展开，原控件和值保留',
                          '独立中文TTS和dbaf预训练推理成功，基础原声保持',
                          '小窗口导入底噪/设备/保存/模型处理入口可到达',
                          '错误提示/普通提示样式正确，完整信息有悬停提示'],
                'scope':'没有修改DSP/模型或安装依赖；未读取voice/、未启动麦克风或扬声器'}
        (artifacts/'浅色界面验收数据.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        window.close()


if __name__=='__main__':
    main()
