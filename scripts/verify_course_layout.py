"""课程分区与同源四方法比较的真实GUI验收；仅公开合成语音，无音频设备。"""
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('MPLCONFIGDIR', str(root/'.runtime/matplotlib'))
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from voice_effect_lab.ui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    app.setFont(QFont('Noto Sans CJK SC', 10))
    artifacts = root/'artifacts'
    window = MainWindow()
    window.show()

    def wait():
        deadline = time.monotonic()+90
        while window.busy and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(10)
        app.processEvents()
        assert not window.busy and not window.last_error, window.last_error

    def capture(name):
        app.processEvents()
        QTest.qWait(100)
        window.plots.draw()
        if window.intelligent_plots is not None:
            window.intelligent_plots.draw()
        assert window.grab().save(str(artifacts/name))

    def select(key):
        window.neural_selection_combo.setCurrentIndex(window.neural_selection_combo.findData(key))
        return window._neural_selected()[0]

    def reachable(scroll, widget):
        scroll.ensureWidgetVisible(widget, 8, 8)
        app.processEvents()
        point = widget.mapTo(scroll.viewport(), widget.rect().center())
        assert widget.isVisible() and scroll.viewport().rect().contains(point), widget.text()

    try:
        window.load_path(artifacts/'DeepFilterNet3_公开中文加噪输入.wav')
        wait()
        source = window.clip
        assert window.denoise_method_combo.currentData() == 'wiener'
        assert window.denoise_spin.value() == 60
        assert not window.noise_reference_panel.isHidden()
        window.start_denoise()
        wait()
        basic = window.denoised_clip
        capture('课程界面_基础工作台.png')
        window.tabs.setCurrentIndex(1)
        capture('课程界面_高级音效.png')
        window.extension_tabs.setCurrentIndex(1)
        window.tts_text.setPlainText('欢迎来到数字信号处理实验。我们正在测试中文合成和房间混响。')
        capture('课程界面_中文TTS.png')
        window.extension_tabs.setCurrentIndex(2)
        assert window.neural_model is None and window.neural_compare_button.isEnabled()
        window.run_neural_comparison()
        wait()
        df = window.neural_deepfilter_clip
        assert df is not None and window.denoised_clip is basic and window.clip is source
        window.load_neural_model_path(artifacts/'教学BP_公开合成模型.npz')
        wait()
        assert window.neural_deepfilter_clip is df
        window.neural_engine_combo.setCurrentIndex(window.neural_engine_combo.findData('bp'))
        window.run_neural_comparison()
        wait()
        bp = window.neural_clip
        assert bp is not None and window.neural_deepfilter_clip is df
        assert window.clip is source and window.denoised_clip is basic
        np.testing.assert_array_equal(window.neural_traditional_clip.samples, basic.samples)
        capture('课程界面_BP教学训练.png')
        window.neural_engine_combo.setCurrentIndex(window.neural_engine_combo.findData('deepfilter'))
        exports = []
        for key in ('source', 'traditional', 'deepfilter', 'bp'):
            clip = select(key)
            assert window.neural_player._clip is clip
            assert clip.sample_rate == source.sample_rate and len(clip.samples) == len(source.samples)
            _, analysis, _ = window._neural_selected()
            np.testing.assert_array_equal(window.intelligent_plots.axes[0, 1].lines[0].get_ydata(), analysis.wave_values)
            np.testing.assert_array_equal(window.intelligent_plots.axes[2, 1].images[0].get_array(), analysis.spectrogram_db)
            path = artifacts/f'同源方法对比_{key}.wav'
            window.save_neural_to(path)
            wait()
            rate, samples = wavfile.read(path)
            error = float(np.max(np.abs(samples/32768-clip.samples)))
            assert rate == clip.sample_rate and error <= 1/32768
            exports.append({'method': key, 'file': path.name, 'samples': len(samples), 'sample_rate': rate,
                            'pcm16_max_error': error})
        assert all(window.neural_metrics[key+'_snr_db'] is None for key in ('source','traditional','deepfilter','bp'))
        select('deepfilter')
        capture('课程界面_智能降噪对比.png')
        metrics = dict(window.neural_metrics)
        window.neural_volume_spin.setValue(42)
        assert window.listen_volume_spin.value() == window.tts_volume_spin.value() == 42
        assert all(p.volume == 42 for p in (window.player, window.tts_player, window.neural_player))
        window.neural_volume_spin.setValue(100)
        window.resize(1080,720)
        app.processEvents()
        assert (window.width(),window.height()) == (1080,720)
        reachable(window.neural_scroll, window.neural_compare_button)
        reachable(window.neural_scroll, window.neural_send_button)
        capture('课程界面_智能降噪小窗口.png')
        window.neural_engine_combo.setCurrentIndex(window.neural_engine_combo.findData('bp'))
        reachable(window.neural_scroll, window.neural_train_button)
        reachable(window.neural_scroll, window.neural_send_button)
        window.extension_tabs.setCurrentIndex(0)
        reachable(window.advanced_scroll, window.advanced_apply_button)
        window.extension_tabs.setCurrentIndex(1)
        app.processEvents()
        button = window.tts_send_button
        assert button.isVisible() and window.rect().contains(button.mapTo(window, button.rect().bottomRight()))
        window.tabs.setCurrentIndex(0)
        reachable(window.control_scroll, window.noise_button)
        reachable(window.control_scroll, window.import_noise_button)
        window.tabs.setCurrentIndex(1)
        window.extension_tabs.setCurrentIndex(2)
        window.neural_engine_combo.setCurrentIndex(window.neural_engine_combo.findData('deepfilter'))
        window.send_neural_to_workbench()
        assert window.clip is df and window.neural_deepfilter_clip is None and window.neural_clip is None
        report = {'scope':'仅公开自写中文TTS人工加噪；未启动麦克风/扬声器、未读取私人录音',
                  'basic_default':'自动维纳60%，底噪采集/导入/选段入口显示',
                  'extension_tabs':[window.extension_tabs.tabText(i) for i in range(3)],
                  'same_source':{'samples':len(source.samples),'sample_rate':source.sample_rate},
                  'metrics':metrics,'exports':exports,'minimum_window':[1080,720],
                  'checks':['DeepFilterNet3无需BP训练或底噪参考','基础结果在智能比较期间保留',
                            '两个模型缓存同时存在，选择、六图、播放器、PCM16导出一致',
                            '普通导入文件无关联干净参考，四项SNR全部为None',
                            '智能/基础/TTS音量同步；小窗口各子页核心按钮可达',
                            '明确送入才替换原声并清除比较结果']}
        (artifacts/'课程界面验收数据.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        window.close()


if __name__ == '__main__':
    main()
