"""小型BP的真实Qt训练、模型重载、独立对比与串联验收。仅用公开合成信号。"""
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from scipy.io import wavfile


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.runtime/matplotlib'))
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from voice_effect_lab.audio.data import save_wav
    from voice_effect_lab.audio.neural import denoise_neural, load_model
    from voice_effect_lab.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    app.setFont(QFont('Noto Sans CJK JP', 10))
    artifacts = root / 'artifacts'
    window = MainWindow()
    window.resize(1500, 980)
    window.show()

    def wait_idle():
        deadline = time.monotonic() + 60
        while window.busy and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(10)
        app.processEvents()
        assert not window.busy and not window.last_error, window.last_error

    def capture(widget, name):
        app.processEvents()
        QTest.qWait(100)
        assert widget.grab().save(str(artifacts / name))

    def select(key):
        window.neural_selection_combo.setCurrentIndex(window.neural_selection_combo.findData(key))
        return window._neural_selected()

    def check_export(path, clip):
        rate, values = wavfile.read(path)
        assert rate == clip.sample_rate and len(values) == len(clip.samples)
        error = float(np.max(np.abs(values / 32768 - clip.samples)))
        assert error <= 1 / 32768
        return {'file': path.name, 'sample_rate': rate, 'samples': len(values),
                'seconds': clip.duration, 'pcm16_max_error': error}

    try:
        window.load_demo()
        wait_idle()
        previous = window.clip
        window.tabs.setCurrentIndex(1)
        window.extension_tabs.setCurrentIndex(2)
        window.neural_engine_combo.setCurrentIndex(window.neural_engine_combo.findData('bp'))
        window.neural_epochs_spin.setValue(20)
        window.neural_batch_spin.setValue(128)
        window.neural_train_button.click()
        wait_idle()
        trained = window.neural_model
        assert trained is not None and window.clip is previous
        losses = re.findall(r'训练损失 ([\d.]+)，验证损失 ([\d.]+)', window.neural_log.toPlainText())
        assert len(losses) == 20
        losses = [tuple(map(float, pair)) for pair in losses]
        assert losses[-1][0] < losses[0][0] and losses[-1][1] < losses[0][1]
        model_path = artifacts / '教学BP_公开合成模型.npz'
        window.save_neural_model_to(model_path)
        wait_idle()
        restored = load_model(model_path)
        for key in ('w1', 'b1', 'w2', 'b2', 'mean', 'scale'):
            np.testing.assert_array_equal(getattr(trained, key), getattr(restored, key))
        window.load_neural_model_path(model_path)
        wait_idle()
        assert window.neural_model is not trained and window.clip is previous
        window.neural_demo_button.click()
        wait_idle()
        source, clean = window.clip, window._neural_demo_clean
        assert source is not previous and clean is not None
        window.neural_compare_button.click()
        wait_idle()
        metrics = dict(window.neural_metrics)
        bp, traditional = window.neural_clip, window.neural_traditional_clip
        assert window.clip is source and window.current_clip is source
        assert bp.sample_rate == source.sample_rate and len(bp.samples) == len(source.samples)
        np.testing.assert_array_equal(bp.samples, denoise_neural(source, restored, 75).clip.samples)
        assert all(metrics[key] is not None for key in ('source_snr_db', 'traditional_snr_db', 'bp_snr_db'))
        exports = []
        clean_path = artifacts / 'BP降噪_公开干净参考.wav'
        save_wav(clean_path, clean)
        exports.append(check_export(clean_path, clean))
        for key, name in [('source', 'BP降噪_公开加噪输入.wav'),
                          ('traditional', 'BP降噪_维纳输出.wav'), ('bp', 'BP降噪_智能输出.wav')]:
            clip, analysis, label = select(key)
            assert window.neural_player._clip is clip
            path = artifacts / name
            window.save_neural_to(path)
            wait_idle()
            exports.append(check_export(path, clip))
        window.neural_scroll.verticalScrollBar().setValue(0)
        capture(window, 'BP降噪_训练面板.png')
        window.neural_scroll.verticalScrollBar().setValue(window.neural_scroll.verticalScrollBar().maximum())
        capture(window, 'BP降噪_比较面板.png')
        window.show_neural_plots()
        app.processEvents()
        np.testing.assert_array_equal(window.neural_plots.axes[0, 0].lines[0].get_ydata(),
                                      window.neural_source_analysis.wave_values)
        np.testing.assert_array_equal(window.neural_plots.axes[0, 1].lines[0].get_ydata(),
                                      window.neural_analysis.wave_values)
        np.testing.assert_array_equal(window.neural_plots.axes[2, 1].images[0].get_array(),
                                      window.neural_analysis.spectrogram_db)
        capture(window.neural_plot_dialog, 'BP降噪_智能六图.png')
        window._neural_close_plots()
        window.resize(1080, 720)
        app.processEvents()
        window.neural_scroll.ensureWidgetVisible(window.neural_train_button, 8, 8)
        app.processEvents()
        point = window.neural_train_button.mapTo(window.neural_scroll.viewport(), window.neural_train_button.rect().center())
        assert window.neural_scroll.viewport().rect().contains(point)
        window.neural_scroll.ensureWidgetVisible(window.neural_send_button, 8, 8)
        app.processEvents()
        point = window.neural_send_button.mapTo(window.neural_scroll.viewport(), window.neural_send_button.rect().center())
        assert window.neural_scroll.viewport().rect().contains(point)
        window.extension_tabs.setCurrentIndex(1)
        app.processEvents()
        button = window.tts_send_button
        point = button.mapTo(window, button.rect().bottomRight())
        assert point.x() < window.width() and point.y() < window.height()
        minimum_size = [window.width(), window.height()]
        window.resize(1500, 980)
        select('bp')
        window.send_neural_to_workbench()
        assert window.clip is bp and window.neural_clip is None
        assert window.neural_model is not None
        window.enhance_check.setChecked(True)
        window.gain_spin.setValue(.9)
        window.reverb_check.setChecked(True)
        window.reverb_decay_spin.setValue(.2)
        window.reverb_wet_spin.setValue(.15)
        window.start_advanced_effects()
        wait_idle()
        processed = window.effected_clip
        assert window.current_clip is processed and window.player._clip is processed
        path = artifacts / 'BP降噪_增强音效串联.wav'
        window.save_to(path)
        wait_idle()
        exports.append(check_export(path, processed))
        window.load_demo()
        wait_idle()
        window.run_neural_comparison()
        wait_idle()
        assert all(window.neural_metrics[key] is None for key in ('source_snr_db', 'traditional_snr_db', 'bp_snr_db'))
        report = {
            'scope': '公开合成信号；未启动麦克风/扬声器，未使用私人录音',
            'model': {'file': model_path.name, 'architecture': '8→tanh16→sigmoid1频谱掩膜',
                      'epochs': 20, 'batch_size': 128, 'training_samples': 24000,
                      'validation_samples': 6000, 'losses': losses,
                      'training_data': trained.training_label},
            'demo': {'seed': 20261022, 'seconds': source.duration, 'metrics': metrics},
            'exports': exports, 'minimum_window_size': minimum_size,
            'checks': ['训练数据与验证/演示信号独立', '模型NPZ重载权重及推理结果相同',
                       '比较保留原声与工作台缓存，三种结果的图形/播放缓存/导出一致',
                       '无干净参考时SNR全部为None且界面显示—',
                       '明确发送BP结果后可继续增强和混响', '最小窗口训练/发送可滚动到达，TTS独立子页按钮完整可见'],
            'limitations': '教学BP仅由合成谐波训练，不保证真实语音泛化或优于维纳。'}
        (artifacts / 'BP降噪验收数据.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        window.close()


if __name__ == '__main__':
    main()
