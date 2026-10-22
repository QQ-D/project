"""离线中文合成与高级音效的真实Qt验收，仅使用公开自写文字。"""
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.runtime' / 'matplotlib'))
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from voice_effect_lab.audio.effects import apply_effects
    from voice_effect_lab.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    app.setFont(QFont('Noto Sans CJK JP', 10))
    artifacts = root / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    window = MainWindow()
    window.resize(1500, 980)
    window.show()

    def wait_idle():
        deadline = time.monotonic() + 90
        while window.busy and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(10)
        app.processEvents()
        assert not window.busy and not window.last_error, window.last_error

    def screenshot(name):
        app.processEvents()
        QTest.qWait(100)
        assert window.grab().save(str(artifacts / name))

    def export(name, clip, tts=False):
        path = artifacts / name
        (window.save_tts_to if tts else window.save_to)(path)
        wait_idle()
        rate, values = wavfile.read(path)
        assert rate == clip.sample_rate and len(values) == len(clip.samples)
        error = float(np.max(np.abs(values / 32768 - clip.samples)))
        assert error <= 1 / 32768
        return {'file': name, 'sample_rate': rate, 'samples': len(values),
                'seconds': clip.duration, 'peak': float(np.max(np.abs(clip.samples))),
                'pcm16_max_error': error}

    def apply():
        window.start_advanced_effects()
        wait_idle()
        result = window.effected_clip
        assert result is window.current_clip and window.player._clip is result
        assert window.clip is generated
        np.testing.assert_array_equal(
            result.samples, apply_effects(generated, window._effect_settings()).clip.samples)
        return result

    try:
        window.load_demo()
        wait_idle()
        original = window.clip
        text = '欢迎来到数字信号处理实验。我们正在测试中文合成和房间混响。'
        window.tabs.setCurrentIndex(1)
        window.extension_tabs.setCurrentIndex(1)
        window.tts_text.setPlainText(text)
        window.tts_generate_button.click()
        wait_idle()
        generated = window.tts_clip
        assert generated is not None and np.any(generated.samples)
        assert window.tts_progress_label.text() == f'0.00 / {window.tts_player.duration_ms / 1000:.2f} s'
        assert window.clip is original
        tts_record = export('中文TTS_公开测试句.wav', generated, tts=True)
        assert window.clip is original
        window.tts_volume_spin.setValue(85)
        assert window.player.volume == window.tts_player.volume == 85
        assert window.listen_volume_spin.value() == 85
        screenshot('高级音效与中文TTS_扩展页.png')

        window.tts_send_button.click()
        assert window.clip is generated and window.current_clip is generated
        assert window.denoised_clip is window.enhanced_clip is window.effected_clip is None
        window.independent_check.setChecked(True)
        window.pitch_spin.setValue(2)
        window.tempo_spin.setValue(1.15)
        window.reverb_check.setChecked(True)
        window.reverb_wet_spin.setValue(.25)
        window.reverb_decay_spin.setValue(.6)
        window.compression_check.setChecked(True)
        processed = apply()
        assert len(processed.samples) == round(len(generated.samples) / 1.15) + int(np.ceil(.6 * generated.sample_rate)) - 1
        advanced_record = export('高级音效_中文合成加工.wav', processed)
        window.control_scroll.ensureWidgetVisible(window.effects_group, 10, 10)
        window.plots.draw()
        screenshot('高级音效与中文TTS_工作台.png')

        window.reset_processing()
        window.modulation_check.setChecked(True)
        window.modulation_frequency_spin.setValue(40)
        window.modulation_mix_spin.setValue(.8)
        mechanical = apply()
        assert len(mechanical.samples) == len(generated.samples)
        assert not np.array_equal(mechanical.samples, generated.samples)
        mechanical_record = export('机械电音_中文合成.wav', mechanical)

        window.reset_processing()
        window.compression_check.setChecked(True)
        window.multiecho_check.setChecked(True)
        window.multiecho_count_spin.setValue(3)
        window.multiecho_delay_spin.setValue(.15)
        window.multiecho_strength_spin.setValue(.35)
        window.reverb_check.setChecked(True)
        window.reverb_decay_spin.setValue(.4)
        window.reverb_wet_spin.setValue(.15)
        before = apply()
        order_before = window._effect_settings().chain_order
        row = next(i for i in range(window.chain_list.count())
                   if window.chain_list.item(i).data(Qt.UserRole) == 'compression')
        window.chain_list.setCurrentRow(row)
        window.chain_down_button.click()
        assert window.effected_clip is None
        after = apply()
        order_after = window._effect_settings().chain_order
        difference = float(np.sqrt(np.mean((before.samples.astype(float) - after.samples) ** 2)))
        assert difference > 1e-6
        chain_record = export('串联音效_中文合成.wav', after)
        window.tabs.setCurrentIndex(1)
        window.extension_tabs.setCurrentIndex(0)
        window.advanced_scroll.verticalScrollBar().setValue(window.advanced_scroll.verticalScrollBar().maximum())
        screenshot('高级音效与中文TTS_串联设置.png')

        window.resize(1080, 720)
        app.processEvents()
        window.extension_tabs.setCurrentIndex(1)
        app.processEvents()
        button = window.tts_send_button
        pos = button.mapTo(window, button.rect().bottomRight())
        assert pos.x() < window.width() and pos.y() < window.height()
        window.extension_tabs.setCurrentIndex(0)
        window.advanced_scroll.ensureWidgetVisible(window.advanced_apply_button, 10, 10)
        app.processEvents()
        button = window.advanced_apply_button
        pos = button.mapTo(window.advanced_scroll.viewport(), button.rect().center())
        assert window.advanced_scroll.viewport().rect().contains(pos)

        report = {
            'source': '公开自写中文测试句，eSpeak NG cmn离线合成，非真人录音',
            'text': text, 'tts_speed': 1, 'exports': [tts_record, advanced_record, mechanical_record, chain_record],
            'advanced_parameters': {'pitch_semitones': 2, 'tempo': 1.15, 'reverb_wet': .25,
                                    'reverb_decay_seconds': .6, 'compression': '阈值−18dB，比率3:1，补偿3dB'},
            'chain_before': order_before, 'chain_after': order_after,
            'chain_order_difference_rms': difference,
            'checks': ['TTS生成/独立保存保留旧原声，送入才替换',
                       '高级音效的图形/播放缓存/PCM16导出一致',
                       '独立变调变速及混响尾音长度准确，机械调制保持长度',
                       '串联顺序改变后缓存失效，重新处理波形确实不同',
                       '两页试听音量共享，未修改原声',
                       '1080×720中文TTS关键按钮可见，高级参数可滚动到达'],
            'scope': '未启动麦克风或扬声器，未联网；声音自然度和真人录音听感仍需人工试听'}
        (artifacts / '高级音效与中文TTS验收数据.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        window.close()


if __name__ == '__main__':
    main()
