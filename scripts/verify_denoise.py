"""生成已标注的合成样本，运行真实去噪/GUI/导出并保存验收证据。

运行：QT_QPA_PLATFORM=offscreen .venv-run/bin/python -m scripts.verify_denoise
不启动麦克风或扬声器；SNR仅对本脚本已知的干净合成参考计算。
"""
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]


def snr(clean, samples):
    return float(10*np.log10(np.sum(clean**2)/np.sum((samples-clean)**2)))


def main():
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.runtime'/'matplotlib'))
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from voice_effect_lab.audio.data import AudioClip
    from voice_effect_lab.audio.denoise import spectral_subtract
    from voice_effect_lab.ui.main_window import MainWindow

    samples_dir, artifacts = ROOT/'samples', ROOT/'artifacts'
    samples_dir.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    rng, rate = np.random.default_rng(20261020), 16000
    times = np.arange(rate*3)/rate
    clean = (.20*np.sin(2*np.pi*440*times)+.08*np.sin(2*np.pi*880*times))
    clean *= .65+.35*np.sin(2*np.pi*2*times)**2
    clean[times < .4] = 0
    noisy = AudioClip(clean+rng.normal(0, .055, len(times)), rate, '去噪测试_人工加噪.wav')
    noise = AudioClip(rng.normal(0, .055, rate*2), rate, '去噪测试_独立底噪.wav')
    path = samples_dir/noisy.name
    wavfile.write(path, rate, noisy.samples)
    wavfile.write(samples_dir/'去噪测试_干净参考.wav', rate, clean.astype(np.float32))
    wavfile.write(samples_dir/noise.name, rate, noise.samples)
    result = spectral_subtract(noisy, noise, 60)
    metrics = {
        'description': '440/880Hz调幅谐波 + 独立白噪声，非真人语音；前0.4秒只有噪声',
        'seed': 20261020, 'sample_rate': rate, 'duration_seconds': noisy.duration,
        'strength_percent': 60,
        'input_snr_db': snr(clean, noisy.samples),
        'independent_reference_output_snr_db': snr(clean, result.clip.samples),
        'algorithm_seconds': result.elapsed_seconds,
    }
    metrics['independent_reference_improvement_db'] = metrics['independent_reference_output_snr_db']-metrics['input_snr_db']

    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    app.setFont(QFont('Noto Sans CJK SC', 10))
    window = MainWindow()

    def wait_idle():
        deadline = time.monotonic()+20
        while window.busy and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(10)
        app.processEvents()
        assert not window.busy, '后台任务超时'
        assert not window.last_error, window.last_error

    try:
        window.show()
        window.load_path(path)
        wait_idle()
        original = window.clip
        assert window.set_noise_segment(0, .35, confirmed=True)
        window.denoise_button.click()
        wait_idle()
        processed = window.current_clip
        assert window.stages.checkedId() == 1
        assert window.player._clip is processed
        assert window.clip is original
        assert processed.duration == original.duration
        metrics['selected_reference_seconds'] = .35
        metrics['selection_output_snr_db'] = snr(clean, processed.samples)
        metrics['selection_improvement_db'] = metrics['selection_output_snr_db']-metrics['input_snr_db']
        assert metrics['selection_improvement_db'] > 3
        export_path = artifacts/'第三阶段_合成去噪输出.wav'
        window.save_to(export_path)
        wait_idle()
        saved_rate, saved = wavfile.read(export_path)
        assert saved_rate == rate
        np.testing.assert_allclose(saved/32768, processed.samples, atol=1/32768)
        window.stages.button(0).click()
        assert window.current_clip is original
        window.stages.button(1).click()
        assert window.current_clip is processed
        window.plots.draw()
        QTest.qWait(100)
        assert window.grab().save(str(artifacts/'第三阶段实际界面.png'))
        metrics['gui_flow'] = '真实导入→确认选段→去噪→导出核对→阶段切换→截图：通过；未启动音频设备'
        (artifacts/'第三阶段验收数据.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        window.close()


if __name__ == '__main__':
    main()
