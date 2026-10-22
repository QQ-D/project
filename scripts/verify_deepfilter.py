"""真实Qt使用dbaf/内置DF3，与用户指定原工程直接调用对照；仅用公开TTS。"""
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.runtime/matplotlib'))
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from voice_effect_lab.audio.data import AudioClip, load_wav, save_wav
    from voice_effect_lab.audio.deepfilter import _python
    from voice_effect_lab.audio.tts import synthesize_chinese
    from voice_effect_lab.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    app.setFont(QFont('Noto Sans CJK JP', 10))
    artifacts = root / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    tts_path = artifacts / '中文TTS_公开测试句.wav'
    clean = load_wav(tts_path) if tts_path.exists() else synthesize_chinese('欢迎来到数字信号处理实验。我们正在测试中文合成和房间混响。')
    if not tts_path.exists():
        save_wav(tts_path, clean)
    values = resample_poly(clean.samples, 48000, clean.sample_rate).astype(np.float32)*.8
    clean = AudioClip(values, 48000, '公开中文合成_干净参考.wav')
    noisy = AudioClip(values + np.random.default_rng(20261021).normal(0, .025, len(values)),
                      48000, '公开中文合成_人工加噪.wav')
    input_path = artifacts / 'DeepFilterNet3_公开中文加噪输入.wav'
    save_wav(input_path, noisy)
    save_wav(artifacts / 'DeepFilterNet3_公开中文干净参考.wav', clean)
    window = MainWindow()
    window.show()

    def wait_idle():
        deadline = time.monotonic() + 90
        while window.busy and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(10)
        app.processEvents()
        assert not window.busy and not window.last_error, window.last_error

    def check_export(path, clip):
        rate, data = wavfile.read(path)
        assert rate == clip.sample_rate and len(data) == len(clip.samples)
        error = float(np.max(np.abs(data/32768 - clip.samples)))
        assert error <= 1/32768
        return {'file': path.name, 'sample_rate': rate, 'samples': len(data),
                'seconds': clip.duration, 'pcm16_max_error': error}

    def capture(name):
        window.plots.draw()
        if window.intelligent_plots is not None:
            window.intelligent_plots.draw()
        app.processEvents()
        QTest.qWait(100)
        assert window.grab().save(str(artifacts / name))

    try:
        window.load_path(input_path)
        wait_idle()
        source = window.clip
        assert window.denoise_method_combo.currentData() == 'wiener'
        assert window.denoise_spin.value() == 60 and not window.noise_reference_panel.isHidden()
        window.tabs.setCurrentIndex(1)
        window.extension_tabs.setCurrentIndex(2)
        assert window.neural_engine_combo.currentData() == 'deepfilter'
        assert window.deepfilter_strength_spin.value() == 100 and window.neural_model is None
        started = time.monotonic()
        window.neural_compare_button.click()
        wait_idle()
        task_seconds = time.monotonic() - started
        result = window.neural_deepfilter_clip
        assert window.current_clip is source and window.neural_player._clip is result
        assert window.clip is source
        assert result.sample_rate == source.sample_rate and len(result.samples) == len(source.samples)
        np.testing.assert_array_equal(window.intelligent_plots.axes[0, 1].lines[0].get_ydata(), window.neural_deepfilter_analysis.wave_values)
        np.testing.assert_array_equal(window.intelligent_plots.axes[2, 1].images[0].get_array(), window.neural_deepfilter_analysis.spectrogram_db)
        output_path = artifacts / 'DeepFilterNet3_公开中文降噪输出.wav'
        window.save_neural_to(output_path)
        wait_idle()
        exports = [check_export(input_path, source), check_export(output_path, result)]
        window.control_scroll.verticalScrollBar().setValue(0)
        capture('DeepFilterNet3_实际工作台.png')

        original_project = Path('/media/rwj/ZX1 2TB/project/DeepFilterNet')
        reference = {'checked': False, 'reason': '参考外置盘未挂载，运行仅使用内置模型'}
        if original_project.is_dir():
            code = '''import sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
sys.path.insert(0,str(Path(sys.argv[1])/'DeepFilterNet'))
from df.enhance import init_df,enhance
model,state,*_=init_df(str(Path(sys.argv[1])/'models/DeepFilterNet3'),log_file=None,log_level='ERROR')
samples=np.load(sys.argv[2],allow_pickle=False)
output=enhance(model,state,torch.from_numpy(samples).unsqueeze(0),pad=True).squeeze(0).numpy()
np.save(sys.argv[3],output.astype(np.float32),allow_pickle=False)
'''
            with TemporaryDirectory(prefix='voice-lab-df3-reference-') as folder:
                inp, out = Path(folder)/'in.npy', Path(folder)/'out.npy'
                np.save(inp, source.samples, allow_pickle=False)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                env.pop('PYTHONPATH', None)
                subprocess.run([str(_python()), '-c', code, str(original_project), str(inp), str(out)],
                               check=True, capture_output=True, timeout=90, env=env, cwd=folder)
                expected = np.clip(np.load(out, allow_pickle=False), -1, 1)
                np.testing.assert_allclose(result.samples, expected, rtol=0, atol=1e-7)
                reference = {'checked': True, 'max_absolute_error': float(np.max(np.abs(result.samples-expected))),
                             'flow': '原工程init_df+enhance(pad=True)，与内置副本/dbaf结果直接比较'}

        window.send_neural_to_workbench()
        assert window.clip is result and window.neural_deepfilter_clip is None
        window.use_denoise_check.setChecked(False)
        window.enhance_check.setChecked(True)
        window.gain_spin.setValue(.8)
        window.reverb_check.setChecked(True)
        window.reverb_decay_spin.setValue(.2)
        window.reverb_wet_spin.setValue(.15)
        window.start_advanced_effects()
        wait_idle()
        assert window.denoised_clip is None and window.enhanced_clip is not None
        assert not window._effects_from_denoise and window._effects_from_enhancement
        chain = window.effected_clip
        assert window.current_clip is chain and window.player._clip is chain
        chain_path = artifacts / 'DeepFilterNet3_增强混响串联.wav'
        window.save_to(chain_path)
        wait_idle()
        exports.append(check_export(chain_path, chain))
        window.control_scroll.ensureWidgetVisible(window.effects_group, 10, 10)
        capture('DeepFilterNet3_音效串联.png')

        window.resize(1080, 720)
        app.processEvents()
        window.tabs.setCurrentIndex(1)
        window.extension_tabs.setCurrentIndex(2)
        window.neural_scroll.ensureWidgetVisible(window.neural_compare_button, 8, 8)
        app.processEvents()
        point = window.neural_compare_button.mapTo(window.neural_scroll.viewport(), window.neural_compare_button.rect().center())
        assert window.neural_scroll.viewport().rect().contains(point)
        window.load_path(tts_path)
        wait_idle()
        other_source = window.clip
        window.run_neural_comparison()
        wait_idle()
        assert window.neural_deepfilter_clip.sample_rate == other_source.sample_rate
        assert len(window.neural_deepfilter_clip.samples) == len(other_source.samples)
        report = {'source': '公开自写中文TTS，加人工高斯噪声；非私人录音',
                  'engine': 'DeepFilterNet3 / dbaf CPU', 'full_gui_task_seconds': task_seconds,
                  'reference_original_project': reference, 'exports': exports,
                  'additional_resample_case': {'sample_rate': other_source.sample_rate,
                                               'samples': len(other_source.samples), 'exact_output_length': True},
                  'checks': ['智能降噪子页默认DF3完整处理，无训练/底噪参考', '原声不变，图形/播放缓存/导出一致',
                             '模型结果明确送入工作台后增强/混响，不重复降噪', '1080×720核心处理按钮可滚动到达'],
                  'scope': '未安装依赖、下载模型或启动麦克风/扬声器；试听效果需人工判断'}
        (artifacts / 'DeepFilterNet3验收数据.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        window.close()


if __name__ == '__main__':
    main()
