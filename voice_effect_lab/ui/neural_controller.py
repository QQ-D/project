"""Independent educational BP model, training data, comparison and preview cache.

Host integration: initialize before building, insert build_neural_panel, call
_neural_controls and _neural_source_changed on source adoption, and close with
_close_neural after the host's active worker has finished. Host owns global job,
capture and playback exclusion, and synchronizes volume/mute on neural_player.
"""
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QDialog, QVBoxLayout

from ..analysis import analyze
from ..audio.data import AudioError, load_wav, save_wav
from ..audio.player import AudioPlayer, PlaybackError
from ..audio.neural import train_model, denoise_neural, load_model, save_model, make_demo_pair
from ..audio.wiener import wiener_denoise
from ..audio.deepfilter import deepfilter_denoise, deepfilter_available
from .neural_panel import make_neural_panel
from .plots import AnalysisPlots


class _NeuralProgress(QObject):
    """Own a main-thread slot so worker progress never touches a widget."""
    message = pyqtSignal(str)

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.message.connect(self._append)

    @pyqtSlot(str)
    def _append(self, text):
        if not self.host._neural_closed and hasattr(self.host, 'neural_log'):
            self.host.neural_log.appendPlainText(text)


class NeuralController:
    def _init_neural(self):
        self.neural_model = self.neural_training_clip = None
        self.neural_clip = self.neural_analysis = None
        self.neural_deepfilter_clip = self.neural_deepfilter_analysis = None
        self.neural_traditional_clip = self.neural_traditional_analysis = None
        self.neural_source_clip = self.neural_source_analysis = None
        self.neural_metrics = {}
        self.neural_plot_dialog = self.neural_plots = None
        self._neural_demo_clean = self._neural_demo_source = None
        self._neural_revision = self._neural_training_revision = 0
        self._neural_closed = False
        self._neural_progress = _NeuralProgress(self)
        self.neural_player = AudioPlayer(self)
        self.neural_player.failed.connect(self._error)
        self.neural_player.state_changed.connect(self._neural_playback_state)
        self.neural_player.position_changed.connect(self._neural_position)
        self.neural_player.output_device_changed.connect(self._neural_output_fallback)

    def build_neural_panel(self):
        if not hasattr(self, 'neural_panel'):
            self.neural_panel = make_neural_panel(self)
        return self.neural_panel

    def _neural_training_changed(self, *_):
        self._neural_training_revision += 1

    def _neural_training_snapshot(self):
        return (id(self.neural_training_clip), self.neural_epochs_spin.value(),
                self.neural_batch_spin.value(), self._neural_training_revision)

    def _neural_use_synthetic(self):
        if self.occupied:
            return
        self.neural_training_clip = None
        self._neural_training_changed()
        self.neural_training_label.setText('训练数据：公开合成谐波与噪声（默认）。')

    def load_neural_training_path(self, path):
        if self.occupied or not path:
            return
        def compute():
            clip = load_wav(path)
            if not 2 <= clip.duration <= 60:
                raise AudioError('干净训练 WAV 时长必须为 2–60 秒。')
            return clip
        def apply(clip):
            if self._neural_closed:
                return
            self.neural_training_clip = clip
            self._neural_training_changed()
            self.neural_training_label.setText(f'干净训练数据：{clip.name} · {clip.duration:.2f} s；将自动混合训练噪声。')
            self._status('干净训练数据已导入；模型和工作台原声保持独立。')
        self._job(compute, apply, '正在读取干净训练 WAV…')

    def start_neural_training(self):
        if self.occupied:
            return
        snapshot = self._neural_training_snapshot()
        clean = self.neural_training_clip
        epochs, batch = snapshot[1:3]
        self.neural_log.clear()
        def compute():
            return train_model(clean, epochs=epochs, batch_size=batch,
                               progress=self._neural_progress.message.emit)
        def apply(result):
            if self._neural_closed or self._neural_training_snapshot() != snapshot:
                self._status('训练数据或参数已变化，本次训练模型已丢弃。')
                return
            self._neural_adopt_model(result.model)
            self.neural_model_label.setText(
                f'模型就绪 · {result.model.training_label}\n'
                f'{len(result.train_losses)} 轮 · 训练损失 {result.train_losses[-1]:.4f} · '
                f'验证损失 {result.validation_losses[-1]:.4f}\n'
                f'训练/验证样本 {result.train_samples}/{result.validation_samples} · {result.elapsed_seconds:.3f} s')
            self._status('BP 训练完成，可保存模型或运行与维纳的比较。')
        self._job(compute, apply, '正在训练小型 BP 网络（CPU）…')

    def _neural_adopt_model(self, model):
        self.neural_model = model
        self._neural_clear_results('bp')
        self.neural_model_label.setText(f'模型就绪 · {model.training_label}')

    def load_neural_model_path(self, path):
        if self.occupied or not path:
            return
        def apply(model):
            if not self._neural_closed:
                self._neural_adopt_model(model)
                self._status('本项目 BP 模型已加载。')
        self._job(lambda: load_model(path), apply, '正在验证并加载 BP 模型 NPZ…')

    def save_neural_model_to(self, path):
        if self.occupied or self.neural_model is None or not path:
            return
        if Path(path).suffix.lower() != '.npz':
            self._error('请使用 .npz 扩展名保存本项目模型。')
            return
        model = self.neural_model
        self._job(lambda: save_model(path, model),
                  lambda _: self._status(f'已保存 BP 模型：{path}'), '正在保存 BP 模型…')

    def load_neural_demo(self):
        if self.occupied:
            return
        def compute():
            clean, noisy = make_demo_pair(seed=20261022)
            return clean, noisy, analyze(noisy)
        def apply(result):
            if self._neural_closed:
                return
            clean, noisy, analysis = result
            self._loaded((noisy, analysis))
            # _loaded invokes the normal source-change hook, clearing any old
            # reference first. Record this exact noisy identity only afterwards.
            self._neural_demo_clean, self._neural_demo_source = clean, noisy
            self._status('公开加噪演示已载入；此原声具有独立干净参考，可计算 SNR。')
        self._job(compute, apply, '正在生成公开加噪演示与分析数据…')

    def _neural_engine_changed(self, *_):
        engine = self.neural_engine_combo.currentData()
        is_bp = engine == 'bp'
        self.neural_training_panel.setVisible(is_bp)
        self.deepfilter_parameters.setVisible(not is_bp)
        self.neural_scope_label.setText(
            '教学小型 BP 频谱掩膜网络；效果依赖训练数据，不保证优于维纳。' if is_bp else
            'DeepFilterNet3 预训练模型；无需训练或底噪参考，100%为完整模型输出。')
        self.neural_selection_combo.setCurrentIndex(self.neural_selection_combo.findData(engine))
        self._neural_selection_changed()

    def _neural_snapshot(self):
        return (id(self.clip), id(self.neural_model), self.neural_strength_slider.value(),
                self.deepfilter_strength_spin.value(), self._neural_revision,
                self.neural_engine_combo.currentData())

    def run_neural_comparison(self):
        engine = self.neural_engine_combo.currentData()
        if self.occupied or self.clip is None or (engine == 'bp' and self.neural_model is None):
            return
        source, model = self.clip, self.neural_model
        snapshot = self._neural_snapshot()
        strength = snapshot[2] if engine == 'bp' else snapshot[3]
        name = '小型 BP' if engine == 'bp' else 'DeepFilterNet3'
        clean = self._neural_demo_clean if self._neural_demo_source is source else None

        def compute():
            traditional = wiener_denoise(source, noise_clip=None, strength=60)
            result = (denoise_neural(source, model, strength=strength) if engine == 'bp'
                      else deepfilter_denoise(source, noise_clip=None, strength=strength))
            analyses = analyze(source), analyze(traditional.clip), analyze(result.clip)
            def snr(clip):
                if clean is None:
                    return None
                reference = clean.samples.astype(np.float64)
                signal = float(np.sum(reference**2))
                error = float(np.sum((clip.samples.astype(np.float64)-reference)**2))
                if signal == 0:
                    return None
                return float(10*np.log10(signal/max(error, np.finfo(float).tiny)))
            metrics = {'source_snr_db': snr(source), 'traditional_snr_db': snr(traditional.clip),
                       engine+'_snr_db': snr(result.clip),
                       'traditional_elapsed_seconds': traditional.elapsed_seconds,
                       engine+'_elapsed_seconds': result.elapsed_seconds,
                       'traditional_clipped_samples': traditional.clipped_samples,
                       engine+'_clipped_samples': result.clipped_samples}
            return traditional.clip, result.clip, analyses, metrics

        def apply(result):
            if self._neural_closed or self._neural_snapshot() != snapshot:
                self._status('原声、模型、方法或强度已变化，本次比较结果已丢弃。')
                return
            traditional, selected, analyses, metrics = result
            self.neural_source_clip, self.neural_traditional_clip = source, traditional
            self.neural_source_analysis, self.neural_traditional_analysis = analyses[:2]
            if engine == 'bp':
                self.neural_clip, self.neural_analysis = selected, analyses[2]
            else:
                self.neural_deepfilter_clip, self.neural_deepfilter_analysis = selected, analyses[2]
            self.neural_metrics.update(metrics)
            self._neural_metrics_text()
            self.neural_selection_combo.setCurrentIndex(self.neural_selection_combo.findData(engine))
            self._neural_selection_changed()
            self._status(f'{name} / 自动维纳 60% 比较完成；工作台原声与已有处理结果保留。')
        self._job(compute, apply, f'正在运行{name} / 自动维纳比较并计算同源分析图…')

    def _neural_metrics_text(self):
        if not self.neural_metrics:
            self.neural_metrics_label.setText('尚无比较结果；无干净参考时 SNR 显示 —。')
            return
        def display(key):
            value = self.neural_metrics.get(key+'_snr_db')
            return '—' if value is None else f'{value:.2f} dB'
        lines = [f'SNR 原声 {display("source")} · 维纳 {display("traditional")}']
        for key, name in [('deepfilter', 'DeepFilterNet3'), ('bp', '小型 BP')]:
            elapsed = self.neural_metrics.get(key+'_elapsed_seconds')
            lines.append(f'{name}：未运行 / 结果已失效' if elapsed is None else
                         f'{name}：SNR {display(key)} · {elapsed:.3f} s')
        lines.append(f'维纳耗时 {self.neural_metrics["traditional_elapsed_seconds"]:.3f} s')
        self.neural_metrics_label.setText('\n'.join(lines))

    def _neural_clear_results(self, engine=None):
        self._neural_revision += 1
        if engine in (None, 'bp'):
            self.neural_clip = self.neural_analysis = None
        if engine in (None, 'deepfilter'):
            self.neural_deepfilter_clip = self.neural_deepfilter_analysis = None
        if engine is None:
            self.neural_traditional_clip = self.neural_traditional_analysis = None
            self.neural_source_clip = self.neural_source_analysis = None
            self.neural_metrics = {}
            self._neural_close_plots()
        else:
            self.neural_metrics = {key: value for key, value in self.neural_metrics.items()
                                   if not key.startswith(engine+'_')}
        if hasattr(self, 'neural_metrics_label'):
            self._neural_metrics_text()
            self._neural_selection_changed()

    def _neural_source_changed(self):
        self._neural_demo_clean = self._neural_demo_source = None
        self._neural_clear_results()

    def _neural_strength_changed(self, value):
        self.neural_strength_label.setText(f'{value}%')
        self._neural_clear_results('bp')

    def _deepfilter_strength_changed(self, value):
        self._neural_clear_results('deepfilter')

    def _neural_selected(self):
        key = self.neural_selection_combo.currentData()
        return {
            'source': (self.neural_source_clip, self.neural_source_analysis, '原声'),
            'traditional': (self.neural_traditional_clip, self.neural_traditional_analysis, '传统维纳 60%'),
            'deepfilter': (self.neural_deepfilter_clip, self.neural_deepfilter_analysis, 'DeepFilterNet3'),
            'bp': (self.neural_clip, self.neural_analysis, '小型 BP'),
        }.get(key, (None, None, '待处理'))

    def _neural_refresh_plots(self):
        clip, analysis, name = self._neural_selected()
        source = self.neural_source_clip if self.neural_source_clip is not None else self.clip
        source_analysis = self.neural_source_analysis if self.neural_source_clip is not None else self.analysis
        for plots in (getattr(self, 'intelligent_plots', None), self.neural_plots):
            if plots is None:
                continue
            if source is not None:
                plots.set_source(source, source_analysis)
            if clip is not None:
                plots.set_result(clip, analysis, name)
            else:
                plots.clear_result()

    def _neural_selection_changed(self, *_):
        clip, analysis, name = self._neural_selected()
        if self.neural_player._clip is not clip:
            self.neural_player.set_clip(clip)
        self._neural_position(self.neural_player.position_ms)
        self._neural_refresh_plots()
        self._neural_controls()

    def play_neural(self):
        clip, _, _ = self._neural_selected()
        if self.occupied or clip is None:
            return
        self.player.stop()
        self.tts_player.stop()
        try:
            self.neural_player.set_output_device(self.output_combo.currentData())
            self.neural_player.play()
        except PlaybackError as exc:
            self._error(str(exc))

    def save_neural_to(self, path):
        clip, _, name = self._neural_selected()
        if self.occupied or clip is None or not path:
            return
        if Path(path).suffix.lower() != '.wav':
            self._error('请使用 .wav 扩展名保存所选音频。')
            return
        self._job(lambda: save_wav(path, clip), lambda _: self._status(f'已保存{name} WAV：{path}'), '正在保存所选比较 WAV…')

    def send_neural_to_workbench(self):
        clip, analysis, name = self._neural_selected()
        if self.occupied or clip is None:
            return
        self.neural_player.stop()
        self.player.stop()
        self.tts_player.stop()
        self._loaded((clip, analysis))
        self.tabs.setCurrentIndex(0)
        self._controls()
        self._status(f'{name}已送入工作台作为新原声。')

    def show_neural_plots(self):
        clip, analysis, name = self._neural_selected()
        if self.occupied or clip is None:
            return
        if self.neural_plot_dialog is None:
            dialog = QDialog(self)
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            dialog.setWindowTitle('原声 / 所选比较结果 · 六图')
            dialog.resize(950, 720)
            layout = QVBoxLayout(dialog)
            self.neural_plots = AnalysisPlots(dialog)
            layout.addWidget(self.neural_plots)
            self.neural_plot_dialog = dialog
            # A QObject-bound receiver is disconnected when its host dies.
            # Do not capture parent/dialog wrappers from destroyed: deferred
            # deletion during cyclic GC can otherwise re-enter freed wrappers.
            dialog.finished.connect(self._neural_plot_finished)
        self.neural_plots.set_source(self.neural_source_clip, self.neural_source_analysis)
        self.neural_plots.set_result(clip, analysis, name)
        self.neural_plot_dialog.show()
        self.neural_plot_dialog.raise_()

    def _neural_plot_finished(self, result):
        if self.neural_plot_dialog is self.sender():
            self.neural_plot_dialog = self.neural_plots = None

    def _neural_close_plots(self):
        dialog = self.neural_plot_dialog
        self.neural_plot_dialog = self.neural_plots = None
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()

    def _neural_controls(self):
        if not hasattr(self, 'neural_train_button'):
            return
        idle = not self.occupied and not self._neural_closed
        if not idle and self.neural_player.state in ('playing', 'paused'):
            self.neural_player.stop()
        clip, _, _ = self._neural_selected()
        ready = idle and clip is not None
        for widget in (self.neural_training_synthetic_button, self.neural_training_load_button,
                       self.neural_epochs_spin, self.neural_batch_spin, self.neural_train_button,
                       self.neural_model_load_button, self.neural_demo_button,
                       self.neural_strength_slider, self.neural_strength_spin,
                       self.deepfilter_strength_slider, self.deepfilter_strength_spin,
                       self.neural_engine_combo):
            widget.setEnabled(idle)
        self.neural_model_save_button.setEnabled(idle and self.neural_model is not None)
        engine = self.neural_engine_combo.currentData()
        available, message = deepfilter_available() if engine == 'deepfilter' else (self.neural_model is not None, '教学 BP 训练、模型与原声独立保存。')
        self.neural_unavailable_label.setText(message)
        if self.clip is not None:
            self.neural_source_label.setText(f'当前原声：{self.clip.name}\n{self.clip.sample_rate} Hz · {self.clip.duration:.2f} s')
        self.neural_compare_button.setText('运行 DeepFilterNet3 / 维纳比较' if engine == 'deepfilter' else '运行 BP / 维纳比较')
        self.neural_compare_button.setEnabled(idle and available and self.clip is not None)
        self.neural_selection_combo.setEnabled(idle and self.neural_source_clip is not None)
        results = {'source': self.neural_source_clip, 'traditional': self.neural_traditional_clip,
                   'bp': self.neural_clip, 'deepfilter': self.neural_deepfilter_clip}
        for index in range(self.neural_selection_combo.count()):
            item = self.neural_selection_combo.model().item(index)
            item.setEnabled(results[self.neural_selection_combo.itemData(index)] is not None)
        self.neural_play_button.setEnabled(ready and self.neural_player.state != 'playing')
        self.neural_stop_button.setEnabled(ready and self.neural_player.state in ('playing', 'paused', 'finished'))
        for widget in (self.neural_plots_button, self.neural_save_button, self.neural_send_button):
            widget.setEnabled(ready)

    def _neural_position(self, position):
        if hasattr(self, 'neural_progress_label'):
            self.neural_progress_label.setText(f'{position/1000:.2f} / {self.neural_player.duration_ms/1000:.2f} s')

    def _neural_playback_state(self, state):
        self._neural_controls()
        if state == 'playing':
            self._status(f'正在试听所选{self._neural_selected()[2]}比较缓存。')

    def _neural_output_fallback(self, name):
        if hasattr(self, '_output_fallback'):
            self._output_fallback(name)

    def _close_neural(self):
        self._neural_closed = True
        self.neural_player.close()
        self._neural_close_plots()

    def _neural_choose_training(self):
        if not self.occupied:
            path, _ = QFileDialog.getOpenFileName(self, '导入干净训练 WAV', '', 'WAV 音频 (*.wav *.WAV)')
            self.load_neural_training_path(path)

    def _neural_choose_model_load(self):
        if not self.occupied:
            path, _ = QFileDialog.getOpenFileName(self, '加载本项目 BP 模型', '', 'BP 模型 (*.npz *.NPZ)')
            self.load_neural_model_path(path)

    def _neural_choose_model_save(self):
        if not self.occupied and self.neural_model is not None:
            path, _ = QFileDialog.getSaveFileName(self, '保存 BP 模型', '小型BP模型.npz', 'BP 模型 (*.npz)')
            self.save_neural_model_to(path)

    def _neural_choose_save(self):
        clip, _, name = self._neural_selected()
        if not self.occupied and clip is not None:
            path, _ = QFileDialog.getSaveFileName(self, f'保存{name}', f'{name}.wav', 'WAV 音频 (*.wav)')
            self.save_neural_to(path)
