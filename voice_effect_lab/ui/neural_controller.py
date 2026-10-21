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
        self._neural_clear_results()
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

    def _neural_snapshot(self):
        return (id(self.clip), id(self.neural_model), self.neural_strength_slider.value(), self._neural_revision)

    def run_neural_comparison(self):
        if self.occupied or self.neural_model is None or self.clip is None:
            return
        source, model = self.clip, self.neural_model
        snapshot = self._neural_snapshot()
        strength = snapshot[2]
        clean = self._neural_demo_clean if self._neural_demo_source is source else None
        def compute():
            traditional = wiener_denoise(source, noise_clip=None, strength=60)
            bp = denoise_neural(source, model, strength=strength)
            analyses = analyze(source), analyze(traditional.clip), analyze(bp.clip)
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
                       'bp_snr_db': snr(bp.clip), 'traditional_elapsed_seconds': traditional.elapsed_seconds,
                       'bp_elapsed_seconds': bp.elapsed_seconds,
                       'traditional_clipped_samples': traditional.clipped_samples,
                       'bp_clipped_samples': bp.clipped_samples}
            return traditional.clip, bp.clip, analyses, metrics
        def apply(result):
            if self._neural_closed or self._neural_snapshot() != snapshot:
                self._status('原声、模型或 BP 强度已变化，本次比较结果已丢弃。')
                return
            traditional, bp, analyses, metrics = result
            self._neural_close_plots()
            self.neural_source_clip, self.neural_traditional_clip, self.neural_clip = source, traditional, bp
            self.neural_source_analysis, self.neural_traditional_analysis, self.neural_analysis = analyses
            self.neural_metrics = metrics
            def display(value):
                return '—' if value is None else f'{value:.2f} dB'
            self.neural_metrics_label.setText(
                f'SNR 原声 {display(metrics["source_snr_db"])} · 维纳 {display(metrics["traditional_snr_db"])} · BP {display(metrics["bp_snr_db"])}\n'
                f'耗时：维纳 {metrics["traditional_elapsed_seconds"]:.3f} s · BP {metrics["bp_elapsed_seconds"]:.3f} s')
            self._neural_selection_changed()
            self._status('BP / 自动维纳 60% 比较完成，可选择同一缓存试听、查看六图或导出。')
        self._job(compute, apply, '正在运行 BP / 自动维纳比较并计算三份分析数据…')

    def _neural_clear_results(self):
        self._neural_revision += 1
        self.neural_player.set_clip(None)
        self.neural_clip = self.neural_analysis = None
        self.neural_traditional_clip = self.neural_traditional_analysis = None
        self.neural_source_clip = self.neural_source_analysis = None
        self.neural_metrics = {}
        self._neural_close_plots()
        if hasattr(self, 'neural_metrics_label'):
            self.neural_metrics_label.setText('尚无比较结果；无干净参考时 SNR 显示 —。')
            self._neural_position(0)
        self._neural_controls()

    def _neural_source_changed(self):
        self._neural_demo_clean = self._neural_demo_source = None
        self._neural_clear_results()

    def _neural_strength_changed(self, value):
        self.neural_strength_label.setText(f'{value}%')
        self._neural_clear_results()

    def _neural_selected(self):
        key = self.neural_selection_combo.currentData()
        if key == 'source':
            return self.neural_source_clip, self.neural_source_analysis, '原声'
        if key == 'traditional':
            return self.neural_traditional_clip, self.neural_traditional_analysis, '传统维纳 60%'
        return self.neural_clip, self.neural_analysis, '小型 BP'

    def _neural_selection_changed(self, *_):
        clip, analysis, name = self._neural_selected()
        self.neural_player.set_clip(clip)
        self._neural_position(0)
        if self.neural_plots is not None and clip is not None:
            self.neural_plots.set_source(self.neural_source_clip, self.neural_source_analysis)
            self.neural_plots.set_result(clip, analysis, name)
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
        ready = idle and self.neural_clip is not None
        for widget in (self.neural_training_synthetic_button, self.neural_training_load_button,
                       self.neural_epochs_spin, self.neural_batch_spin, self.neural_train_button,
                       self.neural_model_load_button, self.neural_demo_button,
                       self.neural_strength_slider, self.neural_strength_spin):
            widget.setEnabled(idle)
        self.neural_model_save_button.setEnabled(idle and self.neural_model is not None)
        self.neural_compare_button.setEnabled(idle and self.neural_model is not None and self.clip is not None)
        self.neural_selection_combo.setEnabled(ready)
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
            self._status('正在试听所选 BP 比较缓存。')

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
