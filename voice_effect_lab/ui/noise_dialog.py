from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QCheckBox, QPushButton, QDialogButtonBox,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.widgets import SpanSelector

from ..analysis import _waveform
from ..audio.data import AudioError
from ..audio.noise import extract_noise, noise_bounds
from ..audio.player import AudioPlayer, PlaybackError
from . import chart_style


class NoiseSelectionDialog(QDialog):
    def __init__(self, clip, parent=None, output_device=None):
        super().__init__(parent)
        self.setWindowTitle('选取纯噪声片段')
        self.resize(860, 520)
        self.clip = clip
        self.selected_clip = None
        self._closed = False
        self.player = AudioPlayer(self)
        self.player.set_output_device(output_device)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('拖动波形选区或修改起止时间，先试听，确认只含环境噪声后采用。'))
        source = QLabel(f'来源：{clip.name} · {clip.duration:.2f} s · {clip.sample_rate} Hz')
        source.setWordWrap(True)
        layout.addWidget(source)
        self.figure = Figure(figsize=(8, 2.6), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.subplots()
        times, samples = _waveform(clip.samples, clip.sample_rate)
        self.ax.plot(times, samples, color='#1685f8', linewidth=.7)
        self.ax.set_xlim(0, max(.01, clip.duration))
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlabel('时间（s）')
        self.ax.set_ylabel('原声幅度')
        self.ax.grid(True, color='#e3e9f1')
        self.selector = SpanSelector(self.ax, self.set_range, 'horizontal',
                                     useblit=True, interactive=True,
                                     props={'facecolor': '#f47729', 'alpha': .25})
        layout.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.start_spin = QDoubleSpinBox()
        self.end_spin = QDoubleSpinBox()
        for label, spin in [('起点', self.start_spin), ('终点', self.end_spin)]:
            spin.setDecimals(3)
            spin.setRange(0, clip.duration)
            spin.setSingleStep(.05)
            spin.setSuffix(' s')
            row.addWidget(QLabel(label))
            row.addWidget(spin)
        self.preview_button = QPushButton('试听选段')
        self.stop_button = QPushButton('停止试听')
        self.preview_button.clicked.connect(self.preview)
        self.stop_button.clicked.connect(self.player.stop)
        row.addWidget(self.preview_button)
        row.addWidget(self.stop_button)
        layout.addLayout(row)
        self.confirm_check = QCheckBox('我已确认：选段只有环境底噪，没有说话声')
        layout.addWidget(self.confirm_check)
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.use_button = buttons.button(QDialogButtonBox.Ok)
        self.use_button.setText('采用选段')
        buttons.button(QDialogButtonBox.Cancel).setText('取消')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.start_spin.valueChanged.connect(self._range_changed)
        self.end_spin.valueChanged.connect(self._range_changed)
        self.confirm_check.toggled.connect(self._refresh)
        self.player.failed.connect(self._show_error)
        self.set_range(0, min(1.5, clip.duration))

    def set_range(self, start, end):
        with QSignalBlocker(self.start_spin), QSignalBlocker(self.end_spin):
            self.start_spin.setValue(start)
            self.end_spin.setValue(end)
        self._range_changed()

    def _range_changed(self):
        self.player.stop()
        self.selector.extents = (self.start_spin.value(), self.end_spin.value())
        # 新区间必须重新确认，避免原区间的确认意外沿用。
        self.confirm_check.setChecked(False)
        self._refresh()
        self.canvas.draw_idle()

    def _refresh(self):
        try:
            first, last = noise_bounds(self.clip, self.start_spin.value(), self.end_spin.value())
        except AudioError as exc:
            self.use_button.setEnabled(False)
            self.preview_button.setEnabled(False)
            self.message.setText(str(exc))
        else:
            self.preview_button.setEnabled(True)
            self.use_button.setEnabled(self.confirm_check.isChecked())
            self.message.setText(f'选段时长：{(last-first)/self.clip.sample_rate:.3f} s；采样率：{self.clip.sample_rate} Hz。')

    def _show_error(self, message):
        self.message.setText('试听错误：'+message)

    def preview(self):
        try:
            clip = extract_noise(self.clip, self.start_spin.value(), self.end_spin.value())
            self.player.set_clip(clip)
            self.player.play()
        except (AudioError, PlaybackError) as exc:
            self._show_error(str(exc))

    def _release(self):
        if not self._closed:
            self._closed = True
            self.player.close()
            self.selector.disconnect_events()

    def accept(self):
        if not self.confirm_check.isChecked():
            self.message.setText('请先确认选段只有底噪、没有说话声。')
            return
        try:
            self.selected_clip = extract_noise(self.clip, self.start_spin.value(), self.end_spin.value())
        except AudioError as exc:
            self.message.setText(str(exc))
            return
        self._release()
        super().accept()

    def reject(self):
        self._release()
        super().reject()

    def closeEvent(self, event):
        self._release()
        super().closeEvent(event)
