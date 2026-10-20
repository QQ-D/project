from pathlib import Path

from PyQt5.QtCore import Qt, QThreadPool, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QScrollArea, QSplitter, QSlider, QDoubleSpinBox,
    QCheckBox, QRadioButton, QButtonGroup, QProgressBar, QFileDialog,
    QTabWidget, QComboBox,
)
from PyQt5.QtMultimedia import QAudio, QAudioDeviceInfo

from ..audio.data import load_wav, save_wav
from ..audio.player import AudioPlayer, PlaybackError
from ..analysis import analyze
from .worker import Job
from .plots import AnalysisPlots
from .capture_controller import CaptureController
from .processing_controller import ProcessingController


STYLE = '''
QMainWindow, QTabWidget::pane { background: #f2f5f9; }
QWidget { color: #263549; font-size: 13px; }
QGroupBox { background: white; border: 1px solid #d8e1eb; border-radius: 5px;
            margin-top: 10px; padding: 10px 8px 6px 8px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #285a9a; }
QPushButton { border: 1px solid #d8e1eb; border-radius: 4px; background: #f8fafd; padding: 6px 7px; }
QPushButton:hover { background: #e9f2ff; }
QPushButton[primary="true"] { background: #1685f8; color: white; border-color: #1685f8; }
QPushButton:disabled { color: #a5afbc; background: #f4f6f9; }
QPushButton:checked { background: #1685f8; color: white; }
QDoubleSpinBox, QComboBox, QPlainTextEdit { border: 1px solid #d8e1eb; border-radius: 3px; padding: 3px; background: #fafcfe; }
QProgressBar { border: 0; background: #e3e8ef; border-radius: 3px; height: 7px; }
QProgressBar::chunk { background: #1685f8; border-radius: 3px; }
QTabBar::tab { padding: 9px 20px; background: #e9eef6; }
QTabBar::tab:selected { background: #1685f8; color: white; }
QSlider::groove:horizontal { height: 4px; background: #d8e1eb; }
QSlider::handle:horizontal { background: white; border: 1px solid #8ca4bf; width: 12px; margin: -5px 0; border-radius: 6px; }
'''


def _button(text, primary=False):
    button = QPushButton(text)
    button.setProperty('primary', primary)
    return button


def _group(title, disabled=False):
    widget = QGroupBox(title)
    layout = QVBoxLayout(widget)
    layout.setSpacing(5)
    if disabled:
        widget.setEnabled(False)
        widget.setToolTip('后续阶段实现，目前仅展示设计入口。')
    return widget, layout


def _row(layout, *widgets):
    row = QHBoxLayout()
    for widget in widgets:
        row.addWidget(widget)
    layout.addLayout(row)


def _parameter(layout, label, low, high, value, suffix='', decimals=2):
    text = QLabel(label)
    text.setMinimumWidth(80)
    slider = QSlider(Qt.Horizontal)
    scale = 10**decimals
    slider.setRange(round(low*scale), round(high*scale))
    slider.setValue(round(value*scale))
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setSuffix(suffix)
    spin.setValue(value)
    spin.setFixedWidth(86)
    slider.valueChanged.connect(lambda v: spin.setValue(v/scale))
    spin.valueChanged.connect(lambda v: slider.setValue(round(v*scale)))
    _row(layout, text, slider, spin)
    return slider, spin


class MainWindow(QMainWindow, CaptureController, ProcessingController):
    loaded = pyqtSignal()

    def __init__(self, parent=None, recorder=None):
        super().__init__(parent)
        self.setWindowTitle('智能人声音效台 · 第三阶段')
        self.resize(1500, 1040)
        self.setMinimumSize(1080, 720)
        self.setStyleSheet(STYLE)
        self.clip = None
        self.analysis = None
        self.busy = False
        self.last_error = ''
        self._active_job = None
        self._close_pending = False
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.player = AudioPlayer(self)
        self.player.state_changed.connect(self._playback_state)
        self.player.position_changed.connect(self._position)
        self.player.failed.connect(self._error)
        self.player.output_device_changed.connect(self._output_fallback)
        self._init_capture(recorder)
        self._init_processing()
        self._build()
        self._controls()

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 10, 14, 10)
        header = QHBoxLayout()
        title = QLabel('智能人声音效台')
        title.setStyleSheet('font-size: 23px; font-weight: bold; color: #263549;')
        header.addWidget(title)
        header.addWidget(QLabel('第三阶段 · 录音 / 底噪采集 / 谱减去噪 / 对比试听'))
        header.addStretch()
        self.demo_button = _button('载入合成测试音')
        self.demo_button.clicked.connect(self.load_demo)
        header.addWidget(self.demo_button)
        outer.addLayout(header)
        tabs = QTabWidget()
        outer.addWidget(tabs, 1)
        basic = QWidget()
        layout = QHBoxLayout(basic)
        layout.setContentsMargins(0, 8, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        controls = QWidget()
        left = QVBoxLayout(controls)
        left.setContentsMargins(0, 0, 5, 0)
        left.setSpacing(5)
        self._source_panel(left)
        self._future_panels(left)
        actions = QHBoxLayout()
        self.process_button = _button('开始处理', True)
        self.process_button.setEnabled(False)
        self.process_button.setToolTip('当前执行谱减去噪；增强与音效将在后续阶段接入。')
        self.process_button.clicked.connect(self.start_denoise)
        self.reset_button = _button('重置')
        self.reset_button.clicked.connect(self.reset_processing)
        self.reset_button.setToolTip('恢复去噪强度60%并清除去噪结果，保留原声与底噪。')
        self.save_button = _button('保存原声 WAV')
        self.save_button.clicked.connect(self._choose_save)
        for widget in (self.process_button, self.reset_button, self.save_button):
            actions.addWidget(widget)
        left.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setFrameShape(QScrollArea.NoFrame)
        sidebar = QWidget()
        sidebar.setMinimumWidth(375)
        sidebar.setMaximumWidth(450)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(5)
        side_layout.addWidget(scroll, 1)
        # 试听与保存固定在侧栏底部，较小屏幕也不必滚动寻找核心操作。
        self._listen_panel(side_layout)
        side_layout.addLayout(actions)
        splitter.addWidget(sidebar)
        right, plots_layout = _group('处理前后可视化对比 · 原声始终保留')
        self.file_label = QLabel('尚未导入音频；右列将在去噪完成后显示当前试听阶段。')
        self.file_label.setWordWrap(True)
        plots_layout.addWidget(self.file_label)
        self.plots = AnalysisPlots(right)
        plots_layout.addWidget(self.plots, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([425, 1050])
        tabs.addTab(basic, '基础工作台 · 必做')
        tabs.addTab(self._extensions(), '扩展功能 · 选做（未实现）')
        self.status_label = QLabel('状态：先导入或录制原声，再采集或选取纯底噪，确认后可执行去噪。')
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet('color: #1771cb; padding: 6px;')
        outer.addWidget(self.status_label)

    def _source_panel(self, left):
        group, layout = _group('01  音频采集与管理')
        self.record_button = _button('开始录音')
        self.record_button.clicked.connect(lambda: self.start_recording('voice'))
        self.end_record_button = _button('结束录音')
        self.end_record_button.clicked.connect(self.stop_recording)
        self.import_button = _button('导入 WAV', True)
        self.import_button.clicked.connect(self._choose_load)
        _row(layout, self.record_button, self.end_record_button, self.import_button)
        self.info_label = QLabel('采样率：—    时长：—')
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        self.input_combo = QComboBox()
        self.input_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.input_combo.setMinimumContentsLength(14)
        self.input_combo.addItem('系统默认输入', None)
        for device in QAudioDeviceInfo.availableDevices(QAudio.AudioInput):
            self.input_combo.addItem(device.deviceName(), device.deviceName())
        self.input_combo.currentIndexChanged.connect(self._choose_input)
        _row(layout, QLabel('输入设备'), self.input_combo)
        self.sample_rate_combo = QComboBox()
        for rate in [16000, 44100, 48000]:
            self.sample_rate_combo.addItem(f'{rate} Hz', rate)
        self.sample_rate_combo.setCurrentIndex(1)
        _row(layout, QLabel('目标采样率'), self.sample_rate_combo)
        self.output_combo = QComboBox()
        self.output_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.output_combo.setMinimumContentsLength(14)
        self.output_combo.addItem('自动选择（优先系统默认）', None)
        for device in QAudioDeviceInfo.availableDevices(QAudio.AudioOutput):
            self.output_combo.addItem(device.deviceName(), device.deviceName())
        self.output_combo.currentIndexChanged.connect(self._choose_output)
        self.output_combo.setToolTip('先尝试系统默认输出；失败时尝试独立声卡路由。也可手动选择扬声器或耳机。')
        _row(layout, QLabel('输出设备'), self.output_combo)
        left.addWidget(group)

    def _future_panels(self, left):
        group, layout = _group('02  环境底噪采集 · 谱减去噪')
        self.noise_button = _button('采集底噪')
        self.noise_button.clicked.connect(lambda: self.start_recording('noise'))
        self.select_noise_button = _button('选取噪声段')
        self.select_noise_button.clicked.connect(self._choose_noise)
        self.denoise_button = _button('一键去噪')
        self.denoise_button.setEnabled(False)
        self.denoise_button.setToolTip('使用已确认的底噪估计噪声功率谱，不修改原声。')
        self.denoise_button.clicked.connect(self.start_denoise)
        _row(layout, self.noise_button, self.select_noise_button, self.denoise_button)
        self.noise_status_label = QLabel('底噪：尚未采集或选取')
        self.noise_status_label.setWordWrap(True)
        layout.addWidget(self.noise_status_label)
        self.confirm_noise_button = _button('确认沿用底噪')
        self.confirm_noise_button.clicked.connect(self.confirm_noise)
        self.clear_noise_button = _button('清除底噪')
        self.clear_noise_button.clicked.connect(self.clear_noise)
        _row(layout, self.confirm_noise_button, self.clear_noise_button)
        self.use_denoise_check = QCheckBox('使用去噪结果（后续增强输入）')
        self.use_denoise_check.setEnabled(False)
        self.use_denoise_check.setToolTip('增强阶段尚未实现；当前在底部选择原声或去噪进行试听与导出。')
        layout.addWidget(self.use_denoise_check)
        self.denoise_parameters = QWidget()
        parameters = QVBoxLayout(self.denoise_parameters)
        parameters.setContentsMargins(0, 0, 0, 0)
        self.denoise_slider, self.denoise_spin = _parameter(parameters, '去噪强度', 0, 100, 60, ' %', 0)
        self.denoise_spin.valueChanged.connect(self._invalidate_denoise)
        layout.addWidget(self.denoise_parameters)
        self.denoise_status_label = QLabel('去噪：尚未处理')
        self.denoise_status_label.setWordWrap(True)
        layout.addWidget(self.denoise_status_label)
        left.addWidget(group)
        group, layout = _group('03  声音增强 · 待实现', True)
        _row(layout, QCheckBox('启用增强'), QCheckBox('峰值归一化（0.95）'))
        _parameter(layout, '音量增益', 0, 2, 1, '×')
        _parameter(layout, '人声清晰度', 0, 100, 0, ' %', 0)
        layout.addWidget(_button('一键增强'))
        left.addWidget(group)
        group, layout = _group('04  一键音效预设 · 待实现', True)
        grid = QGridLayout()
        for index, label in enumerate(['原声', '磁性', '娃娃音', '大叔音', '电话音', '山谷回声', '自定义']):
            grid.addWidget(_button(label), index//4, index % 4)
        layout.addLayout(grid)
        left.addWidget(group)
        group, layout = _group('05  参数调节 · 待实现', True)
        for label, low, high, value, suffix in [
            ('播放倍率', .5, 2, 1, '×'), ('低频增益', -12, 12, 0, ' dB'),
            ('中频增益', -12, 12, 0, ' dB'), ('高频增益', -12, 12, 0, ' dB'),
            ('回声延时', 0, 1, .3, ' s'), ('回声强度', 0, .8, 0, '')]:
            _parameter(layout, label, low, high, value, suffix)
        left.addWidget(group)

    def _listen_panel(self, left):
        self.capture_label = QLabel('未在采集 · 语音最长 300 s / 底噪约 1.5 s')
        self.capture_label.setWordWrap(True)
        self.cancel_capture_button = _button('取消采集')
        self.cancel_capture_button.clicked.connect(self.cancel_recording)
        _row(left, self.capture_label, self.cancel_capture_button)
        group, layout = _group('06  对比试听')
        self.stages = QButtonGroup(self)
        row = QHBoxLayout()
        for i, label in enumerate(['原声', '去噪', '增强', '音效']):
            radio = QRadioButton(label)
            radio.setChecked(i == 0)
            radio.setEnabled(i == 0)
            if i > 1:
                radio.setToolTip('该处理阶段尚未实现。')
            elif i == 1:
                radio.setToolTip('确认底噪并执行去噪后可试听；参数改变后需重新处理。')
            self.stages.addButton(radio, i)
            row.addWidget(radio)
        layout.addLayout(row)
        self.stages.buttonClicked[int].connect(self.select_stage)
        self.play_button = _button('▶ 播放', True)
        self.pause_button = _button('Ⅱ 暂停')
        self.stop_button = _button('■ 停止')
        self.play_button.clicked.connect(self._play)
        self.pause_button.clicked.connect(self.player.pause)
        self.stop_button.clicked.connect(self._stop_action)
        _row(layout, self.play_button, self.pause_button, self.stop_button)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1000)
        self.progress_label = QLabel('0.00 / 0.00 s')
        _row(layout, self.progress, self.progress_label)
        left.addWidget(group)

    def _extensions(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.addWidget(QLabel('选做设计入口 · 当前均未实现；优先完成基础任务。'))
        row = QHBoxLayout()
        for title, labels in [
            ('高级音效', ['多重回声', '房间混响', '机械电音 / 调制', '动态压缩', '音效串联', '音高与语速独立调整']),
            ('中文 TTS', ['输入中文生成语音', '合成语速与音色', '试听与保存', '送入主工作台制作配音']),
            ('智能语音增强', ['预训练模型 / 小型 BP / CNN', '模型与数据集选择', '训练与日志', '降噪 / 去混响', '传统方法与智能方法对比'])]:
            group, layout = _group(title, True)
            for label in labels:
                layout.addWidget(_button(label))
            layout.addStretch()
            row.addWidget(group)
        outer.addLayout(row, 1)
        return page

    def _controls(self):
        available = self.current_clip is not None and not self.occupied
        self.import_button.setEnabled(not self.occupied)
        self.demo_button.setEnabled(not self.occupied)
        self.output_combo.setEnabled(not self.occupied)
        self.play_button.setEnabled(available and self.player.state != 'playing')
        self.play_button.setText('▶ 继续' if self.player.state == 'paused' else '▶ 播放')
        self.pause_button.setEnabled(available and self.player.state == 'playing')
        self.stop_button.setEnabled(available and self.player.state in ('playing', 'paused', 'finished'))
        self.save_button.setEnabled(available)
        self._capture_controls()
        self._processing_controls()

    def _job(self, function, result, message):
        if self.occupied:
            return
        self.busy = True
        self.last_error = ''
        self.player.stop()
        self._status(message)
        self._controls()
        job = Job(function)
        self._active_job = job
        job.signals.result.connect(result)
        job.signals.failed.connect(self._error)
        job.signals.finished.connect(self._job_finished)
        self.pool.start(job)

    def _job_finished(self):
        self.busy = False
        self._active_job = None
        self._controls()
        if self._close_pending:
            self.close()

    def load_path(self, path):
        if self.occupied or not path:
            return
        def load():
            clip = load_wav(path)
            return clip, analyze(clip)
        self._job(load, self._loaded, '正在读取 WAV 并计算真实分析图…')

    def _loaded(self, result, source_kind='import'):
        self.clip, self.analysis = result
        self._source_replaced(source_kind)
        self.player.set_clip(self.clip)
        self.info_label.setText(f'采样率：{self.clip.sample_rate} Hz    时长：{self.clip.duration:.2f} s\n'
                                f'源音频 {self.clip.source_channels} 声道 → 当前单声道')
        self.file_label.setText(f'{self.clip.name}  |  频谱/语谱显示至 {min(8000, self.clip.sample_rate/2):g} Hz  |  当前阶段：原声')
        self.plots.set_source(self.clip, self.analysis)
        self._position(0)
        self._status('原声已加载，可试听；请采集或选取底噪，确认后执行去噪。')
        self.loaded.emit()

    def load_demo(self):
        path = Path(__file__).resolve().parents[2] / 'samples' / '合成测试音.wav'
        self.load_path(path)

    def save_to(self, path):
        if self.current_clip is None or self.occupied or not path:
            return
        clip, stage = self.current_clip, self.stage_name
        self._job(lambda: save_wav(path, clip),
                  lambda _: self._status(f'已保存{stage} WAV：{path}'), f'正在保存{stage} WAV…')

    def _choose_load(self):
        path, _ = QFileDialog.getOpenFileName(self, '导入 WAV', '', 'WAV 音频 (*.wav *.WAV)')
        if path:
            self.load_path(path)

    def _choose_save(self):
        if self.current_clip is None:
            return
        name = Path(self.clip.name).stem + f'_{self.stage_name}.wav'
        path, _ = QFileDialog.getSaveFileName(self, f'保存{self.stage_name}', name, 'WAV 音频 (*.wav)')
        if path:
            if Path(path).suffix.lower() != '.wav':
                self._error('请使用 .wav 扩展名保存文件。')
                return
            self.save_to(path)

    def _play(self):
        try:
            self.player.play()
        except PlaybackError as exc:
            self._error(str(exc))

    def _choose_output(self):
        self.player.set_output_device(self.output_combo.currentData())
        self.output_combo.setToolTip(self.output_combo.currentText())
        self._status('输出设备已切换，点击播放从头开始。')

    def _output_fallback(self, name):
        # 自动回退只更新展示，不能触发 _choose_output 再次停止刚开始的播放。
        with QSignalBlocker(self.output_combo):
            index = self.output_combo.findData(name)
            if index < 0:
                self.output_combo.addItem(name, name)
                index = self.output_combo.count()-1
            self.output_combo.setCurrentIndex(index)
        self.output_combo.setToolTip(name)
        self.last_error = ''
        self._status(f'系统默认输出不可用，已切换至 {name} 播放。')

    def _playback_state(self, state):
        if hasattr(self, 'play_button'):
            self._controls()
            messages = {'playing': f'正在播放{self.stage_name}', 'paused': '已暂停，点击继续从当前位置播放', 'finished': '播放结束，可重新播放'}
            if state in messages:
                self._status(messages[state])

    def _position(self, position):
        duration = self.player.duration_ms
        self.progress.setValue(round(position/duration*1000) if duration else 0)
        self.progress_label.setText(f'{position/1000:.2f} / {duration/1000:.2f} s')

    def _status(self, message):
        self.status_label.setStyleSheet('color: #1771cb; padding: 6px;')
        self.status_label.setText('状态：'+message)

    def _error(self, message):
        self.last_error = message
        self.status_label.setStyleSheet('color: #b42318; padding: 6px;')
        self.status_label.setText('错误：'+message)

    def closeEvent(self, event):
        self.recorder.close()
        self._recording_target = None
        if self.busy:
            self._close_pending = True
            self._status('正在完成文件任务，完成后自动关闭…')
            event.ignore()
            return
        self.player.close()
        super().closeEvent(event)
