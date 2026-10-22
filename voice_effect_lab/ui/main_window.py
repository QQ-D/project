from pathlib import Path

from PyQt5.QtCore import Qt, QSize, QThreadPool, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QScrollArea, QSplitter, QSlider, QDoubleSpinBox,
    QCheckBox, QRadioButton, QButtonGroup, QProgressBar, QFileDialog,
    QTabWidget, QComboBox, QFrame, QSizePolicy,
)
from PyQt5.QtMultimedia import QAudio, QAudioDeviceInfo

from ..audio.data import load_wav, save_wav
from ..audio.player import AudioPlayer, PlaybackError
from ..analysis import analyze
from .worker import Job
from .plots import AnalysisPlots
from .capture_controller import CaptureController
from .processing_controller import ProcessingController
from .enhancement_controller import EnhancementController
from .effects_controller import EffectsController
from .advanced_controls import AdvancedControls
from .tts_controller import TtsController
from .neural_controller import NeuralController
from .theme import STYLE, ACCENT, TEXT, AudioMetadata, RoundButton, CompactLabel, icon, decorate_buttons, status_tone


def _button(text, primary=False):
    button = QPushButton(text)
    button.setProperty('primary', primary)
    return button


def _group(title, disabled=False):
    widget = QFrame()
    widget.setProperty('card', True)
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(6)
    if title:
        heading = QLabel(title)
        heading.setProperty('role', 'card_title')
        layout.addWidget(heading)
    if disabled:
        widget.setEnabled(False)
        widget.setToolTip('后续阶段实现，目前仅展示设计入口。')
    return widget, layout


def _row(layout, *widgets):
    row = QHBoxLayout()
    for widget in widgets:
        row.addWidget(widget)
    layout.addLayout(row)


def _parameter(layout, label, low, high, value, suffix='', decimals=2, extra=None, compact=False):
    text = QLabel(label)
    text.setMinimumWidth(36 if compact else 80)
    slider = QSlider(Qt.Horizontal)
    scale = 10**decimals
    slider.setRange(round(low*scale), round(high*scale))
    slider.setValue(round(value*scale))
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setSuffix(suffix)
    spin.setValue(value)
    spin.setFixedWidth(68 if compact else 90)
    slider.valueChanged.connect(lambda v: spin.setValue(v/scale))
    spin.valueChanged.connect(lambda v: slider.setValue(round(v*scale)))
    _row(layout, text, slider, spin, *([extra] if extra is not None else []))
    return slider, spin


class MainWindow(QMainWindow, CaptureController, ProcessingController, EnhancementController, EffectsController, AdvancedControls, TtsController, NeuralController):
    loaded = pyqtSignal()

    def __init__(self, parent=None, recorder=None):
        super().__init__(parent)
        self.setWindowTitle('智能人声音效台 · 数字信号处理综合实验')
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
        self._init_enhancement()
        self._init_effects()
        self._init_tts()
        self._init_neural()
        self._build()
        self._controls()

    def _build(self):
        root = QWidget()
        root.setObjectName('app_root')
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 9)
        outer.setSpacing(12)
        self.tabs = QTabWidget()
        self.tabs.tabBar().hide()
        header = QHBoxLayout()
        header.setSpacing(12)
        logo = QLabel()
        logo.setPixmap(icon('logo', ACCENT).pixmap(44, 44))
        header.addWidget(logo)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        title = QLabel('智能人声音效台')
        title.setObjectName('app_title')
        subtitle = QLabel('数字信号处理综合实验')
        subtitle.setObjectName('app_subtitle')
        brand.addWidget(title)
        brand.addWidget(subtitle)
        header.addLayout(brand)
        header.addStretch()
        nav = QFrame()
        nav.setObjectName('main_navigation')
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(4, 4, 4, 4)
        nav_layout.setSpacing(4)
        self.main_navigation = QButtonGroup(self)
        for index, text in enumerate(('基础工作台', '拓展功能')):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.setCursor(Qt.PointingHandCursor)
            nav_layout.addWidget(button)
            self.main_navigation.addButton(button, index)
        self.main_navigation.buttonClicked[int].connect(self.tabs.setCurrentIndex)
        nav.setFixedHeight(44)
        header.addWidget(nav)
        header.addStretch()
        online = QLabel('●  本地处理')
        online.setProperty('role', 'online')
        online.setFixedHeight(28)
        header.addWidget(online)
        self.demo_button = _button('载入测试音')
        self.demo_button.clicked.connect(self.load_demo)
        header.addWidget(self.demo_button)
        outer.addLayout(header)
        outer.addWidget(self.tabs, 1)
        basic = QWidget()
        basic.setProperty('transparent', True)
        layout = QHBoxLayout(basic)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        controls = QWidget()
        controls.setProperty('transparent', True)
        left = QVBoxLayout(controls)
        left.setContentsMargins(0, 0, 4, 0)
        left.setSpacing(10)
        self._source_panel(left)
        self._future_panels(left)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.process_button = _button('开始处理', True)
        self.process_button.setEnabled(False)
        self.process_button.setToolTip('依次执行所启用的增强和音效，可选使用去噪输入；未启用增强和音效时单独去噪。')
        self.process_button.clicked.connect(self.start_processing)
        self.reset_button = _button('重置')
        self.reset_button.clicked.connect(self.reset_processing)
        self.reset_button.setToolTip('恢复基础默认参数、清空处理缓存；保留原声与参考底噪。')
        self.save_button = _button('保存原声 WAV')
        self.save_button.clicked.connect(self._choose_save)
        actions.addWidget(self.process_button, 2)
        actions.addWidget(self.reset_button, 1)
        actions.addWidget(self.save_button, 2)
        left.addStretch()
        scroll = self.control_scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setFrameShape(QScrollArea.NoFrame)
        sidebar = QWidget()
        sidebar.setProperty('transparent', True)
        sidebar.setMinimumWidth(365)
        sidebar.setMaximumWidth(460)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)
        side_layout.addWidget(scroll, 1)
        self._listen_panel(side_layout)
        self.transport_layout.addLayout(actions)
        splitter.addWidget(sidebar)
        right, plots_layout = _group('')
        right.setObjectName('analysis_card')
        plot_header = QHBoxLayout()
        chart_title = QLabel('声音分析')
        chart_title.setObjectName('chart_title')
        plot_header.addWidget(chart_title)
        tag = QLabel('原声始终保留')
        tag.setProperty('role', 'tag')
        plot_header.addWidget(tag)
        plot_header.addStretch()
        original = QLabel('●  原声')
        original.setStyleSheet('color: #1688e8;')
        processed = QLabel('●  处理结果')
        processed.setStyleSheet('color: #e99043;')
        plot_header.addWidget(original)
        plot_header.addWidget(processed)
        plots_layout.addLayout(plot_header)
        self.file_label = QLabel('导入或录制音频后，从这里比较每个处理阶段。')
        self.file_label.setProperty('role', 'muted')
        self.file_label.setWordWrap(True)
        plots_layout.addWidget(self.file_label)
        self.plots = AnalysisPlots(right)
        plots_layout.addWidget(self.plots, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([435, 1050])
        self.tabs.addTab(basic, '基础工作台 · 必做')
        self.tabs.addTab(self._extensions(), '拓展功能 · 音效 / TTS / 智能降噪')
        self.tabs.currentChanged.connect(self._preview_tab_changed)
        self.status_label = QLabel('状态：导入原声后，可去噪、增强或应用音效。')
        self.status_label.setObjectName('status_message')
        self.status_label.setWordWrap(True)
        self.status_label.setMaximumHeight(36)
        status_tone(self.status_label)
        outer.addWidget(self.status_label)
        for name in ('neural_scope_label','neural_unavailable_label','neural_baseline_label',
                     'neural_training_label','neural_volume_note','neural_send_note','neural_progress_label',
                     'tts_count_label','tts_progress_label','tts_engine_label'):
            getattr(self,name).setProperty('role','muted')
        self.neural_metrics_label.setProperty('role','badge')
        self.device_details_button.setIcon(icon('settings'))
        decorate_buttons(root)

    def _source_panel(self, left):
        group, layout = _group('01  音频采集')
        self.record_button = _button('开始录音', True)
        self.record_button.clicked.connect(lambda: self.start_recording('voice'))
        self.end_record_button = _button('结束录音')
        self.end_record_button.clicked.connect(self.stop_recording)
        self.import_button = _button('导入 WAV')
        self.import_button.clicked.connect(self._choose_load)
        _row(layout, self.record_button, self.end_record_button, self.import_button)
        self.info_label = AudioMetadata()
        self.rate_badge, self.duration_badge, self.channel_badge = self.info_label.rate, self.info_label.duration, self.info_label.channels
        layout.addWidget(self.info_label)
        self.input_combo = QComboBox()
        self.input_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.input_combo.setMinimumContentsLength(10)
        self.input_combo.addItem('系统默认输入', None)
        for device in QAudioDeviceInfo.availableDevices(QAudio.AudioInput):
            self.input_combo.addItem(device.deviceName(), device.deviceName())
        self.input_combo.currentIndexChanged.connect(self._choose_input)
        self.device_details_button = _button('设备')
        self.device_details_button.setProperty('role', 'link')
        self.device_details_button.setCheckable(True)
        self.device_details_button.setToolTip('展开目标采样率与输出设备；语音最长300秒，底噪约1.5秒。')
        self.device_details_panel = QWidget()
        details = QVBoxLayout(self.device_details_panel)
        details.setContentsMargins(0, 2, 0, 0)
        self.sample_rate_combo = QComboBox()
        for rate in [16000, 44100, 48000]:
            self.sample_rate_combo.addItem(f'{rate} Hz', rate)
        self.sample_rate_combo.setCurrentIndex(1)
        _row(details, QLabel('目标采样率'), self.sample_rate_combo)
        self.output_combo = QComboBox()
        self.output_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.output_combo.setMinimumContentsLength(12)
        self.output_combo.addItem('自动选择（优先系统默认）', None)
        for device in QAudioDeviceInfo.availableDevices(QAudio.AudioOutput):
            self.output_combo.addItem(device.deviceName(), device.deviceName())
        self.output_combo.currentIndexChanged.connect(self._choose_output)
        self.output_combo.setToolTip('自动尝试默认输出和独立声卡路由，也可手动选择扬声器或耳机。')
        _row(details, QLabel('输出设备'), self.output_combo)
        self.device_details_panel.hide()
        self.device_details_button.toggled.connect(self.device_details_panel.setVisible)
        _row(layout, self.input_combo, self.device_details_button)
        layout.addWidget(self.device_details_panel)
        self.capture_label = QLabel('')
        self.capture_label.setProperty('role', 'muted')
        self.capture_label.setWordWrap(True)
        self.cancel_capture_button = _button('取消采集')
        self.cancel_capture_button.setProperty('role', 'link')
        self.cancel_capture_button.clicked.connect(self.cancel_recording)
        _row(layout, self.capture_label, self.cancel_capture_button)
        left.addWidget(group)

    def _future_panels(self, left):
        group, layout = _group('02  传统去噪')
        self.denoise_method_combo = QComboBox()
        self.denoise_method_combo.addItem('自适应维纳滤波','wiener')
        self.denoise_method_combo.addItem('功率谱减法','spectral')
        self.denoise_method_combo.setToolTip('预训练模型和BP训练在拓展页的智能降噪。')
        self.denoise_method_combo.currentIndexChanged.connect(self._invalidate_denoise)
        _row(layout, QLabel('去噪方法'), self.denoise_method_combo)
        self.denoise_engine_hint = QLabel('', group)
        self.denoise_engine_hint.hide()
        self.noise_reference_panel = QWidget()
        reference_layout = QVBoxLayout(self.noise_reference_panel)
        reference_layout.setContentsMargins(0,0,0,0)
        reference_layout.setSpacing(5)
        self.auto_noise_check = QCheckBox('自动估计底噪')
        self.auto_noise_check.setChecked(True)
        self.auto_noise_check.setToolTip('无需参考文件；从低能量帧初始化背景谱，估计帧可能含人声。复杂噪声可能需要纯底噪参考。')
        self.auto_noise_check.toggled.connect(self._noise_mode_changed)
        self.noise_button = _button('采集底噪')
        self.noise_button.clicked.connect(lambda: self.start_recording('noise'))
        self.import_noise_button = _button('导入底噪')
        self.import_noise_button.clicked.connect(self._choose_noise_file)
        self.import_noise_button.setToolTip('导入同环境的纯噪声WAV，不会替换原声。')
        _row(reference_layout, self.auto_noise_check, self.noise_button, self.import_noise_button)
        self.noise_details_panel = QWidget()
        details = QVBoxLayout(self.noise_details_panel)
        details.setContentsMargins(0,0,0,0)
        self.noise_status_label = QLabel('底噪：自动估计，无需额外文件；估计帧可能含人声')
        self.noise_status_label.setProperty('role', 'muted')
        self.noise_status_label.setWordWrap(True)
        details.addWidget(self.noise_status_label)
        self.confirm_noise_button = _button('确认底噪适用')
        self.confirm_noise_button.setToolTip('确认参考只含噪声，没有人声，且与当前噪声环境匹配。')
        self.confirm_noise_button.clicked.connect(self.confirm_noise)
        self.clear_noise_button = _button('清除底噪')
        self.clear_noise_button.clicked.connect(self.clear_noise)
        _row(details, self.confirm_noise_button, self.clear_noise_button)
        reference_layout.addWidget(self.noise_details_panel)
        layout.addWidget(self.noise_reference_panel)
        self.denoise_parameters = QWidget()
        parameters = QVBoxLayout(self.denoise_parameters)
        parameters.setContentsMargins(0,0,0,0)
        self.denoise_slider, self.denoise_spin = _parameter(parameters, '去噪强度', 0, 100, 60, ' %', 0)
        self.denoise_spin.setToolTip('0%精确旁路；抑制强度越高，可能损失更多人声。')
        self.denoise_spin.valueChanged.connect(self._invalidate_denoise)
        layout.addWidget(self.denoise_parameters)
        self.denoise_button = _button('一键去噪', True)
        self.denoise_button.clicked.connect(self.start_denoise)
        self.select_noise_button = _button('选取噪声段')
        self.select_noise_button.setProperty('role', 'link')
        self.select_noise_button.clicked.connect(self._choose_noise)
        self.use_denoise_check = QCheckBox('后续使用')
        self.use_denoise_check.setToolTip('增强/音效使用有效去噪缓存；没有缓存时先去噪。取消时从原声开始。')
        self.use_denoise_check.toggled.connect(self._upstream_policy_changed)
        self.denoise_status_label = CompactLabel('去噪：尚未处理')
        self.denoise_status_label.setProperty('role', 'muted')
        _row(layout,self.denoise_button,self.select_noise_button,self.use_denoise_check)
        layout.addWidget(self.denoise_status_label)
        left.addWidget(group)
        group, layout = _group('03  声音增强')
        self.enhance_group = group
        self.enhance_check = QCheckBox('启用增强')
        self.enhance_check.setProperty('switch', True)
        self.enhance_check.toggled.connect(self._enhancement_parameter_changed)
        self.normalize_check = QCheckBox('峰值归一化')
        self.normalize_check.setChecked(True)
        self.normalize_check.setToolTip('目标峰值0.95，之后应用音量增益。')
        self.normalize_check.toggled.connect(self._enhancement_parameter_changed)
        _row(layout, self.enhance_check, self.normalize_check)
        self.enhance_parameters = QWidget()
        params = QVBoxLayout(self.enhance_parameters)
        params.setContentsMargins(0,0,0,0)
        params.setSpacing(5)
        self.gain_slider, self.gain_spin = _parameter(params,'音量增益',0,2,1,'×')
        self.clarity_slider, self.clarity_spin = _parameter(params,'人声清晰度',0,100,0,' %',0)
        eq_row = QHBoxLayout()
        for name, title in (('low','低频'),('mid','中频'),('high','高频')):
            band = QVBoxLayout()
            band.setSpacing(1)
            label = QLabel(title)
            label.setAlignment(Qt.AlignCenter)
            label.setProperty('role','muted')
            band.addWidget(label)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-1200,1200)
            slider.setFixedHeight(16)
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(-12,12)
            spin.setSuffix(' dB')
            spin.setStyleSheet('padding-top:2px; padding-bottom:2px;')
            spin.setFixedHeight(26)
            slider.valueChanged.connect(lambda value, spin=spin: spin.setValue(value/100))
            spin.valueChanged.connect(lambda value, slider=slider: slider.setValue(round(value*100)))
            band.addWidget(slider)
            band.addWidget(spin)
            eq_row.addLayout(band,1)
            setattr(self,'eq_'+name+'_slider',slider)
            setattr(self,'eq_'+name+'_spin',spin)
        params.addLayout(eq_row)
        for spin in (self.gain_spin,self.clarity_spin,self.eq_low_spin,self.eq_mid_spin,self.eq_high_spin):
            spin.valueChanged.connect(self._enhancement_parameter_changed)
        layout.addWidget(self.enhance_parameters)
        self.enhance_button = _button('一键增强')
        self.enhance_button.clicked.connect(self.start_enhance)
        self.enhance_status_label = CompactLabel('增强：待处理')
        self.enhance_status_label.setProperty('role','muted')
        heading = layout.takeAt(0).widget()
        top = QHBoxLayout()
        top.addWidget(heading,1)
        top.addWidget(self.enhance_button)
        layout.insertLayout(0,top)
        layout.addWidget(self.enhance_status_label)
        left.addWidget(group)
        self._effects_panels(left)

    def _effects_panels(self, left):
        group, layout = _group('04  音效预设')
        self.presets_group = self.effects_group = group
        self.preset_buttons = {}
        grid = QGridLayout()
        grid.setSpacing(4)
        for index, name in enumerate(['原声','磁性','娃娃音','大叔音','电话音','山谷回声','自定义']):
            button = _button(name)
            button.setCheckable(True)
            button.setChecked(name == '原声')
            button.setProperty('role','preset')
            button.clicked.connect(lambda checked,label=name:self.select_preset(label))
            self.preset_buttons[name] = button
            grid.addWidget(button,index//4,index%4)
        layout.addLayout(grid)
        self.preset_label = CompactLabel('当前预设：原声 · 音效旁路')
        self.preset_label.setProperty('role','muted')
        self.effects_check = QCheckBox('启用音效')
        self.effects_check.setProperty('switch',True)
        self.effects_check.toggled.connect(self._effect_parameter_changed)
        self.effects_details_button = _button('更多')
        self.effects_details_button.setToolTip('电话滤波、回声延时、应用音效与处理状态。')
        self.effects_details_button.setProperty('role','link')
        self.effects_details_button.setCheckable(True)
        heading = layout.takeAt(0).widget()
        card_header = QHBoxLayout()
        card_header.addWidget(heading,1)
        card_header.addWidget(self.effects_check)
        card_header.addWidget(self.effects_details_button)
        layout.insertLayout(0,card_header)
        self.effects_parameters = QWidget()
        parameters = QHBoxLayout(self.effects_parameters)
        parameters.setContentsMargins(0,0,0,0)
        parameters.setSpacing(10)
        rate = QVBoxLayout()
        strength = QVBoxLayout()
        self.rate_slider,self.rate_spin = _parameter(rate,'倍率',.5,2,1,'×',compact=True)
        self.rate_spin.setToolTip('同时改变音高和语速，实际样本与导出时长随倍率变化。')
        self.echo_strength_slider,self.echo_strength_spin = _parameter(strength,'回声',0,.8,0,compact=True)
        parameters.addLayout(rate,1)
        parameters.addLayout(strength,1)
        self.effects_details_panel = QWidget()
        details = QVBoxLayout(self.effects_details_panel)
        details.setContentsMargins(0,0,0,0)
        details.addWidget(self.preset_label)
        self.telephone_check = QCheckBox('电话滤波（约300–3400 Hz）')
        self.telephone_check.toggled.connect(self._effect_parameter_changed)
        details.addWidget(self.telephone_check)
        self.echo_delay_slider,self.echo_delay_spin = _parameter(details,'回声延时',0,1,.3,' s')
        self.echo_mode_label = QLabel('回声配置：单次')
        self.echo_mode_label.setProperty('role','muted')
        details.addWidget(self.echo_mode_label)
        self.effects_details_panel.hide()
        self.effects_details_button.toggled.connect(self.effects_details_panel.setVisible)
        layout.addWidget(self.effects_parameters)
        # The disclosure stays operable when effects are bypassed; values remain
        # in the same controls and the enable checkbox governs actual DSP.
        layout.addWidget(self.effects_details_panel)
        for spin in (self.rate_spin,self.echo_delay_spin,self.echo_strength_spin):
            spin.valueChanged.connect(self._effect_parameter_changed)
        self.effects_button = _button('应用音效')
        self.effects_button.clicked.connect(self.start_effects)
        self.effects_status_label = QLabel('音效：待处理')
        self.effects_status_label.setProperty('role','muted')
        self.effects_status_label.setWordWrap(True)
        _row(details,self.effects_status_label,self.effects_button)
        left.addWidget(group)

    def _listen_panel(self, left):
        group, layout = _group('播放与导出')
        self.transport_layout = layout
        segments = QFrame()
        segments.setObjectName('stage_segments')
        row = QHBoxLayout(segments)
        row.setContentsMargins(3,3,3,3)
        row.setSpacing(3)
        self.stages = QButtonGroup(self)
        for index,label in enumerate(['原声','去噪','增强','音效']):
            radio = QRadioButton(label)
            radio.setProperty('segment',True)
            radio.setChecked(index == 0)
            radio.setEnabled(index == 0)
            radio.setToolTip('参数或输入变化后需重新处理；切换后可试听并保存当前阶段。')
            self.stages.addButton(radio,index)
            row.addWidget(radio,1)
        heading = layout.takeAt(0).widget()
        transport_header = QHBoxLayout()
        transport_header.addWidget(heading)
        transport_header.addWidget(segments,1)
        layout.addLayout(transport_header)
        self.stages.buttonClicked[int].connect(self.select_stage)
        buttons = QHBoxLayout()
        buttons.setSpacing(7)
        self.play_button = RoundButton('play', primary=True)
        self.play_button.setObjectName('play_main')
        self.play_button.setFixedSize(46,46)
        self.play_button.setIcon(icon('play','#ffffff'))
        self.play_button.setIconSize(QSize(22,22))
        self.play_button.setProperty('iconOnly',True)
        self.play_button.clicked.connect(self._play)
        self.pause_button = RoundButton('pause')
        self.stop_button = RoundButton('stop')
        for button,name,title in ((self.pause_button,'pause','暂停'),(self.stop_button,'stop','停止')):
            button.setProperty('role','icon')
            button.setProperty('iconOnly',True)
            button.setFixedSize(34,34)
            button.setIcon(icon(name))
            button.setToolTip(title)
            button.setAccessibleName(title)
        self.pause_button.clicked.connect(self.player.pause)
        self.stop_button.clicked.connect(self._stop_action)
        self.mute_button = RoundButton('volume')
        self.mute_button.setProperty('iconOnly',True)
        self.mute_button.setProperty('role','icon')
        self.mute_button.setCheckable(True)
        self.mute_button.setFixedSize(28,28)
        self.mute_button.setIcon(icon('volume'))
        self.mute_button.setToolTip('静音')
        self.mute_button.setAccessibleName('静音')
        self.mute_button.toggled.connect(self._set_listen_muted)
        for widget in (self.play_button,self.pause_button,self.stop_button,self.mute_button):
            buttons.addWidget(widget)
        self.listen_volume_slider = QSlider(Qt.Horizontal)
        self.listen_volume_slider.setRange(0,100)
        self.listen_volume_slider.setValue(100)
        self.listen_volume_spin = QDoubleSpinBox()
        self.listen_volume_spin.setRange(0,100)
        self.listen_volume_spin.setDecimals(0)
        self.listen_volume_spin.setSuffix('%')
        self.listen_volume_spin.setValue(100)
        self.listen_volume_spin.setFixedWidth(72)
        self.listen_volume_slider.valueChanged.connect(self.listen_volume_spin.setValue)
        self.listen_volume_spin.valueChanged.connect(lambda value:self.listen_volume_slider.setValue(round(value)))
        self.listen_volume_spin.valueChanged.connect(self._set_listen_volume)
        self.listen_volume_slider.setToolTip('试听音量，不改变音频缓存与导出。')
        buttons.addWidget(self.listen_volume_slider,1)
        buttons.addWidget(self.listen_volume_spin)
        layout.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0,1000)
        self.progress_label = QLabel('0.00 / 0.00 s')
        self.progress_label.setProperty('role','muted')
        _row(layout,self.progress,self.progress_label)
        left.addWidget(group)

    def _multiecho_panel(self, layout):
        self.multiecho_panel = QWidget()
        controls = QVBoxLayout(self.multiecho_panel)
        controls.setContentsMargins(0,0,0,0)
        self.multiecho_check = QCheckBox('启用多重回声')
        self.multiecho_check.setProperty('switch',True)
        self.multiecho_check.toggled.connect(self._multiecho_changed)
        controls.addWidget(self.multiecho_check)
        self.multiecho_count_controls = QWidget()
        count_layout = QVBoxLayout(self.multiecho_count_controls)
        count_layout.setContentsMargins(0,0,0,0)
        self.multiecho_count_slider, self.multiecho_count_spin = _parameter(count_layout,'回声次数',1,8,3,' 次',0)
        self.multiecho_count_spin.valueChanged.connect(self._effect_parameter_changed)
        controls.addWidget(self.multiecho_count_controls)
        self.multiecho_delay_slider, self.multiecho_delay_spin = _parameter(controls,'回声间隔',0,1,.3,' s')
        self.multiecho_strength_slider, self.multiecho_strength_spin = _parameter(controls,'强度 / 衰减',0,.8,0)
        for extra, basic in [(self.multiecho_delay_spin,self.echo_delay_spin),
                             (self.multiecho_strength_spin,self.echo_strength_spin)]:
            extra.valueChanged.connect(basic.setValue)
            basic.valueChanged.connect(extra.setValue)
        note = QLabel('间隔、强度与主工作台共享；每次回声按强度系数继续衰减，保留全部尾音。')
        note.setWordWrap(True)
        controls.addWidget(note)
        self.multiecho_button = _button('应用多重回声并返回工作台',True)
        self.multiecho_button.clicked.connect(self.start_multiecho)
        controls.addWidget(self.multiecho_button)
        layout.addWidget(self.multiecho_panel)

    def _extensions(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 8, 0, 0)
        tabs = self.extension_tabs = QTabWidget()
        tabs.setObjectName('extension_tabs')
        outer.addWidget(tabs)

        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        note = QLabel('多重回声、独立变调/变速、混响、电音和压缩可排序串联；应用后返回工作台对比与保存。')
        note.setWordWrap(True)
        advanced_layout.addWidget(note)
        group, layout = _group('高级音效 · 可串联')
        content = QWidget()
        controls = QVBoxLayout(content)
        self._multiecho_panel(controls)
        controls.addWidget(self.build_advanced_panel(_parameter, _button))
        controls.addStretch()
        scroll = self.advanced_scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QScrollArea.NoFrame)
        layout.addWidget(scroll)
        advanced_layout.addWidget(group)
        tabs.addTab(advanced, '高级音效')

        tts = QWidget()
        tts_layout = QVBoxLayout(tts)
        group, layout = _group('中文 TTS · 离线合成与配音')
        group.setMaximumWidth(920)
        group.setMinimumWidth(690)
        self.tts_mute_button = _button('静音')
        self.tts_mute_button.setCheckable(True)
        self.tts_volume_slider, self.tts_volume_spin = _parameter(layout, '试听音量', 0, 100, 100, ' %', 0, extra=self.tts_mute_button)
        self.tts_volume_spin.valueChanged.connect(self.listen_volume_spin.setValue)
        self.listen_volume_spin.valueChanged.connect(self.tts_volume_spin.setValue)
        self.tts_mute_button.toggled.connect(self.mute_button.setChecked)
        self.mute_button.toggled.connect(self.tts_mute_button.setChecked)
        layout.addWidget(self.build_tts_panel())
        tts_layout.addWidget(group,1,Qt.AlignHCenter)
        tabs.addTab(tts, '中文 TTS')

        intelligent = QWidget()
        intelligent_layout = QHBoxLayout(intelligent)
        splitter = QSplitter(Qt.Horizontal)
        intelligent_layout.addWidget(splitter)
        sidebar, controls = _group('智能降噪 · 预训练 / 教学训练')
        sidebar.setMinimumWidth(375)
        sidebar.setMaximumWidth(470)
        self.neural_scroll = QScrollArea()
        self.neural_scroll.setWidgetResizable(True)
        self.neural_scroll.setFrameShape(QScrollArea.NoFrame)
        self.neural_scroll.setWidget(self.build_neural_panel())
        controls.addWidget(self.neural_scroll)
        self.neural_mute_button = _button('静音')
        self.neural_mute_button.setCheckable(True)
        self.neural_volume_slider, self.neural_volume_spin = _parameter(controls, '试听音量', 0, 100, 100, ' %', 0, extra=self.neural_mute_button)
        self.neural_volume_spin.valueChanged.connect(self.listen_volume_spin.setValue)
        self.listen_volume_spin.valueChanged.connect(self.neural_volume_spin.setValue)
        self.neural_mute_button.toggled.connect(self.mute_button.setChecked)
        self.mute_button.toggled.connect(self.neural_mute_button.setChecked)
        splitter.addWidget(sidebar)
        charts, chart_layout = _group('同一原声 · 传统方法 / 智能模型对比')
        note = QLabel('先选择模型运行比较，再切换所选结果。图形、试听和导出共用缓存；仅有干净参考时显示 SNR。')
        note.setWordWrap(True)
        chart_layout.addWidget(note)
        self.intelligent_plot_layout = chart_layout
        self.intelligent_plots = None
        splitter.addWidget(charts)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([415, 1050])
        tabs.addTab(intelligent, '智能降噪')
        tabs.currentChanged.connect(self._extension_preview_changed)
        return page

    def _extension_preview_changed(self, index):
        self.tts_player.stop()
        self.neural_player.stop()
        if index == 2:
            if self.intelligent_plots is None:
                self.intelligent_plots = AnalysisPlots()
                self.intelligent_plot_layout.addWidget(self.intelligent_plots, 1)
            self._neural_refresh_plots()

    def _set_listen_volume(self,value):
        self.player.set_volume(value)
        self.tts_player.set_volume(value)
        self.neural_player.set_volume(value)

    def _set_listen_muted(self,muted):
        self.player.set_muted(muted)
        self.tts_player.set_muted(muted)
        self.neural_player.set_muted(muted)
        self.mute_button.setIcon(icon('muted' if muted else 'volume'))
        self.mute_button.setToolTip('取消静音' if muted else '静音')
        self.mute_button.setAccessibleName(self.mute_button.toolTip())
        self.tts_mute_button.setText('取消静音' if muted else '静音')
        self.neural_mute_button.setText('取消静音' if muted else '静音')

    def _controls(self):
        if self.occupied and self.tts_player.state in ('playing','paused'):
            self.tts_player.stop()
        if self.occupied and self.neural_player.state in ('playing','paused'):
            self.neural_player.stop()
        available = self.current_clip is not None and not self.occupied
        self.import_button.setEnabled(not self.occupied)
        self.demo_button.setEnabled(not self.occupied)
        self.output_combo.setEnabled(not self.occupied)
        self.play_button.setEnabled(available and self.player.state != 'playing')
        self.play_button.setToolTip('继续播放' if self.player.state == 'paused' else '播放')
        self.play_button.setAccessibleName(self.play_button.toolTip())
        self.pause_button.setEnabled(available and self.player.state == 'playing')
        self.stop_button.setEnabled(available and self.player.state in ('playing', 'paused', 'finished'))
        self.save_button.setEnabled(available)
        self._capture_controls()
        self.device_details_button.setEnabled(not self.occupied)
        self.cancel_capture_button.setVisible(self.recorder.state == 'recording')
        self.capture_label.setVisible(self.recorder.state == 'recording' or bool(self.capture_label.text()))
        self.stop_button.setText('')
        self.stop_button.setToolTip('结束录音' if self.recorder.state == 'recording' else '停止播放')
        self.stop_button.setAccessibleName(self.stop_button.toolTip())
        self._processing_controls()
        self._tts_controls()
        self._neural_controls()

    def _preview_tab_changed(self,index):
        self.main_navigation.button(index).setChecked(True)
        if index == 1:
            self.player.stop()
        else:
            self.tts_player.stop()
            self.neural_player.stop()

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
        self._neural_source_changed()
        self._source_replaced(source_kind)
        self.player.set_clip(self.clip)
        self.info_label.set_clip(self.clip)
        self.file_label.setText(f'{self.clip.name}  |  频谱/语谱显示至 {min(8000, self.clip.sample_rate/2):g} Hz  |  当前阶段：原声')
        self.plots.set_source(self.clip, self.analysis)
        self._position(0)
        self._status('原声已加载，可去噪、增强或加载预设后应用音效；原声始终保留。')
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
        self.tts_player.stop()
        self.neural_player.stop()
        try:
            self.player.play()
        except PlaybackError as exc:
            self._error(str(exc))

    def _choose_output(self):
        self.tts_player.stop()
        self.neural_player.stop()
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
        status_tone(self.status_label)
        self.status_label.setText('●  '+message)
        self.status_label.setToolTip(message)

    def _error(self, message):
        self.last_error = message
        status_tone(self.status_label, error=True)
        self.status_label.setText('错误：'+message)
        self.status_label.setToolTip(message)

    def closeEvent(self, event):
        self.tts_player.stop()
        self.neural_player.stop()
        self.recorder.close()
        self._recording_target = None
        if self.busy:
            self._close_pending = True
            self._status('正在完成文件任务，完成后自动关闭…')
            event.ignore()
            return
        self.player.close()
        self.tts_player.close()
        self._close_neural()
        super().closeEvent(event)
