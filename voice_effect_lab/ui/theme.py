"""第一版浅色工作台的视觉标记、SVG图标与信息组件。"""
from functools import lru_cache
from pathlib import Path

from PyQt5.QtCore import QByteArray, Qt, QSize, QRect
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap, QPalette
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QPushButton, QSizePolicy

ASSETS = Path(__file__).resolve().parents[2]/'assets/icons'
ACCENT = '#008d98'
TEXT = '#20344d'

STYLE = r'''
QWidget { color: #20344d; font-family: "Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei"; font-size: 12px; }
QMainWindow, QWidget#app_root { background: #f4f7fa; }
QWidget[transparent="true"], QTabWidget, QTabWidget::pane { background: transparent; border: none; }
QGroupBox { background: #ffffff; border: 1px solid #e0e8ef; border-radius: 11px; margin-top: 9px; padding: 10px 10px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 4px; color: #20344d; }
QLabel { background: transparent; }
QFrame[card="true"] { background: white; border: 1px solid #e0e8ef; border-radius: 11px; }
QLabel[role="card_title"] { font-size: 13px; font-weight: 700; color: #20344d; }
QPushButton[role="preset"] { padding: 4px 4px; }
QLabel[role="muted"] { color: #718299; font-size: 11px; }
QLabel[role="badge"] { background: #edf3f7; color: #4d6580; border-radius: 5px; padding: 3px 6px; }
QLabel[role="online"] { background: #e6f5ed; color: #29805c; border-radius: 12px; padding: 5px 12px; }
QLabel[role="tag"] { background: #e9f5f8; color: #14869a; border-radius: 5px; padding: 3px 8px; font-size: 11px; }
QLabel#app_title { font-size: 24px; font-weight: 700; color: #18304c; }
QLabel#app_subtitle { color: #6b7f98; font-size: 11px; }
QLabel#chart_title { font-size: 19px; font-weight: 700; color: #18304c; }
QPushButton { border: 1px solid #d8e2eb; border-radius: 7px; background: #fbfcfe; padding: 5px 9px; min-height: 18px; }
QPushButton:hover { background: #edf8f8; border-color: #91c8cd; }
QPushButton:pressed { background: #deeff1; }
QPushButton:focus { border-color: #008d98; }
QPushButton[primary="true"] { background: #008d98; color: white; border-color: #008d98; font-weight: 600; }
QPushButton[primary="true"]:hover { background: #007f8a; border-color: #007f8a; }
QPushButton[primary="true"]:pressed { background: #006f7c; }
QPushButton:checked { background: #e1f3f5; color: #007a86; border-color: #77bbc4; }
QPushButton:disabled { color: #a3afbd; background: #f1f4f7; border-color: #e3e9ef; }
QPushButton[role="link"] { color: #157c92; background: transparent; border: none; padding: 2px 3px; }
QPushButton[role="icon"] { padding: 0; border-radius: 17px; }
QPushButton#play_main { border-radius: 23px; }
QFrame#main_navigation { background: #edf3f7; border: 1px solid #dbe5ed; border-radius: 10px; }
QFrame#main_navigation QPushButton { border: none; background: transparent; color: #64758d; padding: 7px 18px; min-width: 90px; }
QFrame#main_navigation QPushButton:checked { background: #008d98; color: white; }
QFrame#stage_segments { background: #f1f5f8; border: 1px solid #e4ebf0; border-radius: 7px; }
QRadioButton[segment="true"] { spacing: 0; padding: 4px 5px; border-radius: 5px; color: #687b91; }
QRadioButton[segment="true"]::indicator { width: 0; height: 0; }
QRadioButton[segment="true"]:checked { background: #008d98; color: white; }
QRadioButton[segment="true"]:disabled { color: #aab5c1; }
QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit { border: 1px solid #dbe4ec; border-radius: 6px; padding: 4px 6px; background: #fbfcfe; selection-background-color: #d6f0f2; selection-color: #1c5262; }
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QPlainTextEdit:focus { border-color: #6bb8c1; }
QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled { background: #f5f7f9; color: #9eabbc; border-color: #e7edf2; }
QComboBox { padding-right: 23px; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox::down-arrow { image: url("__DOWN__"); width: 11px; height: 11px; }
QComboBox QAbstractItemView { background: white; border: 1px solid #dbe4ec; selection-background-color: #e0f3f5; selection-color: #087583; padding: 4px; outline: none; }
QDoubleSpinBox, QSpinBox { padding-right: 17px; }
QDoubleSpinBox::up-button, QSpinBox::up-button { width: 17px; border: none; }
QDoubleSpinBox::down-button, QSpinBox::down-button { width: 17px; border: none; }
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow { image: url("__UP__"); width: 9px; height: 7px; }
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow { image: url("__DOWN__"); width: 9px; height: 7px; }
QCheckBox, QRadioButton { spacing: 6px; font-weight: 400; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #cbd8e2; border-radius: 3px; background: white; }
QCheckBox::indicator:checked { background: #008d98; border-color: #008d98; image: url("__CHECK__"); }
QCheckBox::indicator:disabled { background: #eef2f6; border-color: #dbe3eb; }
QCheckBox[switch="true"]::indicator { width: 31px; height: 18px; border: none; background: transparent; image: url("__OFF__"); }
QCheckBox[switch="true"]::indicator:checked { image: url("__ON__"); }
QCheckBox:disabled, QRadioButton:disabled { color: #9daabc; }
QSlider::groove:horizontal { height: 5px; border-radius: 2px; background: #dce5ec; }
QSlider::sub-page:horizontal { background: #008d98; border-radius: 2px; }
QSlider::handle:horizontal { width: 13px; margin: -5px 0; border: 1px solid #b8c9d6; border-radius: 7px; background: #ffffff; }
QSlider::handle:horizontal:hover { border-color: #008d98; }
QSlider::sub-page:horizontal:disabled { background: #cad8e3; }
QProgressBar { border: none; background: #dfe7ee; border-radius: 3px; min-height: 5px; max-height: 5px; }
QProgressBar::chunk { background: #008d98; border-radius: 3px; }
QTabWidget#extension_tabs::pane { background: transparent; border: none; }
QTabBar::tab { background: transparent; color: #6b7c92; padding: 7px 18px; margin: 1px 4px 6px 0; border-radius: 7px; }
QTabBar::tab:selected { color: #007d88; background: #e1f1f3; }
QTabBar::tab:hover { background: #edf4f7; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 6px; margin: 0; }
QScrollBar::handle:vertical { background: #c6d4df; border-radius: 3px; min-height: 32px; }
QScrollBar::handle:vertical:hover { background: #9eb7c7; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { height: 6px; background: transparent; }
QScrollBar::handle:horizontal { min-width: 28px; background: #c6d4df; border-radius: 3px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QSplitter::handle { background: transparent; width: 10px; }
QListWidget { border: 1px solid #e0e8ef; background: #f9fbfd; border-radius: 7px; padding: 5px; }
QListWidget::item { padding: 4px 8px; border-radius: 4px; }
QListWidget::item:selected { color: #007b86; background: #e1f2f4; }
QToolTip { background: #24374c; color: white; border: none; padding: 6px; }
QLabel#status_message { color: #678096; font-size: 11px; padding: 3px 0; }
QLabel#status_message[tone="error"] { color: #b93838; }
'''
for token,name in [('DOWN','chevron-down'),('UP','chevron-up'),('CHECK','check'),('ON','switch-on'),('OFF','switch-off')]:
    STYLE = STYLE.replace('__'+token+'__',str(ASSETS/(name+'.svg')))

PATHS = {
 'folder': '<path d="M3 7h6l2-3h9a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/>',
 'mic': '<rect x="8" y="2" width="8" height="13" rx="4"/><path d="M5 11v1a7 7 0 0 0 14 0v-1M12 19v3M8 22h8"/>',
 'play': '<path d="m8 4 13 8-13 8z" fill="currentColor" stroke-linejoin="round"/>',
 'pause': '<path d="M8 5v14M16 5v14" stroke-width="3"/>',
 'stop': '<rect x="5" y="5" width="14" height="14" rx="2" fill="currentColor"/>',
 'save': '<path d="M12 2v13m-5-5 5 5 5-5M4 15v5h16v-5"/>',
 'reset': '<path d="M4 10a8 8 0 1 1 1 8M4 4v6h6"/>',
 'wand': '<path d="m4 20 11-11 3 3L7 23zM17 1v4m-2-2h4M5 3v6m-3-3h6M20 15v5m-2-2h5"/>',
 'settings': '<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2" fill="white"/><circle cx="15" cy="12" r="2" fill="white"/><circle cx="8" cy="18" r="2" fill="white"/>',
 'volume': '<path d="M3 9h4l5-5v16l-5-5H3zM16 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14"/>',
 'muted': '<path d="M3 9h4l5-5v16l-5-5H3zM17 9l5 6m0-6-5 6"/>',
 'send': '<path d="m3 3 19 9-19 9 4-9zm4 9h15"/>',
 'chart': '<path d="M3 3v18h19M6 16l4-6 4 3 7-9"/>',
 'logo': '<path d="M2 12v1M6 8v9M10 3v19M14 6v13M18 9v7M22 12v1" stroke-width="2.5"/>',
}


@lru_cache(maxsize=64)
def icon(name, color=TEXT):
    body=PATHS.get(name,PATHS['settings']).replace('currentColor',color)
    source=f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    renderer=QSvgRenderer(QByteArray(source.encode()))
    result=QIcon()
    for size in (16,24,32,48,64):
        pixmap=QPixmap(size,size)
        pixmap.fill(Qt.transparent)
        painter=QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        result.addPixmap(pixmap)
    return result


def decorate_buttons(root):
    mapping=[('播放','play'),('继续','play'),('试听','play'),('暂停','pause'),('停止','stop'),
             ('保存','save'),('导入','folder'),('载入','folder'),('录音','mic'),('底噪','mic'),
             ('重置','reset'),('送入','send'),('查看','chart'),('应用','wand'),('处理','settings'),
             ('生成','wand'),('训练','wand')]
    for button in root.findChildren(QPushButton):
        button.setCursor(Qt.PointingHandCursor)
        if button.property('iconOnly'):
            continue
        text=button.text().lstrip('▶Ⅱ■↑↓ ')
        for word,name in mapping:
            if word in text:
                button.setIcon(icon(name,'#ffffff' if button.property('primary') else TEXT))
                button.setIconSize(QSize(16,16))
                break
        button.setText(text)


class AudioMetadata(QWidget):
    """三个徽标仅显示已加载音频的真实信息，同时保留可读描述。"""
    def __init__(self,parent=None):
        super().__init__(parent)
        layout=QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(6)
        self.rate,self.duration,self.channels=[QLabel('—') for _ in range(3)]
        for label in (self.rate,self.duration,self.channels):
            label.setProperty('role','badge')
            label.setAlignment(Qt.AlignCenter)
            label.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
            layout.addWidget(label,1)
        self._description='采样率：—    时长：—'

    def text(self):
        return self._description

    def set_clip(self,clip):
        self.rate.setText(f'{clip.sample_rate/1000:g} kHz' if clip.sample_rate>=1000 else f'{clip.sample_rate} Hz')
        self.duration.setText(f'{clip.duration:.2f} s')
        self.channels.setText('单声道' if clip.source_channels==1 else f'{clip.source_channels} → 1 声道')
        self._description=f'采样率：{clip.sample_rate} Hz    时长：{clip.duration:.2f} s\n源音频 {clip.source_channels} 声道 → 当前单声道'
        self.setToolTip(self._description)


def status_tone(label,error=False):
    label.setProperty('tone','error' if error else 'normal')
    label.style().unpolish(label)
    label.style().polish(label)
    label.update()


class RoundButton(QPushButton):
    """沿用Qt按钮交互，圆形绘制避免平台原生按钮背景破坏播放区。"""
    def __init__(self,name,primary=False,parent=None):
        super().__init__(parent)
        self.setProperty('primary',primary)
        self.setProperty('iconOnly',True)
        self.setIcon(icon(name,'#ffffff' if primary else TEXT))
        self.setIconSize(QSize(16,16))
        self.setFixedSize(34,34)
        self.setCursor(Qt.PointingHandCursor)

    def setFixedSize(self,*args):
        width,height=(args[0].width(),args[0].height()) if len(args)==1 else args
        super().setFixedSize(width,height)
        self.setStyleSheet(f'min-width:{width}px; max-width:{width}px; min-height:{height}px; max-height:{height}px; padding:0; border:none;')

    def paintEvent(self,event):
        painter=QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        primary=bool(self.property('primary'))
        fill=('#007c87' if self.isDown() else ACCENT) if primary else ('#e4f2f5' if self.isDown() or self.isChecked() else '#ffffff')
        if not self.isEnabled():
            fill='#eef2f6'
        border=fill if primary else '#d8e3eb'
        if self.hasFocus():
            border=ACCENT
        painter.setPen(QColor(border))
        painter.setBrush(QColor(fill))
        painter.drawEllipse(1,1,self.width()-2,self.height()-2)
        size=self.iconSize().width()
        rect=QRect((self.width()-size)//2,(self.height()-size)//2,size,size)
        self.icon().paint(painter,rect,Qt.AlignCenter,QIcon.Normal if self.isEnabled() else QIcon.Disabled)


class CompactLabel(QLabel):
    """短状态行绘制省略号，完整文字仍由text()和悬停提示提供。"""
    def __init__(self,text='',parent=None):
        super().__init__(text,parent)
        self.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        self.setMaximumHeight(18)
        self.setToolTip(text)

    def setText(self,text):
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self,event):
        painter=QPainter(self)
        painter.setPen(self.palette().color(QPalette.WindowText))
        rect=self.contentsRect()
        text=self.fontMetrics().elidedText(self.text(),Qt.ElideRight,rect.width())
        painter.drawText(rect,self.alignment() | Qt.AlignVCenter,text)
