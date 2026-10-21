"""启动应用。--demo 使用明确标注的合成测试音，不冒充真人录音。"""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description='智能人声音效台 · DeepFilterNet3降噪、音效与中文TTS')
    parser.add_argument('--demo', action='store_true', help='载入附带的合成测试音')
    parser.add_argument('--file', type=Path, help='启动时导入指定 WAV')
    parser.add_argument('--screenshot', type=Path, help='保存实际 GUI 截图并退出；无显示环境时使用离屏模式')
    args = parser.parse_args()
    if args.demo and args.file:
        parser.error('--demo 与 --file 只能选择一项')
    root = Path(__file__).resolve().parent
    cache = root/'.runtime'
    cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache/'matplotlib'))
    if args.screenshot and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    app.setFont(QFont('Noto Sans CJK SC', 10))
    from voice_effect_lab.ui.main_window import MainWindow
    window = MainWindow()
    window.show()
    captured = False

    def capture():
        nonlocal captured
        if captured:
            return
        captured = True
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        window.plots.draw()
        if not window.grab().save(str(args.screenshot)):
            print('无法保存截图。', file=sys.stderr)
            app.exit(1)
            return
        print(f'已保存实际应用截图：{args.screenshot}')
        window.close()

    if args.screenshot:
        if args.demo or args.file:
            window.loaded.connect(lambda: QTimer.singleShot(350, capture))
        else:
            QTimer.singleShot(350, capture)
        def timeout():
            if not captured:
                print('截图失败：加载未完成。'+window.last_error, file=sys.stderr)
                window.close()
                app.exit(1)
        QTimer.singleShot(15000, timeout)
    if args.file:
        QTimer.singleShot(0, lambda: window.load_path(args.file))
    elif args.demo:
        QTimer.singleShot(0, window.load_demo)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
