from PyQt5.QtCore import QObject, QRunnable, pyqtSignal


class JobSignals(QObject):
    result = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()


class Job(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = JobSignals()

    def run(self):
        try:
            result = self.function()
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
