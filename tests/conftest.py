import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/voice-lab-test-mpl')

import pytest
from PyQt5.QtWidgets import QApplication

@pytest.fixture(scope='session')
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()
