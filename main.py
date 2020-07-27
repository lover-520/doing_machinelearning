# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow

from GUI import algorithm_ui


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = algorithm_ui.AlgorithmUI()
    ui.show()
    sys.exit(app.exec_())
