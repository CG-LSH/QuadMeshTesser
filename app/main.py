"""Application entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Prefer PyQt5 on Windows (PySide6/PyQt6 often hit DLL issues in some Anaconda setups)
os.environ.setdefault("QT_API", "pyqt5")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ensure relative paths (data/, output/) resolve when launched from elsewhere
try:
    os.chdir(_ROOT)
except OSError:
    pass

from qtpy.QtWidgets import QApplication

from app.window import MainWindow
from quadmeshtesser import __version__


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("QuadMeshTesser")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("QuadMeshTesser")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())


if __name__ == "__main__":
    main()