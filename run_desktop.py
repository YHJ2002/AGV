"""Desktop entry point for the pure PyQt WareRover UI."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepend_env_path(path: Path) -> None:
    if not path.is_dir():
        return
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    path_str = str(path)
    if path_str not in parts:
        os.environ["PATH"] = path_str if not current else f"{path_str}{os.pathsep}{current}"


def _configure_qt_runtime() -> None:
    """Help Qt find its Windows platform plugin and dependent DLLs."""
    candidates: list[tuple[str, str, str]] = [
        ("PyQt5", "Qt5", "plugins"),
        ("PySide6", "", "plugins"),
    ]

    for package_name, qt_dir_name, plugin_dir_name in candidates:
        package_dir = Path(sys.prefix) / "Lib" / "site-packages" / package_name
        if not package_dir.exists():
            continue

        qt_root = package_dir / qt_dir_name if qt_dir_name else package_dir
        plugin_root = qt_root / plugin_dir_name
        platforms_dir = plugin_root / "platforms"
        bin_dir = qt_root / "bin"

        if platforms_dir.is_dir():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms_dir))
            os.environ.setdefault("QT_PLUGIN_PATH", str(plugin_root))

        _prepend_env_path(bin_dir)
        if hasattr(os, "add_dll_directory") and bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
        return


_configure_qt_runtime()

from gui.main_window import launch


if __name__ == "__main__":
    raise SystemExit(launch())
