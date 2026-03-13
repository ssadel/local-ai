#!/usr/bin/env python3
"""Local AI Model Manager — list, start/stop, delete MLX models from HuggingFace cache."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
)
from PySide6.QtCore import Qt

# HuggingFace cache location
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
PORT_RANGE = range(8765, 8791)

# Prefer venv Python for mlx_lm
SCRIPT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"
MLX_PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def cache_dir_to_model_id(dirname: str) -> str | None:
    """Convert models--org--name to org/name."""
    if not dirname.startswith("models--"):
        return None
    rest = dirname[len("models--") :]
    if "--" not in rest:
        return None
    first_dash = rest.index("--")
    org = rest[:first_dash]
    name = rest[first_dash + 2 :]
    return f"{org}/{name}"


def model_id_to_cache_dir(model_id: str) -> str:
    """Convert org/name to models--org--name."""
    org, name = model_id.split("/", 1)
    return f"models--{org}--{name}"


def get_cached_models() -> list[tuple[str, int]]:
    """Scan HF cache for models. Returns [(model_id, size_bytes), ...]."""
    if not HF_HUB.exists():
        return []
    result = []
    for d in HF_HUB.iterdir():
        if d.is_dir() and d.name.startswith("models--"):
            model_id = cache_dir_to_model_id(d.name)
            if model_id:
                try:
                    size = sum(
                        f.stat().st_size
                        for f in d.rglob("*")
                        if f.is_file()
                    )
                except OSError:
                    size = 0
                result.append((model_id, size))
    return sorted(result, key=lambda x: x[0].lower())


def probe_port_for_model(port: int) -> str | None:
    """GET /v1/models on port. Returns model id if server running, else None."""
    url = f"http://127.0.0.1:{port}/v1/models"
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=2) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode())
            models = data.get("data", [])
            if models and isinstance(models[0], dict):
                return models[0].get("id")
    except (URLError, json.JSONDecodeError, OSError):
        pass
    return None


def probe_running_servers(extra_ports: list[int] | None = None) -> dict[str, int]:
    """Probe ports to find model_id -> port. Returns {model_id: port}."""
    ports = list(PORT_RANGE)
    if extra_ports:
        ports = list(set(ports) | set(extra_ports))
    result = {}
    for port in ports:
        model_id = probe_port_for_model(port)
        if model_id:
            result[model_id] = port
    return result


def is_port_in_use(port: int) -> bool:
    """Returns True if port has a listener."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        err = sock.connect_ex(("127.0.0.1", port))
        return err == 0
    finally:
        sock.close()


def get_pids_for_port(port: int) -> list[int]:
    """Get PIDs listening on port (macOS/Linux)."""
    try:
        out = subprocess.run(
            ["lsof", "-i", f":{port}", "-t"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return [int(p) for p in out.stdout.strip().split() if p.isdigit()]
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return []


def format_size(n: int) -> str:
    """Format bytes as human-readable."""
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


class ModelManagerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local AI Model Manager")
        self.setMinimumSize(500, 400)
        self.resize(600, 450)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top: Refresh + Port
        top = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        top.addWidget(self.refresh_btn)
        top.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(8765)
        top.addWidget(self.port_spin)
        top.addStretch()
        layout.addLayout(top)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Model", "Size", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()
        for label, slot in [("Start", self.on_start), ("Stop", self.on_stop), ("Delete", self.on_delete)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_layout.addWidget(b)
        layout.addLayout(btn_layout)

        self.refresh()

    def refresh(self):
        """Reload models and running status."""
        self.table.setRowCount(0)
        models = get_cached_models()
        try:
            port_val = self.port_spin.value()
            extra = [port_val] if 1024 <= port_val <= 65535 else []
        except ValueError:
            extra = []
        running = probe_running_servers(extra)
        if not models:
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("No models found"))
            return
        for row, (model_id, size) in enumerate(models):
            self.table.insertRow(row)
            status = f"Running on port {running[model_id]}" if model_id in running else "Stopped"
            self.table.setItem(row, 0, QTableWidgetItem(model_id))
            self.table.setItem(row, 1, QTableWidgetItem(format_size(size)))
            self.table.setItem(row, 2, QTableWidgetItem(status))

    def get_selected_model(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if not item:
            return None
        model_id = item.text()
        if model_id == "No models found":
            return None
        return model_id

    def on_start(self):
        model_id = self.get_selected_model()
        if not model_id:
            QMessageBox.warning(self, "No selection", "Select a model first.")
            return
        port = self.port_spin.value()
        if is_port_in_use(port):
            QMessageBox.critical(self, "Port in use", f"Port {port} is already in use.")
            return
        try:
            subprocess.Popen(
                [MLX_PYTHON, "-m", "mlx_lm.server", "--model", model_id, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            QMessageBox.information(
                self, "Started",
                f"Starting {model_id} on port {port}. Give it a moment to load.",
            )
            self.refresh()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_stop(self):
        model_id = self.get_selected_model()
        if not model_id:
            QMessageBox.warning(self, "No selection", "Select a model first.")
            return
        extra = [self.port_spin.value()]
        running = probe_running_servers(extra)
        port = running.get(model_id)
        if port is None:
            QMessageBox.information(self, "Not running", f"{model_id} is not running.")
            return
        pids = get_pids_for_port(port)
        if not pids:
            QMessageBox.warning(self, "Unknown PID", f"Could not find process on port {port}.")
            self.refresh()
            return
        try:
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            QMessageBox.information(self, "Stopped", f"Stopped {model_id} on port {port}.")
        except PermissionError:
            QMessageBox.critical(self, "Permission denied", "Cannot stop process (try running with sudo).")
        self.refresh()

    def on_delete(self):
        model_id = self.get_selected_model()
        if not model_id:
            QMessageBox.warning(self, "No selection", "Select a model first.")
            return
        running = probe_running_servers([self.port_spin.value()])
        if model_id in running:
            port = running[model_id]
            QMessageBox.warning(
                self, "Model in use",
                f"{model_id} is running on port {port}. Stop it first.",
            )
            return
        cache_dir = HF_HUB / model_id_to_cache_dir(model_id)
        if not cache_dir.exists():
            QMessageBox.critical(self, "Not found", f"Cache dir not found: {cache_dir}")
            return
        reply = QMessageBox.question(
            self, "Confirm delete",
            f"Delete {model_id}?\n\n{cache_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            shutil.rmtree(cache_dir)
            QMessageBox.information(self, "Deleted", f"Deleted {model_id}.")
        except OSError as e:
            QMessageBox.critical(self, "Delete failed", str(e))
        self.refresh()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ModelManagerApp()
    win.show()
    sys.exit(app.exec())
