"""PyQt6 application shell for AeroBat."""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict

import imageio.v2 as imageio

# Ensure Qt DLLs are discoverable even when PyQt6 lives in user-site
try:  # pragma: no cover
    import PyQt6

    qt_bin_dir = Path(PyQt6.__file__).parent / "Qt6" / "bin"
    if qt_bin_dir.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(qt_bin_dir))
except Exception:
    pass

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from src.scenarios import make_env, load_config
from src.ui.app_tad import ParameterPage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG_PATH = PROJECT_ROOT / "config" / "tad.yaml"


class ScenarioPage(QWidget):
    """一级菜单：选择场景与配置文件。"""

    proceed = pyqtSignal(str, str)

    def __init__(self, default_cfg: Path, parent=None):
        super().__init__(parent)
        self.default_cfg = default_cfg
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        title = QLabel("AeroBat 场景选择")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.scenario_combo = QComboBox()
        self.scenario_combo.addItem("TAD", userData="tad")
        scenario_row = QHBoxLayout()
        scenario_row.addWidget(QLabel("场景："))
        scenario_row.addWidget(self.scenario_combo)
        layout.addLayout(scenario_row)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("配置文件："))
        self.cfg_edit = QLineEdit(str(self.default_cfg))
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse)
        cfg_row.addWidget(self.cfg_edit, stretch=1)
        cfg_row.addWidget(browse_btn)
        layout.addLayout(cfg_row)

        button_row = QHBoxLayout()
        next_btn = QPushButton("下一步")
        next_btn.clicked.connect(self._on_next)
        quit_btn = QPushButton("退出")
        quit_btn.clicked.connect(QApplication.instance().quit)
        button_row.addWidget(next_btn)
        button_row.addWidget(quit_btn)
        layout.addLayout(button_row)

        layout.addStretch()
        self.setLayout(layout)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择配置文件",
            str(self.default_cfg.parent),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self.cfg_edit.setText(path)

    def _on_next(self):
        cfg_path = Path(self.cfg_edit.text()).expanduser()
        if not cfg_path.exists():
            QMessageBox.warning(self, "文件不存在", f"未找到配置文件：{cfg_path}")
            return
        scenario_key = self.scenario_combo.currentData()
        self.proceed.emit(scenario_key, str(cfg_path))


class SimulationPage(QWidget):
    """仿真过程中展示状态，并允许完成后返回一级菜单。"""

    back = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.label = QLabel("正在启动仿真...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 15px;")
        layout.addWidget(self.label)

        self.back_btn = QPushButton("返回主菜单")
        self.back_btn.setEnabled(False)
        self.back_btn.clicked.connect(self.back.emit)
        layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        self.setLayout(layout)

    def set_status(self, text: str, finished: bool = False):
        self.label.setText(text)
        self.back_btn.setEnabled(finished)


class SimulationWorker(QThread):
    """后台线程运行仿真，避免阻塞 UI。"""

    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        cfg_path: str,
        positions: Dict[str, list],
        sleep_time: float = 0.05,
        frames_override=None,
    ):
        super().__init__()
        self.cfg_path = Path(cfg_path)
        self.positions = positions
        self.sleep_time = sleep_time
        self.frames_override = frames_override

    def run(self):
        try:
            cfg = load_config(self.cfg_path)
            if not hasattr(cfg, "save_gif"):
                cfg.save_gif = False
            if not hasattr(cfg, "gif_dir"):
                cfg.gif_dir = "gifs"
            frames_count = (
                self.frames_override if self.frames_override is not None else getattr(cfg, "episode_length", 300)
            )

            agent_params = getattr(cfg, "agent_params", {}) or {}
            for role, pos_list in self.positions.items():
                if role not in agent_params:
                    agent_params[role] = {}
                cleaned = [[float(v) for v in entry[:3]] for entry in pos_list]
                agent_params[role]["init_pos"] = cleaned[0] if len(cleaned) == 1 else cleaned
            cfg.agent_params = agent_params

            env = make_env(config=cfg)
            env.reset()

            gif_frames = []
            gif_dir = Path(cfg.gif_dir)
            if not gif_dir.is_absolute():
                gif_dir = PROJECT_ROOT / gif_dir
            gif_path = None

            try:
                for _ in range(int(frames_count)):
                    env.step()
                    if cfg.save_gif:
                        frame = env.render(mode="rgb_array")[0]
                        gif_frames.append(frame)
                    env.render(mode="human")
                    time.sleep(self.sleep_time)
            finally:
                if cfg.save_gif and gif_frames:
                    gif_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    gif_path = gif_dir / f"render_{timestamp}.gif"
                    imageio.mimsave(gif_path, gif_frames, duration=self.sleep_time)
                env.close()

            self.finished.emit(str(gif_path) if gif_path else "")
        except Exception as exc:  # pragma: no cover - UI surface
            traceback.print_exc()
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, default_cfg: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AeroBat 控制台")
        self.resize(640, 520)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.scenario_page = ScenarioPage(default_cfg)
        self.scenario_page.proceed.connect(self._on_scenario_chosen)
        self.stack.addWidget(self.scenario_page)

        self.param_page = None
        self.sim_page = SimulationPage()
        self.sim_page.back.connect(self._back_to_main)
        self.stack.addWidget(self.sim_page)

        self.worker: SimulationWorker | None = None

    def _on_scenario_chosen(self, scenario_key: str, cfg_path: str):
        if scenario_key != "tad":
            QMessageBox.warning(self, "暂不支持", f"场景 {scenario_key} 暂未实现。")
            return

        if self.param_page is not None:
            self.stack.removeWidget(self.param_page)
            self.param_page.deleteLater()

        self.param_page = ParameterPage(cfg_path)
        self.param_page.start_sim.connect(lambda positions: self._start_simulation(cfg_path, positions))
        self.param_page.back.connect(self._back_to_main)
        self.stack.insertWidget(1, self.param_page)
        self.stack.setCurrentWidget(self.param_page)

    def _start_simulation(self, cfg_path: str, positions: Dict[str, list]):
        if self.worker is not None:
            self.worker.quit()
            self.worker.wait()

        self.sim_page.set_status("正在仿真，请稍候...")
        self.stack.setCurrentWidget(self.sim_page)
        self.worker = SimulationWorker(cfg_path, positions)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, gif_path: str):
        msg = "仿真结束。"
        if gif_path:
            msg += f"\nGIF 已保存：{gif_path}"
        self.sim_page.set_status(msg, finished=True)
        QMessageBox.information(self, "完成", msg)
        self._back_to_main()

    def _on_failed(self, err: str):
        QMessageBox.critical(self, "仿真失败", err)
        self._back_to_main()

    def _back_to_main(self):
        if self.worker is not None:
            self.worker.quit()
            self.worker.wait()
            self.worker = None
        self.stack.setCurrentWidget(self.scenario_page)


def launch_app(default_config: str | Path = DEFAULT_CFG_PATH):
    """Entry point to start the PyQt6 UI."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(Path(default_config))
    window.show()
    app.exec()
