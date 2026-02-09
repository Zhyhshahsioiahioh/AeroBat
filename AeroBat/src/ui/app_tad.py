"""TAD-specific second-level menu for initial position settings."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.scenarios import load_config

# Default fallback positions match scenario defaults
ROLE_DEFAULTS = {
    "target": [1.0, 3.0, 5.0],
    "attacker": [8.0, 1.0, 3.0],
    "defender": [4.0, 1.8, 2.0],
}


def _normalize_positions(value) -> List[List[float]]:
    """Convert init_pos values to a list-of-list format."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return []
        if all(isinstance(x, (int, float)) for x in value):
            return [list(map(float, value))]
        if all(isinstance(x, (list, tuple)) for x in value):
            return [[float(v) for v in entry] for entry in value]
    return []


def _positions_for_role(cfg, role: str) -> List[List[float]]:
    """Derive editable init positions for a given role."""
    agent_params = getattr(cfg, "agent_params", {}) or {}
    init_pos = agent_params.get(role, {}).get("init_pos")
    positions = _normalize_positions(init_pos)
    if not positions:
        positions = [ROLE_DEFAULTS.get(role, [0.0, 0.0, 0.0])]

    desired_counts = {
        "target": getattr(cfg, "num_target", len(positions)),
        "attacker": getattr(cfg, "num_attacker", len(positions)),
        "defender": getattr(cfg, "num_defender", len(positions)),
    }
    desired = int(desired_counts.get(role, len(positions)))
    while len(positions) < desired:
        positions.append(positions[-1])
    if len(positions) > desired:
        positions = positions[:desired]
    return positions


class PositionGroup(QGroupBox):
    """Group box that holds editable position rows for a role."""

    def __init__(self, role: str, positions: List[List[float]], parent=None):
        super().__init__(parent)
        self.role = role
        self.setTitle(f"{role.title()} 初始位置")
        self.rows = []
        grid = QGridLayout()
        header = ["索引", "X", "Y", "Z"]
        for col, text in enumerate(header):
            label = QLabel(text)
            label.setProperty("class", "header")
            grid.addWidget(label, 0, col)

        for idx, pos in enumerate(positions):
            idx_label = QLabel(str(idx + 1))
            grid.addWidget(idx_label, idx + 1, 0)
            spin_row = []
            for axis, val in enumerate(pos[:3]):
                spin = QDoubleSpinBox()
                spin.setRange(-10000.0, 10000.0)
                spin.setDecimals(3)
                spin.setSingleStep(0.1)
                spin.setValue(float(val))
                grid.addWidget(spin, idx + 1, axis + 1)
                spin_row.append(spin)
            self.rows.append(spin_row)

        self.setLayout(grid)

    def values(self) -> List[List[float]]:
        return [[spin.value() for spin in row] for row in self.rows]


class ParameterPage(QWidget):
    """二级菜单：TAD 场景初始设置。"""

    start_sim = pyqtSignal(dict)
    back = pyqtSignal()

    def __init__(self, cfg_path: str, parent=None):
        super().__init__(parent)
        self.cfg_path = Path(cfg_path)
        self.groups: Dict[str, PositionGroup] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        title = QLabel("场景设置")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        try:
            cfg = load_config(self.cfg_path)
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "配置加载失败", str(exc))
            return

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout()

        for role in ("target", "attacker", "defender"):
            positions = _positions_for_role(cfg, role)
            group = PositionGroup(role, positions, parent=self)
            self.groups[role] = group
            inner_layout.addWidget(group)

        inner_layout.addStretch()
        inner.setLayout(inner_layout)
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        btn_row = QHBoxLayout()
        back_btn = QPushButton("返回")
        back_btn.clicked.connect(self.back.emit)
        start_btn = QPushButton("开始仿真")
        start_btn.clicked.connect(self._emit_start)
        btn_row.addWidget(back_btn)
        btn_row.addWidget(start_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def _emit_start(self):
        positions = {role: group.values() for role, group in self.groups.items()}
        self.start_sim.emit(positions)


__all__ = ["ParameterPage"]
