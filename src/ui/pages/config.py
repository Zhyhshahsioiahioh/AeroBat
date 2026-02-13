# src/ui/config.py
import yaml
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QGroupBox, QPushButton, QScrollArea, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class PositionGroup(QGroupBox):
    """管理单个角色的 (X, Y, Z) 输入框"""

    def __init__(self, role_name, init_pos, parent=None):
        super().__init__(role_name.capitalize(), parent)
        self.inputs = []
        layout = QHBoxLayout()

        # 样式美化
        self.setStyleSheet("""
            QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 10px; font-weight: bold; color: #ddd; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
        """)

        coords = ["X", "Y", "Z"]
        # 确保 init_pos 是列表
        if not isinstance(init_pos, list) or len(init_pos) < 3:
            init_pos = [0.0, 0.0, 0.0]

        for i, axis in enumerate(coords):
            layout.addWidget(QLabel(f"{axis}:"))
            spin = QDoubleSpinBox()
            spin.setRange(-1000.0, 1000.0)  # 范围给大点
            spin.setSingleStep(0.5)
            spin.setDecimals(2)
            spin.setValue(float(init_pos[i]))

            spin.setStyleSheet("background-color: #444; color: white; border: 1px solid #666; padding: 3px;")
            self.inputs.append(spin)
            layout.addWidget(spin)

        self.setLayout(layout)

    def get_values(self):
        return [spin.value() for spin in self.inputs]


class ParameterPage(QWidget):
    # 信号：传出临时配置文件的【绝对路径】(str)
    start_sim = pyqtSignal(str)
    back_home = pyqtSignal()

    def __init__(self, template_path: str, parent=None):
        super().__init__(parent)
        self.template_path = Path(template_path)
        self.groups = {}
        self.raw_yaml_data = {}  # 存储读取到的原始 YAML 数据（字典）

        # 核心：为了防止崩坏，所有初始化逻辑包在 try-except 里
        try:
            self._load_template()
            self._build_ui()
        except Exception as e:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel(f"初始化配置页失败: {e}"))
            import traceback
            traceback.print_exc()

    def _load_template(self):
        """读取 init_tad.yaml"""
        if not self.template_path.exists():
            raise FileNotFoundError(f"找不到模板文件: {self.template_path}")

        with open(self.template_path, 'r', encoding='utf-8') as f:
            self.raw_yaml_data = yaml.safe_load(f)

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. 标题
        title = QLabel(f"场景参数配置 (基于模板)")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px; color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # 2. 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)

        # 3. 动态生成控件 (基于 raw_yaml_data 字典)
        # 注意：这里我们只关心 agent_params 下的 target, attacker, defender
        agent_params = self.raw_yaml_data.get('agent_params', {})
        target_roles = ['target', 'attacker', 'defender']

        for role in target_roles:
            # 从字典中安全获取
            role_data = agent_params.get(role, {})
            # 如果 YAML 里写的是 init_pos: [0,0,0]，读出来就是 list
            init_pos = role_data.get('init_pos', [0.0, 0.0, 0.0])

            group = PositionGroup(role, init_pos)
            self.groups[role] = group
            content_layout.addWidget(group)

        content_layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        # 4. 底部按钮
        btn_layout = QHBoxLayout()
        back_btn = QPushButton("返回首页")
        back_btn.clicked.connect(self.back_home.emit)

        start_btn = QPushButton("保存并开始")
        start_btn.setStyleSheet("background-color: #198754; color: white; font-weight: bold; padding: 10px;")
        start_btn.clicked.connect(self._on_save_and_start)

        btn_layout.addWidget(back_btn)
        btn_layout.addWidget(start_btn)
        main_layout.addLayout(btn_layout)

    def _on_save_and_start(self):
        try:
            # 1. 更新内存中的数据字典
            agent_params = self.raw_yaml_data.get('agent_params', {})
            for role, group in self.groups.items():
                if role in agent_params:
                    # 更新 init_pos
                    agent_params[role]['init_pos'] = group.get_values()

            # 2. 生成临时文件路径 (temp_tad.yaml)
            temp_path = self.template_path.parent / f"temp_{self.template_path.name}"

            # 3. 写入文件
            with open(temp_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.raw_yaml_data, f, default_flow_style=False)

            # 4. 发送临时文件的路径给主窗口
            self.start_sim.emit(str(temp_path))

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法写入临时配置文件:\n{e}")