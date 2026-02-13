# src/ui/pages/home.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGridLayout
from PyQt6.QtCore import Qt, pyqtSignal

class HomePage(QWidget):
    scenario_selected = pyqtSignal(str) # 发送信号告诉主窗口选了哪个

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        title = QLabel("AeroBat 仿真平台")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        grid = QGridLayout()
        
        # 按钮 1: TAD 算法
        btn_tad = QPushButton("TAD 对抗场景")
        btn_tad.setFixedSize(200, 100)
        btn_tad.setStyleSheet("font-size: 16px; background-color: #0d6efd;")
        btn_tad.clicked.connect(lambda: self.scenario_selected.emit("tad"))
        grid.addWidget(btn_tad, 0, 0)
        
        # 按钮 2: 预留接口
        btn_other = QPushButton("多机协同 (开发中)")
        btn_other.setFixedSize(200, 100)
        btn_other.setEnabled(False) # 禁用
        grid.addWidget(btn_other, 0, 1)
        
        layout.addLayout(grid)
        self.setLayout(layout)