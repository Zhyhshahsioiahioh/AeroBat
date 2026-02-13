# src/ui/pages/result.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class ResultPage(QWidget):
    go_home = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # 标题栏
        self.status_label = QLabel("仿真结果")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.layout.addWidget(self.status_label)
        
        # 图表区域
        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor='#2b2b2b')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout.addWidget(self.canvas)
        
        # 底部按钮
        btn = QPushButton("返回首页")
        btn.clicked.connect(self.go_home.emit)
        self.layout.addWidget(btn)

    def update_data(self, logs):
        """主窗口调用此函数传入数据"""
        self.figure.clear()
        
        # 设置结果标题颜色
        status_text = "拦截成功" if logs['success'] else "拦截失败"
        color = "#00ff00" if logs['success'] else "#ff4444"
        self.status_label.setText(f"仿真结束 - {status_text}")
        self.status_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {color};")

        # 绘图
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#3d3d3d') # 图表背景色
        
        # 绘制曲线
        ax.plot(logs['time'], logs['dist_DA'], label='Missile-Target Dist', color='#0d6efd', linewidth=2)
        ax.plot(logs['time'], logs['dist_AT'], label='Attacker-Target Dist', color='#ffc107', linestyle='--')
        
        # 美化坐标轴
        ax.set_xlabel("Time (s)", color='white')
        ax.set_ylabel("Distance (km)", color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.legend(facecolor='#3d3d3d', labelcolor='white')
        
        self.canvas.draw()