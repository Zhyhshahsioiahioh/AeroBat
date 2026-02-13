# src/ui/main_window.py
import traceback
import shutil
from pathlib import Path
from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QMessageBox, QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt

# 导入页面组件
from src.ui.pages.home import HomePage
from src.ui.pages.config import ParameterPage
from src.ui.pages.result import ResultPage
from src.ui.worker import SimulationWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AeroBat 仿真平台")
        self.resize(1000, 700)

        # 样式加载
        style_path = Path(__file__).parent / "styles.qss"
        if style_path.exists():
            try:
                with open(style_path, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
            except Exception:
                pass

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.init_pages()

    def init_pages(self):
        # 1. 首页
        self.home_page = HomePage()
        self.home_page.scenario_selected.connect(self.go_to_config)
        self.stack.addWidget(self.home_page)

        # 2. 仿真加载页
        self.loading_page = QWidget()
        l_layout = QVBoxLayout()
        l_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label = QLabel("正在初始化仿真...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(400)
        l_layout.addWidget(self.progress_label)
        l_layout.addWidget(self.progress_bar)
        self.loading_page.setLayout(l_layout)
        self.stack.addWidget(self.loading_page)

        # 3. 结果页
        self.result_page = ResultPage()
        self.result_page.go_home.connect(self.go_home)
        self.stack.addWidget(self.result_page)

        self.worker = None

    def go_to_config(self, scenario_name):
        try:
            # 获取项目根目录
            project_root = Path(__file__).resolve().parents[2]
            config_dir = project_root / "config"

            # --- 关键修改：确保 init_tad.yaml 存在 ---
            if scenario_name == "tad":
                init_cfg_path = config_dir / "init_tad.yaml"
                default_cfg_path = config_dir / "tad.yaml"

                # 如果模板不存在，自动从 tad.yaml 复制一个
                if not init_cfg_path.exists():
                    if default_cfg_path.exists():
                        shutil.copy(default_cfg_path, init_cfg_path)
                    else:
                        raise FileNotFoundError(f"找不到配置文件，请确保 {default_cfg_path} 存在")

                # 实例化配置页，传入模板路径
                self.config_page = ParameterPage(str(init_cfg_path))
                self.config_page.start_sim.connect(self.start_simulation)
                self.config_page.back_home.connect(self.go_home)

                self.stack.addWidget(self.config_page)
                self.stack.setCurrentWidget(self.config_page)

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"无法进入配置页面：\n{str(e)}")

    def start_simulation(self, temp_cfg_path):
        """temp_cfg_path 是 ParameterPage 生成的临时文件路径"""
        self.stack.setCurrentWidget(self.loading_page)
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在执行仿真逻辑...")

        try:
            self.worker = SimulationWorker(temp_cfg_path)
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self.show_results)
            self.worker.failed.connect(self.show_error)
            self.worker.start()
        except Exception as e:
            self.show_error(str(e))

    def show_results(self, logs):
        self.result_page.update_data(logs)
        self.stack.setCurrentWidget(self.result_page)

    def show_error(self, err_msg):
        QMessageBox.critical(self, "仿真出错", f"错误详情：\n{err_msg}")
        self.go_home()

    def go_home(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        self.stack.setCurrentWidget(self.home_page)