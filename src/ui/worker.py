# src/ui/worker.py
import traceback
import time
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

# 引入项目核心
from src.scenarios import make_env, load_config


class SimulationWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, cfg_path: str):
        super().__init__()
        self.cfg_path = cfg_path
        self._is_running = True

    def run(self):
        env = None
        try:
            # 1. 直接加载传进来的临时配置文件 (temp_tad.yaml)
            # 因为文件已经在 Config 页被修改并保存了，这里直接读就是最新的参数
            cfg = load_config(self.cfg_path)

            # 2. 创建环境
            env = make_env(config=cfg)
            env.reset()

            # 3. 初始化数据记录
            logs = {"time": [], "dist_AT": [], "dist_DA": [], "success": False, "gif_path": ""}
            gif_frames = []

            # 读取配置参数
            save_gif = getattr(cfg, 'save_gif', True)
            max_steps = int(getattr(cfg, 'episode_length', 300))

            # 4. 循环仿真
            for step in range(max_steps):
                if not self._is_running: break

                env.step()

                # 数据采集
                logs["time"].append(step * env.world.dt)
                try:
                    agents = env.world.agents
                    t_pos, a_pos, d_pos = agents[0].state.p_pos, agents[1].state.p_pos, agents[2].state.p_pos
                    logs["dist_AT"].append(float(np.linalg.norm(a_pos - t_pos)))
                    logs["dist_DA"].append(float(np.linalg.norm(d_pos - a_pos)))
                except:
                    pass

                # 画面渲染
                if save_gif:
                    frame = env.render(mode="rgb_array")[0]
                    gif_frames.append(frame)

                self.progress.emit(int((step / max_steps) * 100))

            env.close()

            # 5. 结果判定与保存
            if logs["dist_DA"]:
                intercept_r = getattr(env.world, 'intercept_radius', 0.5)
                logs["success"] = min(logs["dist_DA"]) < intercept_r

            if save_gif and gif_frames:
                gif_dir = Path(getattr(cfg, 'gif_dir', 'gifs'))
                if not gif_dir.is_absolute():
                    gif_dir = Path(self.cfg_path).parent.parent / gif_dir  # 尝试相对于项目根目录

                gif_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                gif_path = gif_dir / f"result_{timestamp}.gif"

                imageio.mimsave(str(gif_path), gif_frames, duration=0.05)
                logs["gif_path"] = str(gif_path)

            self.finished.emit(logs)

        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))
        finally:
            if env:
                try:
                    env.close()
                except:
                    pass