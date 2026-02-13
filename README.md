#AeroBat工程管理
## 初始：AeroBat 场景与渲染工具箱
面向「Target–Attacker–Defender」(TAD) 三机对抗的轻量 3D 可视化环境，含脚本策略、OBJ 模型与渲染管线，便于快速演示或二次开发。
### 1. 快速上手指南
- 环境配置（Conda，推荐）  
  ```bash
  conda env create -f environment.yml
  conda activate aerobat
  ```  
  关键包：Python 3.10、gymnasium、pyglet、moderngl、PyQt6、opencv-python、imageio。

- 程序入口  
  - 默认GUI界面入口：`python main.py`（需要 PyQt6）。  
  - 渲染模式区别：`human` 会打开交互窗口（鼠标旋转/平移/缩放）；`rgb_array` 直接返回 `numpy.ndarray` 图像帧（不弹窗），用于录屏/训练采样。

- 直接在代码中创建环境  
  ```python
  from src.scenarios import make_env
  env = make_env()                # 默认使用 config/tad.yaml
  env.reset()
  for _ in range(200):
      env.step()                  # 场景脚本驱动动作
      env.render(mode="human")
  env.close()
  ```

- 镜头/交互（`rendering_3d.Viewer`）  
  - 鼠标左键拖拽：绕焦点旋转；中键拖拽：平移；滚轮：缩放。  
  - 按 `R` 复位默认视角。

- GIF 录制  
  - 在配置里设 `save_gif: true`、`gif_dir: gifs`（默认已开启）。  
  - CLI 跑完一轮后自动把帧写入 `gif_dir/render_时间戳.gif`。  
  - 离屏录制：`python main.py --cli --mode rgb_array --episode_length 200`.

### 2. 运行逻辑与核心组件
1) 配置：`config/tad.yaml` 给出场景名、数量、初始状态、策略选择、GIF 开关等。  
2) 入口装载：`src/scenarios/manager.py`  
   - `load_config(path)` → `SimpleNamespace`，自动记录配置所在目录。  
   - `make_env(config|config_path)` → 依据 `scenario_name` 动态导入场景模块（默认 `tad`），构建 `World`，再封装成 `MultiAgentEnv`。
3) 场景：`src/scenarios/tad.py`  
   - `Scenario.make_world(cfg)`：创建 `Target/Attacker/Defender` 实例、装载模型、策略选择、设置发射/拦截参数。  
   - `reset_world(world)`：按 `init_pos_offset` 添加扰动并生成初始分配。   
4) 世界演化：`TadWorld.step()`（继承自 `core_3d.World`）  
   - 为各角色调用对应策略函数 → `apply_action_force` → `integrate_state`。  
   - 同步检查击落/拦截、按需重算最优发射时刻。  
5) 环境封装：`src/core/environment_3d.py` 的 `MultiAgentEnv`  
   - `reset()` / `step()` 简化为驱动场景脚本；  
   - `render(mode="human"|"rgb_array")`：加载 `Viewer`，绑定几何并更新轨迹。  
6) 渲染栈：`src/render/rendering_3d.py` + `src/render/objloader.py`  
   - ModernGL 绘制，支持 OBJ 模型、轨迹线、辅助线；资源位于 `src/agents/3dmodels/`。  
7) UI（可选）：`main.py` 默认调用 `src/ui/app.py` 的 PyQt6 界面。  
   - 首屏 `ScenarioPage`：选择场景（当前支持 TAD）并浏览/更换 YAML 配置。  
   - 二级 `ParameterPage`（TAD 专用）：用表格+微调框编辑 Target/Attacker/Defender 的初始坐标，行数与配置数量一致，修改后会写入 `agent_params.<role>.init_pos` 再开局。  
   - 运行页 `SimulationPage`：在后台线程 (`SimulationWorker`) 执行仿真，实时渲染 `human` 窗口；如 `save_gif: true`，同时累积 `rgb_array` 帧并在结束后保存到 `gif_dir/render_*.gif`。后台线程保证 UI 不卡顿，可随时返回主菜单。  
   - 额外兼容：如果配置里没写 `save_gif` / `gif_dir` 会自动填默认值；未装 PyQt6 时打印原因并直接走 CLI。

- 运行时调用链（细化）
  - `main.py` → `parse_args` 
  - `load_config` 读取 YAML 并记录 `_cfg_dir`，便于在 `tad.py` 里用相对路径加载模型与 GIF 目录。  
  - `make_env` 根据 `scenario_name` 动态 import 同名模块，构造 `Scenario`，调用 `Scenario.make_world(cfg)` 返回 `World`，再封装成 `MultiAgentEnv`。  
  - `Scenario.make_world` 为三类角色设置尺寸/速度/加速度/颜色/策略回调，解析 `model_path`（优先相对仓库根或 `_cfg_dir`），并写入 `init_positions`。  
  - `core_3d.World` 定义 `dt=0.1`、`world_step`、实体列表等；`TadWorld.step` 先调各自策略，再 `apply_action_force` / `integrate_state`，有命中/拦截判定。  
  - `MultiAgentEnv.render` 加载 `Viewer`，第一次调用时设置相机中心；`mode='human'` 打开窗口，`mode='rgb_array'` 返回 `numpy.ndarray` 帧（列表形式，每个 viewer 一张）；`close()` 释放窗口。

### 3. TAD 场景说明
- 基础设定：1 Target + 1 Attacker + 1 Defender（目前只支持1v1v1），步长 0.1s，默认 300 步。T沿正 x 方向逃逸，A追击，D拦截A。

- `tad.yaml` 主要参数
  - 场景：`scenario_name`（模块名，默认 `tad`），`episode_length`（最大步数），`init_pos_offset`（初始位置噪声大小）。    
  - GIF：`save_gif`，`gif_dir`。  
  - `agent_params.*`（每类角色）：`size`、`max_speed`、`max_accel`、`color` (0~1)、`policy`（策略/机动选项，详细见下文）、`init_pos`（初始位置）、`model_path`（渲染用模型的路径）。

- 策略选择与自定义（见 `src/scenarios/tad.py`）
  - Target `policy`：1 直线；2 圆周（半径 1.5）；3 滚筒。  
  - Attacker `policy`：1 APNG（`N_ma` 可调）；2 微分博弈制导（GTG）。
  - Defender `policy`：PNG（`N_md` 可调）。  
  - 自定义方式：直接修改 `target_policy` / `attacker_policy` / `defender_policy` 函数逻辑，或在 `make_world` 中替换 `action_callback`。调整 YAML 里的 `agent_params.<role>.policy` 以切换脚本。
  - 自定义脚本策略（不限于现有 APNG/GTG）：  
    - 可在创建环境后动态替换回调：如 `env.world.attackers[0].action_callback = my_policy`，或在 `Scenario.make_world` 里绑定。  
    - 函数签名：Target/Attacker 用 `def my_policy(target, attacker, defender, world): ... return np.array([ax, ay, az])`；Defender 用 `def my_policy(attacker, defender): ... return np.array([ax, ay, az])`。返回值是 3 维加速度向量（单位 km/s²）。  
    - 获取状态：`agent.state.p_pos` / `agent.state.p_vel`（np.array，单位 km / km/s）；`world.world_step` 与 `world.dt`（默认 0.1s，可用 `t = world.world_step * world.dt`）；列表访问 `world.targets / world.attackers / world.defenders` 可拿到其它实体；命中阈值等在 `world.miss_tol`、`world.intercept_radius`。  
    - 约束：最好在策略内手动限幅 `np.linalg.norm(u) > agent.max_accel` 时做缩放；返回必须是 shape=(3,) 的 `numpy.ndarray` 才能被动力学积分器接受。  
    - 示例：  
      ```python
      import numpy as np
      def my_policy(target, attacker, defender, world):
          rel = target.state.p_pos - attacker.state.p_pos          # 相对位置
          vrel = target.state.p_vel - attacker.state.p_vel          # 相对速度
          u = 1.8 * rel - 0.4 * vrel                                # 简单 PD 制导
          norm = np.linalg.norm(u)
          if norm > attacker.max_accel:
              u = u / norm * attacker.max_accel
          return u
      ```

### 4. 扩展到新场景（非 TAD）
1) 新建配置：复制 `config/tad.yaml` 为 `config/<name>.yaml`，设置 `scenario_name: <name>` 及自定义参数。  
2) 编写场景：在 `src/scenarios/<name>.py` 继承 `BaseScenario`，实现 `make_world` 与至少 `reset_world`，必要时提供 `render_overlays`。若需脚本策略，定义相应 `policy` 函数。  
3) 模型：将 OBJ/纹理放入 `src/agents/3dmodels/` 或配置自定义路径。  
4) 运行：`python main.py --cli --config config/<name>.yaml`（或在代码里 `make_env(config_path=...)`）。无需额外注册，`make_env` 会按 `scenario_name` 自动导入同名模块。

###5. 类继承关系（核心）
- 场景：`BaseScenario` → `Scenario` (src/scenarios/tad.py)。  
- 世界：`World` → `TadWorld`。  
- 状态：`EntityState` → `AgentState`。  
- 实体：`Entity` → `Agent` → `{Target, Attacker, Defender}`。  
- 环境封装：`gym.Env` → `MultiAgentEnv`。  
- 渲染：`Attr` → `Transform`；`Geom` → `{Line, Point, PolyLine, FilledPolygon, FilledMesh, OBJ}`；`Viewer` 组合/持有上述 `Geom` 与 `Transform`。

##2026-2-14修改
-主要修改在src文件夹中，实现了四个页面的设计以及切换
-增添了AreoBat.bat文件，可以直接运行，但注意需要切换自己的路径和环境，也可以直接运行根目录中的main.py文件，conda环境要求和上文一致

