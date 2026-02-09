import copy
import math
from pathlib import Path

import numpy as np

from ..core.core_3d import World, Agent
from ..core import global_var as glv
from .base import BaseScenario
from .util_3d import *

'''
距离单位: km
时间单位: s
'''
Mach = 0.34  # km/s
G = 9.8e-3  # km/s^2

global a1, b1, c1, d1, N_md, N_ma
a1 = 0
b1 = 1e10
c1 = 0.5
d1 = 0.5
N_md = 2
N_ma = 5


class Scenario(BaseScenario):
    def __init__(self) -> None:
        super().__init__()
        self.cp = 0.4
        self.assign_list = []  # 为attackers初始化分配target, 长度为num_A, 值为target的id
        self.init_assign = True  # 仅在第一次分配时使用
        self.miss_tol = 0.1
        self.intercept_radius = 0.1
        self.launch_time_default = 0.0
        self.launch_time_mode = "immediate"
        self.use_optimal_launch = False
        self.launch_scan_u_max = 3.0
        self.launch_scan_N = 10
        self.launch_scan_horizon = 10.0
        self.launch_rescan_steps = None
        self.last_launch_plan = None

    # 设置agent,landmark的数量，运动属性。
    def make_world(self, args):
        self.num_target = args.num_target
        self.num_attacker = args.num_attacker
        self.num_defender = args.num_defender

        base_dir = Path(getattr(args, "_cfg_dir", Path(__file__).resolve().parents[2]))
        # If config lives under ./config, use repo root as base for relative assets
        if base_dir.name == "config":
            base_dir = base_dir.parent
        agent_params = getattr(args, "agent_params", {})

        def param(role: str, key: str, default):
            return agent_params.get(role, {}).get(key, default)

        def resolve_model(role: str):
            rel = param(role, "model_path", None)
            # backward compatibility: fall back to legacy model_paths if present
            if rel is None:
                legacy = getattr(args, "model_paths", {})
                rel = legacy.get(role)
            if not rel:
                return None
            p = Path(rel)
            if p.is_absolute() and p.exists():
                return p

            # Try relative to base_dir first (typically repo root)
            candidate = base_dir / rel
            if candidate.exists():
                return candidate

            # Fall back to path relative to config directory if provided
            cfg_dir = Path(getattr(args, "_cfg_dir", base_dir))
            candidate = cfg_dir / rel
            return candidate if candidate.exists() else None

        world = TadWorld()
        world.collaborative = True
        world.world_length = args.episode_length
        world.init_pos_offset = args.init_pos_offset
        world.attacker_policy = param("attacker", "policy", getattr(args, "attacker_policy", 0))
        # launch-time control params
        self.miss_tol = getattr(args, "miss_tol", self.miss_tol)
        self.intercept_radius = getattr(args, "intercept_radius", self.intercept_radius)
        self.launch_time_mode = getattr(args, "launch_time_mode", self.launch_time_mode)
        self.launch_time_default = getattr(args, "launch_time", self.launch_time_default)
        self.launch_scan_N = getattr(args, "launch_scan_N", self.launch_scan_N)
        self.launch_scan_horizon = getattr(args, "launch_scan_horizon", self.launch_scan_horizon)
        self.launch_scan_u_max = getattr(args, "launch_scan_u_max", self.launch_scan_u_max)
        self.launch_rescan_steps = getattr(args, "launch_rescan_steps", self.launch_rescan_steps)

        world.scenario = self
        world.intercept_radius = self.intercept_radius
        world.miss_tol = self.miss_tol

        # set any world properties first
        world.targets = [Target() for _ in range(self.num_target)]
        world.attackers = [Attacker() for _ in range(self.num_attacker)]
        world.defenders = [Defender() for _ in range(self.num_defender)]
        world.agents = world.targets + world.attackers + world.defenders

        for i, target in enumerate(world.targets):
            target.id = i
            target.size = param("target", "size", 0.08)
            target.color = np.array(param("target", "color", [0.45, 0.95, 0.45]))
            target.max_speed = param("target", "max_speed", 0.15)
            target.max_accel = param("target", "max_accel", 0.06)
            target.action_callback = target_policy
            target.select_policy = param("target", "policy", getattr(args, "target_policy", 0))
            target.model_path = resolve_model("target")

        for i, attacker in enumerate(world.attackers):
            attacker.id = i
            attacker.size = param("attacker", "size", 0.04)
            attacker.color = np.array(param("attacker", "color", [0.95, 0.45, 0.45]))
            attacker.max_speed = param("attacker", "max_speed", 0.4)
            attacker.max_accel = param("attacker", "max_accel", 0.2)
            attacker.model_path = resolve_model("attacker")
            attacker.select_policy = param("attacker", "policy", getattr(args, "attacker_policy", 0))
            if attacker.select_policy != 0:
                attacker.action_callback = attacker_policy

        for i, defender in enumerate(world.defenders):
            defender.id = i
            defender.size = param("defender", "size", 0.04)
            defender.color = np.array(param("defender", "color", [0.45, 0.45, 0.95]))
            defender.max_speed = param("defender", "max_speed", 0.4)
            defender.max_accel = param("defender", "max_accel", 0.2)
            defender.model_path = resolve_model("defender")
            defender.action_callback = defender_policy
            defender.is_launched = True  # defenders spawn in-world, not launched from targets
            defender.launcher = None

        legacy_init = getattr(args, "init_positions", {})
        def _init_for(role, key, default):
            p = param(role, "init_pos", None)
            if p is not None:
                return [p] if np.array(p).ndim == 1 else p
            if isinstance(legacy_init, dict) and key in legacy_init:
                return legacy_init[key]
            return [default]

        # build unified init positions from agent params (with legacy fallback)
        self.init_positions = {
            "targets": _init_for("target", "targets", [1.0, 3.0, 5.0]),
            "attackers": _init_for("attacker", "attackers", [8.0, 1.0, 3.0]),
            "defenders": _init_for("defender", "defenders", [4.0, 1.8, 2.0]),
        }
        self.base_dir = base_dir

        self.reset_world(world)
        return world

    def reset_world(self, world):
        init_pos_offset = world.init_pos_offset
        normal_std = init_pos_offset / 4
        self.init_assign = True
        self.assign_list = rand_assign_targets(self.num_target, self.num_attacker)
        if hasattr(world, "sim_metrics"):
            world.sim_metrics.update({
                "J": 0.0,
                "peak_acc": 0.0,
                "success": False,
                "miss_distance": None,
                "min_dm": float("inf"),
            })
        # print('init assign_list is:', self.assign_list)

        # properties and initial states for agents
        def pick_positions(key, count, base, mask):
            cfg = np.array(self.init_positions.get(key, []), dtype=float) if isinstance(self.init_positions, dict) else None
            if cfg is not None and len(cfg) >= count:
                return cfg[:count]
            arr = np.tile(np.array(base, dtype=float), (count, 1))
            arr += np.random.randn(*arr.shape) * normal_std * np.array(mask)
            return arr

        init_pos_target = pick_positions("targets", self.num_target, [1.0, 3.0, 5.0], [0, 1, 0])
        
        for i, target in enumerate(world.targets):
            #print("world.targets",world.targets)
            target.done = False
            target.state.p_pos = init_pos_target[i]
            target.state.p_vel = np.array([target.max_speed, 0, 0.0])
            if target.select_policy == 1:
                # target.state.p_vel = self.randomize_init_vel(np.array([target.max_speed, 0.0, 0.0]), 15)
                target.state.p_vel = np.array([target.max_speed * np.cos(np.deg2rad(20)), 0.0, target.max_speed * np.sin(np.deg2rad(20))])
            elif target.select_policy == 3:
                target.state.p_vel = np.array([target.max_speed*np.sqrt(3)/2, -target.max_speed*0.5, 0.0])
            else:
                target.state.p_vel = np.array([target.max_speed, 0.0, 0.0])
            target.state.V = np.linalg.norm(target.state.p_vel)
            target.state.phi = 0.
            target.attacker = [j for j in range(self.num_attacker) if
                               self.assign_list[j] == i]  # the id of attacker in world.attackers
            target.defender = None
            self.attackers = []
            self.defenders = []
            self.cost = []

        init_pos_attacker = pick_positions("attackers", self.num_attacker, [8.0, 1.0, 3.0], [0, 1, 1])

        for i, attacker in enumerate(world.attackers):
            attacker.done = False
            attacker.state.p_pos = init_pos_attacker[i]
            attacker.state.p_vel = np.array([-attacker.max_speed, 0.0, 0.0])
            attacker.state.V = np.linalg.norm(attacker.state.p_vel)
            attacker.state.phi = np.pi
            attacker.true_target = self.assign_list[i]
            attacker.fake_target = self.assign_list[i]
            attacker.last_belief = attacker.fake_target
            attacker.flag_kill = False
            attacker.flag_dead = False
            attacker.is_locked = False
            attacker.defenders = []

        init_pos_defender = pick_positions("defenders", self.num_defender, [4.0, 1.8, 2.0], [0, 1, 1])
        
        for i, defender in enumerate(world.defenders):
            defender.done = False
            defender.state.p_pos = init_pos_defender[i]
            defender.state.p_vel = np.array([defender.max_speed, 0.0, 0.0])
            defender.state.V = np.linalg.norm(defender.state.p_vel)
            defender.state.phi = float(np.arctan2(defender.state.p_vel[1], defender.state.p_vel[0])) if defender.state.V > 0 else 0.0
            defender.attacker = None
            defender.target = None
            defender.is_launched = True
            defender.launcher = None

        self.prev_tar_dist = np.linalg.norm(target.state.p_pos - attacker.state.p_pos)
        self.update_belief(world)
        self._apply_launch_mode(world, initial=True)

    def _apply_launch_mode(self, world, initial=False):
        mode = (self.launch_time_mode or "immediate").lower()
        self.launch_time_mode = mode
        chosen_launch = float(self.launch_time_default)
        if mode == "immediate":
            chosen_launch = 0.0
        elif mode == "fixed":
            chosen_launch = float(self.launch_time_default)
        elif mode == "optimal" and self.use_optimal_launch:
            plan = self._search_optimal_launch(world)
            if plan is not None:
                chosen_launch = float(plan["launch_time"])
                self.last_launch_plan = plan
        for target in world.targets:
            target.launch_time = chosen_launch
        for defender in world.defenders:
            defender.is_launched = True  # defenders are always active
        if hasattr(world, "sim_metrics"):
            world.sim_metrics.update({
                "launch_mode": mode,
                "launch_time": chosen_launch,
            })
            if self.last_launch_plan is not None:
                world.sim_metrics["predicted_J"] = self.last_launch_plan.get("cost")
                world.sim_metrics["predicted_miss"] = self.last_launch_plan.get("miss_distance")

    def maybe_update_launch_time(self, world):
        if self.launch_time_mode != "optimal" or not self.use_optimal_launch:
            return
        if self.launch_rescan_steps in (None, 0):
            if world.world_step > 0:
                return
        else:
            if world.world_step % self.launch_rescan_steps != 0:
                return
        plan = self._search_optimal_launch(world)
        if plan is None:
            return
        self.last_launch_plan = plan
        for target in world.targets:
            target.launch_time = float(plan["launch_time"])
        if hasattr(world, "sim_metrics"):
            world.sim_metrics.update({
                "launch_mode": "optimal",
                "launch_time": float(plan["launch_time"]),
                "predicted_J": plan.get("cost"),
                "predicted_miss": plan.get("miss_distance"),
            })

    def _search_optimal_launch(self, world):
        candidates = np.linspace(0.0, self.launch_scan_u_max, int(self.launch_scan_N))
        best_plan = None
        for u in candidates:
            rollout_res = self._rollout_launch_time(world, u)
            if rollout_res["success"] and rollout_res["miss_distance"] <= self.miss_tol:
                if best_plan is None or rollout_res["cost"] < best_plan["cost"]:
                    best_plan = {"launch_time": float(u), **rollout_res}
        if best_plan is None:
            fallback_time = 0.0 if self.launch_time_mode == "optimal" else self.launch_time_default
            fallback_res = self._rollout_launch_time(world, fallback_time)
            best_plan = {"launch_time": float(fallback_time), **fallback_res, "fallback": True}
        return best_plan

    def _rollout_launch_time(self, world, launch_time):
        sim_world = copy.deepcopy(world)
        sim_world.world_step = world.world_step
        for target in sim_world.targets:
            target.launch_time = float(launch_time)
        for defender in sim_world.defenders:
            defender.launcher = None
            defender.is_launched = True
            defender.done = False
            if defender.attacker is None and len(sim_world.attackers) > 0:
                defender.attacker = 0
        for attacker in sim_world.attackers:
            attacker.done = False
            attacker.flag_dead = False
            attacker.flag_kill = False
        max_steps = max(1, int(self.launch_scan_horizon / sim_world.dt))
        max_steps = min(max_steps, max(1, world.world_length - world.world_step))
        cost = 0.0
        peak_acc = 0.0
        min_dm = float("inf")
        success = False
        for _ in range(max_steps):
            sim_world.step()
            self._update_termination_flags(sim_world)
            for defender in sim_world.defenders:
                if defender.attacker is None or defender.attacker >= len(sim_world.attackers):
                    continue
                dm = np.linalg.norm(defender.state.p_pos - sim_world.attackers[defender.attacker].state.p_pos)
                min_dm = min(min_dm, dm)
                if defender.is_launched and not defender.done:
                    acc_norm = float(np.linalg.norm(defender.action.u))
                    peak_acc = max(peak_acc, acc_norm)
                    cost += 0.5 * acc_norm * sim_world.dt
            if any(att.flag_dead for att in sim_world.attackers):
                success = True
                break
            if any(t.done for t in sim_world.targets):
                break
        if not np.isfinite(min_dm):
            min_dm = float(np.linalg.norm(sim_world.defenders[0].state.p_pos - sim_world.attackers[0].state.p_pos))
        miss_ok = min_dm <= self.miss_tol
        if miss_ok and not any(t.done for t in sim_world.targets):
            success = True
        return {
            "success": success,
            "miss_distance": float(min_dm),
            "cost": float(cost),
            "peak_acc": float(peak_acc),
        }

    def update_events(self, world, record_metrics=False):
        self._update_termination_flags(world)
        if record_metrics and hasattr(world, "sim_metrics"):
            if len(world.defenders) > 0 and len(world.attackers) > 0:
                attacker_idx = world.defenders[0].attacker if world.defenders[0].attacker is not None else 0
                attacker_idx = min(attacker_idx, len(world.attackers) - 1)
                dm = np.linalg.norm(world.defenders[0].state.p_pos - world.attackers[attacker_idx].state.p_pos)
                world.sim_metrics["min_dm"] = min(world.sim_metrics.get("min_dm", float("inf")), dm)
                world.sim_metrics["miss_distance"] = world.sim_metrics["min_dm"]
            world.sim_metrics["success"] = any(att.flag_dead for att in world.attackers)
            if world.sim_metrics.get("miss_distance") is not None:
                if world.sim_metrics["miss_distance"] <= getattr(world, "miss_tol", self.miss_tol):
                    if any(not t.done for t in world.targets):
                        world.sim_metrics["success"] = True

    def render_overlays(self, world, viewers):
        """Draw scenario-specific helper lines on top of the generic render."""
        for viewer in viewers:
            for attacker in getattr(world, "attackers", []):
                if getattr(attacker, "done", False):
                    continue
                tgt_idx = min(getattr(attacker, "fake_target", 0), len(world.targets) - 1)
                tgt = world.targets[tgt_idx]
                line = viewer.draw_line(attacker.state.p_pos, tgt.state.p_pos)
                line.set_color(*(attacker.color if attacker.color is not None else (1, 0, 0)), alpha=0.5)

            for defender in getattr(world, "defenders", []):
                if getattr(defender, "done", False) or defender.attacker is None:
                    continue
                atk_idx = min(defender.attacker, len(world.attackers) - 1)
                att = world.attackers[atk_idx]
                line = viewer.draw_line(defender.state.p_pos, att.state.p_pos)
                line.set_color(*(defender.color if defender.color is not None else (0, 1, 0)), alpha=0.5)

    # ------------------------------------------------------------------ #
    # Minimal stubs for compatibility (RL logic removed)                 #
    # ------------------------------------------------------------------ #
    def reward(self, agent, world):
        return 0.0

    def observation(self, agent, world):
        return np.zeros(0)

    def info(self, agent, world):
        return {}

    def done(self, agent, world):
        return False

    def _update_termination_flags(self, world):
        intercept_radius = getattr(world, "intercept_radius", self.intercept_radius)
        for attacker in world.attackers:
            if attacker.done:
                continue
            target = world.targets[attacker.true_target]
            dist_at = np.linalg.norm(attacker.state.p_pos - target.state.p_pos)
            if dist_at <= intercept_radius:
                attacker.done = True
                attacker.flag_kill = True
                target.done = True
        for defender in world.defenders:
            if defender.attacker is None:
                continue
            attacker = world.attackers[defender.attacker]
            if defender.done or attacker.done:
                continue
            dist_da = np.linalg.norm(defender.state.p_pos - attacker.state.p_pos)
            if dist_da <= intercept_radius:
                attacker.done = True
                attacker.flag_dead = True
                defender.done = True

    def randomize_init_vel(self, base_vel, max_angle_deg):
        max_angle_rad = max_angle_deg * np.pi / 180
        theta = np.random.rand() * max_angle_rad 
        vel = rotate_vector(v=base_vel, angle=theta)
        phi = np.random.rand() * 2 * math.pi
        vel = rotate_vector(v=vel, angle=phi, p=base_vel)
        return vel

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        dist = np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos)
        dist_min = agent1.size + agent2.size
        if dist<dist_min:
            print("AAAAAAAAA", agent1, agent2)
        return True if dist < dist_min else False

    def target_defender(self, world):
        return [agent for agent in world.agents if not agent.name == 'attacker']

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def done(self, agent, world):  #
        self.print_reward_proportions()
        # for attackers only
        if self.is_collision(agent, world.targets[agent.true_target]) and not agent.done:
            agent.done = True
            agent.flag_kill = True
            world.targets[agent.true_target].done = True
            #return True

        for i, defender_id in enumerate(agent.defenders):
            if self.is_collision(agent, world.defenders[defender_id]) and not agent.done:
                agent.done = True
                agent.flag_dead = True
                world.defenders[defender_id].done = True
                return True

        # 要加下面这句话，否则done一次后前面两个if不会进入，立刻又变成False了。
        if agent.done:
            return True
        if world.world_step >= world.world_length:
            return True
        return False

    def update_belief(self, world):
        target_list = []
        defender_list = []
        attacker_list = []
        target_id_list = []
        defender_id_list = []
        attacker_id_list = []
        for target in world.targets:
            if not target.done:
                target_list.append(target)
                target_id_list.append(target.id)
                target.defenders = []
                target.attackers = []  # 重新分配ADs
                target.cost = []
        for defender in world.defenders:
            if not defender.done:
                defender_list.append(defender)
                defender_id_list.append(defender.id)
        for attacker in world.attackers:
            if not attacker.done:
                attacker.defenders = []
                attacker_list.append(attacker)
                attacker_id_list.append(attacker.id)

        if len(defender_list) > 0:
            # calculate the cost matrix
            T = np.zeros((len(defender_list), len(attacker_list)))
            if self.init_assign:
                for i, defender in enumerate(defender_list):
                    for j, attacker in enumerate(attacker_list):
                        T[i, j] = get_init_cost(attacker, defender, world.targets[attacker.fake_target])
                self.init_assign = False
            else:
                for i, defender in enumerate(defender_list):
                    for j, attacker in enumerate(attacker_list):
                        #T[i, j] = get_energy_cost(attacker, defender, world.targets[attacker.fake_target])
                        T[i, j] = 0
            # print('T is:', T)
            # print('TAD list are:', target_id_list, defender_id_list, attacker_id_list)
            assign_result = target_assign(T)  # |D|*|A|的矩阵
            # print('assign_result is:', assign_result)

            '''
            如果assign报错，检查'TAD list，查看A的list是否为空。如果全部A被拦截，不需要update belief
            '''
            # update belief list of TDs according to assign_result
            for i, defender in enumerate(defender_list):  # 遍历行
                for j in range(len(attacker_list)):
                    if assign_result[i, j] == 1:
                        defender.attacker = attacker_list[j].id
                        defender.target = attacker_list[j].fake_target
                        attacker_list[j].defenders.append(defender.id)
                        target = world.targets[attacker_list[j].fake_target]
                        target.defenders.append(defender.id)
                        target.attackers.append(attacker_list[j].id)
                        target.cost.append(T[i, j])

            # 为target从list中选择AD
            for i, target in enumerate(target_list):
                if len(target.cost) > 0:
                    target.defender = target.defenders[np.argmin(target.cost)]
                    target.attacker = target.attackers[np.argmin(target.cost)]
                else:
                    # 有的target已经不需要AD了，AD太少了
                    target.defender = np.random.choice(defender_id_list)
                    target.attacker = np.random.choice(attacker_id_list)

            # print('T believes are:', )


'''
low-level policy for TADs
'''

def target_policy(target, attacker, defender, world):
    if target.select_policy == 0 or target.select_policy == 1:
        return np.array([0, 0, 0])

    elif target.select_policy == 2:
        r = 1.5  # 圆周运动半径
        
        omega = target.max_speed / r     
        a_mag = target.max_speed * omega 
        
        a_x = - a_mag * np.sin(omega * world.world_step * world.dt) 
        a_y = - a_mag * np.cos(omega * world.world_step * world.dt)
        a_z = 0.0

        u_q = np.array([a_x, a_y, a_z])
        norm_uq = np.linalg.norm(u_q)
        if norm_uq > target.max_accel:
            u_q = (u_q / norm_uq) * target.max_accel

        return u_q

    elif target.select_policy == 3:
        # 升角30度固定
        v_m = target.max_speed
        v_parallel = v_m * np.sqrt(3)/2                     # 轴向速度分量
        v_perp = v_m * 0.5              # 法向速度分量

        r = 0.1  # 螺旋半径
        
        omega = v_perp / r     
        a_mag = v_perp * omega
        
        a_x = 0.0 
        a_y = a_mag * np.sin(omega * world.world_step * world.dt)
        a_z = a_mag * np.cos(omega * world.world_step * world.dt)
        
        u_q = np.array([a_x, a_y, a_z])

        norm_uq = np.linalg.norm(u_q)
        if norm_uq > target.max_accel:
            u_q = (u_q / norm_uq) * target.max_accel

        # print(world.world_step, u_q, a_mag, np.dot(u_q, target.state.p_vel))
        return u_q
    

def defender_policy(attacker, defender):
    global N_md

    if attacker.done or defender.done:
        return np.array([0, 0, 0])

    # TAD基本关系 - 使用三维向量
    V_d = defender.state.p_vel
    V_a = attacker.state.p_vel

    # 计算attacker与defender的相对位置
    x_da = attacker.state.p_pos - defender.state.p_pos
    r_da = np.linalg.norm(x_da)
    e_da = x_da / r_da

    # 计算速度差异
    r_da_dot = np.dot(V_a - V_d, e_da)
    q_da_dot = np.cross(e_da, V_a - V_d) / r_da

    # 计算defender加速度
    if r_da_dot > 0:
        u_q = N_md * (e_da - V_d / np.linalg.norm(V_d))
    else:
        u_q = -N_md * r_da_dot * np.cross(q_da_dot, e_da)

    norm_uq = np.linalg.norm(u_q)
    if norm_uq > defender.max_accel:
        u_q = (u_q / norm_uq) * defender.max_accel
    return u_q

def attacker_policy(target, attacker, defender, world):
    if target.done or attacker.done:
        return np.array([0, 0, 0])
    # APNG策略
    if attacker.select_policy == 1:
        global N_ma
        # target = world.targets[attacker.fake_target]

        V_a = attacker.state.p_vel
        V_t = target.state.p_vel
        V_d = defender.state.p_vel

        # 计算相对位置
        x_at = target.state.p_pos - attacker.state.p_pos
        x_ad = defender.state.p_pos - attacker.state.p_pos
        r_at = np.linalg.norm(x_at)
        r_ad = np.linalg.norm(x_ad)

        e_at = x_at / r_at

        # 计算速度差异
        r_mt_dot = np.dot(V_t - V_a, e_at)
        q_mt_dot = np.cross(e_at, V_t - V_a) / r_at

        # 计算attacker加速度
        if r_mt_dot > 0:
                u_q = N_ma * (e_at - V_t / np.linalg.norm(V_t))
        else:
            u_q = -N_ma * r_mt_dot * np.cross(q_mt_dot, e_at)

        u_q += 0.5 * N_ma * np.cross(e_at, target.action.u)
        
        if r_ad < 4 and r_at > 4:
            e_ad = x_ad / r_ad

            # 计算速度差异
            r_mt_dot = np.dot(V_d - V_a, e_ad)
            q_mt_dot = np.cross(e_ad, V_d - V_a) / r_ad

            # 计算adtacker加速度
            if r_mt_dot > 0:
                    u_q1 = N_ma * (e_ad - V_d / np.linalg.norm(V_d))
            else:
                u_q1 = -N_ma * r_mt_dot * np.cross(q_mt_dot, e_ad)

            u_q1 += 0.5 * N_ma * np.cross(e_ad, defender.action.u)

            u_q -= u_q1
        
        norm_uq = np.linalg.norm(u_q)
        if norm_uq > attacker.max_accel:
            u_q = (u_q / norm_uq) * attacker.max_accel

        return u_q
    
    elif attacker.select_policy == 2:
        V_a = attacker.state.p_vel
        V_t = target.state.p_vel
        V_d = defender.state.p_vel

        r_ad = np.linalg.norm(defender.state.p_pos - attacker.state.p_pos)
        r_at = np.linalg.norm(target.state.p_pos - attacker.state.p_pos)

        # Define scalar variables (replace with actual values as needed)
        r_Te = 0.00011   # Example value
        r_Ap = 0.0001
        r_Ae = 0.00011
        r_Dp = 0.0001
        # t = world.world_step * world.dt
        # t_f1 = 0.6 * world.world_length * world.dt
        # t_f2 = 1.0 * world.world_length * world.dt
        R_AT = attacker.state.p_pos - target.state.p_pos
        R_DA = defender.state.p_pos - attacker.state.p_pos
        V_AT = attacker.state.p_vel - target.state.p_vel
        V_DA = defender.state.p_vel - attacker.state.p_vel
        X_AT = np.concatenate((R_AT, V_AT))
        X_DA = np.concatenate((R_DA, V_DA))


        dist_AT = np.linalg.norm(R_AT)
        dist_DA = np.linalg.norm(R_DA)
        t_r1 = abs(dist_AT **2 /np.dot(R_AT, V_AT))
        t_r2 = abs(dist_DA **2 /np.dot(R_DA, V_DA))

        # Compute intermediate variables
        # t_r1 = max(0, t_f1 - t)
        # t_r2 = max(0, t_f2 - t)

        gamma1 = (r_Te * r_Ap) / (r_Te - r_Ap)
        gamma2 = (r_Ae * r_Dp) / (r_Ae - r_Dp)
        den1 = 12.0 * gamma1 * gamma1  + 4.0 * gamma1 * t_r1 ** 3
        den2 = 12.0 * gamma2 * gamma2  + 4.0 * gamma2 * t_r2 ** 3

        N1 = N2 = np.zeros((3, 6))
        N1[0,0] = N1[1,1] = N1[2,2] = 12.0 * gamma1 * gamma1 * t_r1
        N1[0,3] = N1[1,4] = N1[2,5] = 12.0 * gamma1 * gamma1 * t_r1 * t_r1
        N2[0,0] = N2[1,1] = N2[2,2] = 12.0 * gamma2 * gamma2 * t_r2
        N2[0,3] = N2[1,4] = N2[2,5] = 12.0 * gamma2 * gamma2 * t_r2 * t_r2

        a_Ap = -1/(r_Ap * den1) * N1 @ X_AT
        a_Ae = -1/(r_Ae * den2) * N2 @ X_DA
        u_q = a_Ap + a_Ae

        norm_uq = np.linalg.norm(u_q)
        if norm_uq > attacker.max_accel:
            u_q = (u_q / norm_uq) * attacker.max_accel

        return u_q
    
class Target(Agent):
    def __init__(self):
        super().__init__()
        self.name = "target"
        self.id = None
        self.attackers = []
        self.defenders = []
        self.cost = []
        self.attacker = None
        self.defender = None
        self.select_policy = 0
        self.projectile = None
        self.launch_time = 5


class Attacker(Agent):
    def __init__(self):
        super().__init__()
        self.name = "attacker"
        self.id = None
        self.true_target = None
        self.fake_target = None
        self.defenders = []
        self.flag_kill = False
        self.flag_dead = False
        self.last_belief = None
        self.is_locked = False


class Defender(Agent):
    def __init__(self):
        super().__init__()
        self.name = "defender"
        self.id = None
        self.attacker = None
        self.target = None
        self.is_launched = True
        self.launcher = None


class TadWorld(World):
    """World with TAD-specific step/integration logic."""

    def step(self):
        # scripted policies for each role
        for agent in self.agents:
            if agent.name == "target":
                action = agent.action_callback(
                    agent,
                    self.attackers[agent.attacker],
                    self.defenders[agent.defender],
                    self,
                )
                agent.action.u = action
            elif agent.name == "attacker":
                action = agent.action_callback(
                    self.targets[agent.true_target],
                    agent,
                    self.defenders[agent.defenders[0]],
                    self,
                )
                agent.action.u = action
            elif agent.name == "defender":
                if agent.attacker is None or agent.attacker >= len(self.attackers):
                    action = np.zeros(3)
                else:
                    action = agent.action_callback(self.attackers[agent.attacker], agent)
                agent.action.u = action

        u = [None] * len(self.agents)
        u = self.apply_action_force(u)
        self.integrate_state(u)
        self.world_step += 1

        # collision / termination checks
        if hasattr(self, "scenario") and hasattr(self.scenario, "_update_termination_flags"):
            self.scenario._update_termination_flags(self)
            # update launch timing plan if enabled
            if hasattr(self.scenario, "maybe_update_launch_time"):
                self.scenario.maybe_update_launch_time(self)

    def integrate_state(self, u):
        for i, agent in enumerate(self.agents):
            v = agent.state.p_vel + u[i] * self.dt
            speed = np.linalg.norm(v)
            if speed != 0 and agent.max_speed is not None:
                v = v / speed * agent.max_speed

            if agent.done:
                agent.state.p_vel = np.zeros(3)
            else:
                v_x, v_y, v_z = v[0], v[1], v[2]
                theta = np.arctan2(v_y, v_x)
                if theta < 0:
                    theta += np.pi * 2
                horizontal_norm = np.sqrt(v_x ** 2 + v_y ** 2)
                elevation = np.arctan2(v_z, horizontal_norm)
                agent.state.phi = theta
                agent.state.theta = elevation
                agent.state.p_vel = np.array([v_x, v_y, v_z])
            agent.state.p_pos += agent.state.p_vel * self.dt
