import gymnasium as gym
import numpy as np
from pathlib import Path
from collections import deque

# Camera range used for auto-centering (kept for compatibility)
cam_range = 5
max_trace_len = 800

class MultiAgentEnv(gym.Env):
    """
    Lightweight environment wrapper for visualization.
    Reinforcement-learning specific logic (action/observation/reward) has been
    removed; this class now only handles world reset/step and 3D rendering.
    Scenario-specific behavior should be implemented on the world/scenario
    objects, not here.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world,
        reset_callback=None,
        reward_callback=None,  # ignored (kept for backward compatibility)
        observation_callback=None,  # ignored
        info_callback=None,  # ignored
        done_callback=None,  # ignored
        update_belief=None,  # ignored
        post_step_callback=None,
        shared_viewer=True,
        discrete_action=False,  # ignored
    ):
        self.world = world
        self.scenario = getattr(world, "scenario", None)
        self.reset_callback = reset_callback
        self.post_step_callback = post_step_callback
        self.shared_viewer = shared_viewer

        # viewer / geometry caches
        self.viewers = [None] if shared_viewer else [None] * max(1, len(getattr(world, "agents", [])))
        self.render_geoms = []
        self.render_geoms_xform = []
        self.comm_geoms = []
        self.line = []
        self._camera_initialized = False
        self.traces = []
        self._reset_render()

    def reset(self):
        if self.reset_callback is not None:
            self.reset_callback(self.world)
        self._reset_render()
        return None

    def step(self, action=None):
        # Actions are scenario-specific; the world drives its own logic.
        self.world.step()
        if self.post_step_callback is not None:
            self.post_step_callback(self.world)
        return None

    def close(self):
        self.render(close=True)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _reset_render(self):
        self.render_geoms = []
        self.render_geoms_xform = []
        self.comm_geoms = []
        self.line = []
        self._camera_initialized = False
        self.traces = []

    def _init_camera(self):
        """Place the viewer so that agents start near the center of view."""
        if self._camera_initialized or not getattr(self.world, "entities", None):
            return

        positions = np.array(
            [getattr(e.state, "p_pos", np.zeros(3)) for e in self.world.entities],
            dtype=float,
        )
        scale = cam_range  # world coords are scaled by cam_range in set_bounds
        positions *= scale

        center = positions.mean(axis=0) if len(positions) > 0 else np.zeros(3)
        max_offset = (
            np.linalg.norm(positions - center, axis=1).max() if len(positions) > 0 else 0.0
        )
        radius = max(25.0, max_offset * 2.2)

        for viewer in self.viewers:
            viewer.look_at(*center)
            viewer.camera_radius = radius
            viewer.update_camera_position()
            viewer._default_camera = (
                viewer.camera_radius,
                viewer.camera_theta,
                viewer.camera_phi,
                tuple(viewer.camera_target),
            )

        self._camera_initialized = True

    def render(self, mode="human", close=False):
        from ..render import rendering_3d as rendering
        from ..render import OBJ

        if close:
            for viewer in self.viewers:
                if viewer is not None:
                    viewer.close()
            return None

        # lazily create viewers
        for i in range(len(self.viewers)):
            if self.viewers[i] is None:
                self.viewers[i] = rendering.Viewer(1200, 800)

        # set the initial camera once viewers exist
        self._init_camera()

        # build static geometry once
        if self.render_geoms == []:
            self.render_geoms = []
            self.render_geoms_xform = []
            self.line = {}
            self.comm_geoms = []

            for entity in self.world.entities:
                geom = None
                model_path = getattr(entity, "model_path", None)
                if model_path:
                    p = Path(model_path)
                    if p.exists():
                        geom = OBJ(fdir=str(p.parent) + "/", filename=p.name, swapyz=True)
                        try:
                            geom.create_bbox(target_center=(0, 0, 0), target_half_size=getattr(entity, "size", 0.05))
                        except Exception:
                            # Fall back to raw geometry if normalization fails
                            pass
                        if getattr(entity, "color", None) is not None:
                            geom.set_color(*entity.color)
                if geom is None:
                    geom = rendering.make_sphere(radius=getattr(entity, "size", 0.05))
                    color = getattr(entity, "color", (0.5, 0.5, 0.5))
                    geom.set_color(*color)

                xform = rendering.Transform()
                geom.add_attr(xform)

                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                self.comm_geoms.append([])

            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

            # initialize trajectory buffers
            self.traces = [deque(maxlen=max_trace_len) for _ in self.world.entities]

        results = []
        for i in range(len(self.viewers)):
            # set simple bounds
            self.viewers[i].set_bounds(-cam_range, cam_range, -cam_range, cam_range, -cam_range, cam_range)

            # update transforms
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                # Align model orientation with velocity direction
                speed = np.linalg.norm(entity.state.p_vel)
                # self.line[e] = self.viewers[i].draw_line(entity.state.p_pos, entity.state.p_pos+entity.state.p_vel*0.4)
                
                if speed > 1e-6:
                    vx, vy, vz = entity.state.p_vel
                    dx, dy, dz = vx / speed, vy / speed, vz / speed
                    horizontal_norm = np.sqrt(dx * dx + dy * dy)
                    if dy >= 0:
                        pitch = np.arctan2(dy, dx)
                    else:
                        pitch = np.arctan2(dy, dx) + np.pi * 2
                    if dx >= 0:
                        yaw = np.arctan2(-dz, horizontal_norm)
                    else:
                        yaw = np.arctan2(dz, horizontal_norm)
                    # rotation order is Rz(yaw) after Ry(pitch); Rx kept zero
                    self.render_geoms_xform[e].set_rotation(0, yaw, pitch)
                # update trajectory
                if self.traces and e < len(self.traces):
                    self.traces[e].append(tuple(entity.state.p_pos))

            # scenario-specific overlays (e.g., connector lines) live in scenario
            if self.scenario and hasattr(self.scenario, "render_overlays"):
                self.scenario.render_overlays(self.world, self.viewers)

            # draw trajectories as polylines (onetime geoms)
            if self.traces:
                for e, trace in enumerate(self.traces):
                    if len(trace) < 2:
                        continue
                    color = getattr(self.world.entities[e], "color", (0.2, 0.2, 0.2))
                    points = list(trace)
                    for p0, p1 in zip(points[:-1], points[1:]):
                        seg = self.viewers[i].draw_line(p0, p1)
                        seg.set_color(*color, alpha=0.6)
                        seg.set_linewidth(2.0)

            results.append(
                self.viewers[i].render(return_rgb_array=mode == "rgb_array")
            )

        return results
