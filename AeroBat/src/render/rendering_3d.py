"""
3D rendering framework rewritten to use ModernGL (core profile) while
preserving the original Viewer/Geom API used by the environments.
"""
from __future__ import division
import os
import six
import sys
import math
import numpy as np

try:
    from gymnasium import error
except ImportError:  # gymnasium may be absent; fall back to std error.
    class _DummyError(Exception):
        pass
    class error:  # type: ignore
        Error = _DummyError

import pyglet
from pyglet import gl as pyglet_gl
import moderngl

RAD2DEG = 57.29577951308232

# --------------------------------------------------------------------------- #
# Matrix helpers                                                              #
# --------------------------------------------------------------------------- #

def mat_to_bytes(mat: np.ndarray) -> bytes:
    """Convert a row-major 4x4 matrix into column-major bytes for OpenGL."""
    return np.asarray(mat, dtype="f4").T.tobytes()


def perspective(fovy_deg: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    fovy = math.radians(fovy_deg)
    f = 1.0 / math.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (z_far + z_near) / (z_near - z_far)
    m[2, 3] = (2 * z_far * z_near) / (z_near - z_far)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _vec3(v):
    """Ensure a 3-component vector."""
    if len(v) == 3:
        return v
    return (v[0], v[1], 0.0)


def get_display(spec):
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(f"Invalid display spec: {spec}")


# --------------------------------------------------------------------------- #
# Viewer                                                                      #
# --------------------------------------------------------------------------- #

class Viewer(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)
        self.width = width
        self.height = height
        config = pyglet_gl.Config(double_buffer=True, sample_buffers=1, samples=4)

        self.window = pyglet.window.Window(
            width=width, height=height, display=display, config=config
        )
        self.window.on_close = self.window_closed_by_user
        self.window.on_mouse_drag = self.on_mouse_drag
        self.window.on_mouse_scroll = self.on_mouse_scroll
        self.window.on_mouse_press = self.on_mouse_press
        self.window.on_key_press = self.on_key_press

        # Scene content
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        # Camera parameters
        self.camera_pos = [300, 200, 0]
        self.camera_target = [20, 0, 15]
        self.camera_up = [0, 0, 1]
        self.fov = 90.0
        self.z_near = 0.1
        self.z_far = 300.0

        self.camera_radius = 125.0
        self.camera_theta = 0.4353981633974482
        self.camera_phi = 0.6435987755982986
        self._default_camera = (
            self.camera_radius,
            self.camera_theta,
            self.camera_phi,
            tuple(self.camera_target),
        )

        self.last_mouse_x = 0
        self.rotate_sensitivity = 0.004
        self.rotate_vertical_sensitivity = 0.004
        self.pan_sensitivity = 0.004
        self.zoom_sensitivity = 0.25

        # ModernGL context
        self.window.switch_to()
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.programs = self._build_programs()

        self._vp = np.eye(4, dtype=np.float32)

    def _build_programs(self):
        solid = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 mvp;
                in vec3 in_pos;
                void main() {
                    gl_Position = mvp * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec4 u_color;
                out vec4 f_color;
                void main() {
                    f_color = u_color;
                }
            """,
        )
        return {"solid": solid}

    # ------------------------------------------------------------------ camera
    def set_projection(self):
        self.projection = perspective(
            self.fov, self.window.width / float(self.window.height), self.z_near, self.z_far
        )

    def update_camera_position(self):
        self.camera_pos[0] = self.camera_radius * math.cos(self.camera_theta) * math.cos(
            self.camera_phi
        )
        self.camera_pos[1] = self.camera_radius * math.sin(self.camera_theta) * math.cos(
            self.camera_phi
        )
        self.camera_pos[2] = self.camera_radius * math.sin(self.camera_phi)

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.last_mouse_x = x

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & pyglet.window.mouse.LEFT:
            delta_theta = dx * self.rotate_sensitivity
            self.camera_theta += delta_theta
            self.camera_phi = max(
                -math.radians(85),
                min(math.radians(85), self.camera_phi + dy * self.rotate_vertical_sensitivity),
            )
        elif buttons & pyglet.window.mouse.MIDDLE:
            # pan camera target in view plane (world-aligned up)
            right = np.array(
                [
                    math.cos(self.camera_theta + math.pi / 2.0),
                    math.sin(self.camera_theta + math.pi / 2.0),
                    0,
                ],
                dtype=float,
            )
            up = np.array([0, 0, 1], dtype=float)
            move = -dx * self.pan_sensitivity * right + dy * self.pan_sensitivity * up
            self.camera_target = list(np.array(self.camera_target) + move)
        self.update_camera_position()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.camera_radius = max(1.0, self.camera_radius - scroll_y * self.zoom_sensitivity)
        self.update_camera_position()

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.R:
            self.camera_radius, self.camera_theta, self.camera_phi, tgt = self._default_camera
            self.camera_target = list(tgt)
            self.update_camera_position()
        elif symbol == pyglet.window.key.F:
            self.camera_radius = max(1.0, self.camera_radius * 0.5)
            self.update_camera_position()

    def set_camera(self):
        eye = np.array(self.camera_pos, dtype=np.float32)
        target = np.array(self.camera_target, dtype=np.float32)
        up = np.array(self.camera_up, dtype=np.float32)
        self.view = look_at(eye, target, up)

    def look_at(self, target_x, target_y, target_z):
        self.camera_target = [target_x, target_y, target_z]

    def set_position(self, x, y, z):
        dx = x - self.camera_target[0]
        dy = y - self.camera_target[1]
        dz = z - self.camera_target[2]

        self.camera_radius = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        self.camera_theta = math.atan2(dy, dx)
        xy_distance = math.sqrt(dx ** 2 + dy ** 2)
        self.camera_phi = math.atan2(dz, xy_distance)
        self.update_camera_position()
        self.camera_pos[0] = dx
        self.camera_pos[1] = dy
        self.camera_pos[2] = dz

    # ----------------------------------------------------------------- drawing
    def set_light(self):
        # Lighting is handled in shaders for ModernGL; keep stub for API parity.
        pass

    def _render_lines(self, vertices, color, linewidth=1.0):
        if len(vertices) == 0:
            return
        data = np.array([_vec3(v) for v in vertices], dtype="f4")
        program = self.programs["solid"]
        program["mvp"].write(mat_to_bytes(self._vp))
        program["u_color"].value = tuple(color)
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.simple_vertex_array(program, vbo, "in_pos")
        self.ctx.line_width = linewidth
        vao.render(mode=moderngl.LINES)

    def drawCoordinate(self):
        # 1 km grid only in the positive octant
        step = 1.0
        line_num = 30  # 30 km span along positive axes
        line_len = step * line_num
        axis_colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
        grid_color = (0.5, 0.5, 0.5, 0.25)

        # Positive axes only
        for i in range(3):
            end = [0, 0, 0]
            end[i] = line_len
            self._render_lines([(0, 0, 0), tuple(end)], axis_colors[i], linewidth=3)

        # Grid lines on each principal plane (positive quadrant)
        coords = range(0, line_num + 1)
        for i in range(3):
            fixed_axis = i
            moving_axes = [j for j in range(3) if j != fixed_axis]
            for offset in coords:
                pos = offset * step
                start = [0, 0, 0]
                start[fixed_axis] = pos
                a0 = start.copy()
                a1 = start.copy()
                a0[moving_axes[0]] = 0
                a1[moving_axes[0]] = line_len
                b0 = start.copy()
                b1 = start.copy()
                b0[moving_axes[1]] = 0
                b1[moving_axes[1]] = line_len
                self._render_lines([tuple(a0), tuple(a1)], grid_color, linewidth=1.3)
                self._render_lines([tuple(b0), tuple(b1)], grid_color, linewidth=1.3)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top, front, back):
        self.transform.set_scale(
            (right - left) / 2,
            (top - bottom) / 2,
            (back - front) / 2,
        )
        self.transform.set_translation(
            -(left + right) / 2,
            -(bottom + top) / 2,
            -(front + back) / 2,
        )

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        arr = None
        self.set_light()
        self.window.switch_to()
        self.window.dispatch_events()
        self.ctx.viewport = (0, 0, self.window.width, self.window.height)
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)

        self.set_projection()
        self.update_camera_position()
        self.set_camera()
        base_model = self.transform.matrix()
        self._vp = self.projection @ self.view @ base_model

        self.drawCoordinate()

        for geom in self.geoms:
            geom.render(self.ctx, self._vp, self.programs)
        for geom in self.onetime_geoms:
            geom.render(self.ctx, self._vp, self.programs)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]

        self.window.flip()
        self.onetime_geoms = []
        return arr if return_rgb_array else None

    # Convenience helpers matching old API
    def draw_circle(self, radius=1, res=30, axis="z", **attrs):
        geom = make_disc(radius=radius, res=res, axis=axis)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_sphere(self, radius=1, **attrs):
        geom = make_sphere(radius)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v, filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom


# --------------------------------------------------------------------------- #
# Geometry primitives                                                         #
# --------------------------------------------------------------------------- #

class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


class Geom(object):
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self, ctx, vp, programs):
        color, model, linewidth = self._collect_state()
        self.render1(ctx, vp, programs, color, model, linewidth)

    def _collect_state(self):
        color = self._color.vec4
        model = np.eye(4, dtype=np.float32)
        linewidth = 1.0
        for attr in self.attrs:
            if isinstance(attr, Transform):
                model = model @ attr.matrix()
            elif isinstance(attr, LineWidth):
                linewidth = attr.stroke
            elif isinstance(attr, Color):
                color = attr.vec4
        return color, model, linewidth

    def render1(self, ctx, vp, programs, color, model, linewidth):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)

    def set_linewidth(self, stroke):
        # Replace or append a LineWidth attribute
        for attr in self.attrs:
            if isinstance(attr, LineWidth):
                attr.stroke = stroke
                return
        self.add_attr(LineWidth(stroke))


class Transform(Attr):
    def __init__(self, translation=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        self.set_translation(*translation)
        self.set_rotation(*rotation)
        self.set_scale(*scale)

    def matrix(self):
        tx, ty, tz = self.translation
        sx, sy, sz = self.scale
        rx, ry, rz = self.rotation

        T = np.eye(4, dtype=np.float32)
        T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz

        cx, sx_ = math.cos(rx), math.sin(rx)
        cy, sy_ = math.cos(ry), math.sin(ry)
        cz, sz_ = math.cos(rz), math.sin(rz)

        Rx = np.array(
            [[1, 0, 0, 0], [0, cx, -sx_, 0], [0, sx_, cx, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        Ry = np.array(
            [[cy, 0, sy_, 0], [0, 1, 0, 0], [-sy_, 0, cy, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        Rz = np.array(
            [[cz, -sz_, 0, 0], [sz_, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        S = np.diag([sx, sy, sz, 1]).astype(np.float32)

        return T @ Rx @ Ry @ Rz @ S

    def set_translation(self, x, y, z):
        self.translation = (x, y, z)

    def set_rotation(self, rx, ry, rz):
        self.rotation = (rx, ry, rz)

    def set_scale(self, sx, sy, sz):
        self.scale = (sx, sy, sz)


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke


class Line(Geom):
    def __init__(self, start, end):
        super().__init__()
        self.start = np.array(start, dtype="f4")
        self.end = np.array(end, dtype="f4")
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
        self._vao = None
        self._vbo = None

    def _ensure_gpu(self, ctx, programs):
        if self._vao is None:
            data = np.vstack([self.start, self.end]).astype("f4")
            self._vbo = ctx.buffer(data.tobytes())
            self._vao = ctx.simple_vertex_array(programs["solid"], self._vbo, "in_pos")

    def render1(self, ctx, vp, programs, color, model, linewidth):
        self._ensure_gpu(ctx, programs)
        program = programs["solid"]
        mvp = vp @ model
        program["mvp"].write(mat_to_bytes(mvp))
        program["u_color"].value = tuple(color)
        ctx.line_width = linewidth
        self._vao.render(mode=moderngl.LINES)


class Point(Geom):
    def __init__(self):
        super().__init__()
        self._vao = None

    def _ensure_gpu(self, ctx, programs):
        if self._vao is None:
            data = np.array([[0.0, 0.0, 0.0]], dtype="f4")
            vbo = ctx.buffer(data.tobytes())
            self._vao = ctx.simple_vertex_array(programs["solid"], vbo, "in_pos")

    def render1(self, ctx, vp, programs, color, model, linewidth):
        self._ensure_gpu(ctx, programs)
        program = programs["solid"]
        mvp = vp @ model
        program["mvp"].write(mat_to_bytes(mvp))
        program["u_color"].value = tuple(color)
        self._vao.render(mode=moderngl.POINTS)


class PolyLine(Geom):
    def __init__(self, vertices, closed):
        super().__init__()
        self.vertices = [(_vec3(v)) for v in vertices]
        self.closed = closed
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
        self._vao = None
        self._vbo = None

    def _ensure_gpu(self, ctx, programs):
        if self._vao is None:
            verts = self.vertices
            if self.closed and verts[0] != verts[-1]:
                verts = verts + [verts[0]]
            data = np.array(verts, dtype="f4")
            self._vbo = ctx.buffer(data.tobytes())
            self._vao = ctx.simple_vertex_array(programs["solid"], self._vbo, "in_pos")

    def render1(self, ctx, vp, programs, color, model, linewidth):
        self._ensure_gpu(ctx, programs)
        program = programs["solid"]
        mvp = vp @ model
        program["mvp"].write(mat_to_bytes(mvp))
        program["u_color"].value = tuple(color)
        ctx.line_width = linewidth
        mode = moderngl.LINE_STRIP
        self._vao.render(mode=mode)


class FilledPolygon(Geom):
    def __init__(self, v):
        super().__init__()
        self.v = [(_vec3(p)) for p in v]
        self._vao_fill = None
        self._vao_line = None
        self._vbo_fill = None
        self._vbo_line = None
        self._ibo = None

    def _triangulate(self):
        indices = []
        for i in range(1, len(self.v) - 1):
            indices.extend([0, i, i + 1])
        return indices

    def _ensure_gpu(self, ctx, programs):
        if self._vao_fill is None:
            verts = np.array(self.v, dtype="f4")
            indices = np.array(self._triangulate(), dtype="i4")
            self._vbo_fill = ctx.buffer(verts.tobytes())
            self._ibo = ctx.buffer(indices.tobytes())
            self._vao_fill = ctx.vertex_array(
                programs["solid"], [(self._vbo_fill, "3f", "in_pos")], self._ibo
            )
            self._vbo_line = ctx.buffer(verts.tobytes())
            self._vao_line = ctx.simple_vertex_array(programs["solid"], self._vbo_line, "in_pos")

    def render1(self, ctx, vp, programs, color, model, linewidth):
        self._ensure_gpu(ctx, programs)
        program = programs["solid"]
        mvp = vp @ model
        program["mvp"].write(mat_to_bytes(mvp))

        # Fill
        program["u_color"].value = tuple(color)
        self._vao_fill.render(mode=moderngl.TRIANGLES)

        # Outline (darker)
        outline_color = (
            color[0] * 0.5,
            color[1] * 0.5,
            color[2] * 0.5,
            color[3] * 0.5,
        )
        program["u_color"].value = outline_color
        ctx.line_width = linewidth
        self._vao_line.render(mode=moderngl.LINE_LOOP)


def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius, 0))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_polygon(vertices, filled=True):
    verts = [(_vec3(v)) for v in vertices]
    return FilledPolygon(verts) if filled else PolyLine(verts, closed=True)


class FilledMesh(Geom):
    def __init__(self, v, faces, color=(1.0, 0.0, 0.0, 1.0)):
        super().__init__()
        self.v = np.array(v, dtype="f4")
        self.faces = np.array(faces, dtype=np.int32)
        self._vao = None
        self._vbo = None
        self._ibo = None
        self.set_color(*color)

    def _ensure_gpu(self, ctx, programs):
        if self._vao is None:
            self._vbo = ctx.buffer(self.v.astype("f4").tobytes())
            self._ibo = ctx.buffer(self.faces.astype("i4").tobytes())
            self._vao = ctx.vertex_array(
                programs["solid"], [(self._vbo, "3f", "in_pos")], self._ibo
            )

    def render1(self, ctx, vp, programs, color, model, linewidth):
        self._ensure_gpu(ctx, programs)
        program = programs["solid"]
        mvp = vp @ model
        program["mvp"].write(mat_to_bytes(mvp))
        program["u_color"].value = tuple(color)
        self._vao.render(mode=moderngl.TRIANGLES)


def make_disc(radius=1, res=30, axis="z"):
    vertices = []
    for i in range(res):
        theta = 2 * math.pi * i / res
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        if axis == "z":
            vertices.append((x, y, 0))
        elif axis == "x":
            vertices.append((0, x, y))
        else:
            vertices.append((x, 0, y))
    return FilledPolygon(vertices)


def make_sphere(radius=1.0, l_segments=30, a_segments=30, filled=True):
    vertices = []
    for i in range(l_segments + 1):
        theta = i * 2 * math.pi / l_segments
        for j in range(a_segments + 1):
            phi = j * math.pi / a_segments - math.pi / 2
            x = radius * math.cos(phi) * math.cos(theta)
            y = radius * math.cos(phi) * math.sin(theta)
            z = radius * math.sin(phi)
            vertices.append((x, y, z))

    indices = []
    for i in range(l_segments):
        for j in range(a_segments):
            p1 = i * (a_segments + 1) + j
            p2 = p1 + a_segments + 1
            indices.extend([(p1, p2, p1 + 1), (p1 + 1, p2, p2 + 1)])

    return FilledMesh(vertices, indices)
