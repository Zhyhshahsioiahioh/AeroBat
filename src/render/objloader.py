import numpy as np

from .rendering_3d import Geom, mat_to_bytes


def MTL(fdir, filename):
    contents = {}
    mtl = None
    for line in open(fdir + filename, "r", encoding="utf-8"):
        if line.startswith("#"):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == "newmtl":
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError("mtl file doesn't start with newmtl stmt")
        else:
            mtl[values[0]] = [float(x) for x in values[1:4]]
    return contents


class OBJ(Geom):
    def __init__(self, fdir, filename, swapyz=False):
        """Loads a Wavefront OBJ file."""
        super().__init__()
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        self.mtl = None

        self._positions = None
        self._indices = None
        self._vao = None
        self._vbo = None
        self._ibo = None

        for line in open(fdir + filename, "r", encoding="utf-8"):
            if line.startswith("#"):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == "v":
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == "vn":
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == "vt":
                v = [float(x) for x in values[1:3]]
                self.texcoords.append(v)
            elif values[0] in ("usemtl", "usemat"):
                material = values[1]
            elif values[0] == "mtllib":
                self.mtl = [fdir, values[1]]
            elif values[0] == "f":
                face = []
                norms = []
                texcoords = []
                for v in values[1:]:
                    w = v.split("/")
                    face.append(int(w[0]))
                    texcoords.append(int(w[1]) if len(w) >= 2 and len(w[1]) > 0 else 0)
                    norms.append(int(w[2]) if len(w) >= 3 and len(w[2]) > 0 else 0)
                self.faces.append((face, norms, texcoords, material if "material" in locals() else None))

    def create_bbox(self, target_center=(0, 0, 0), target_half_size=1.0):
        ps = np.array(self.vertices)
        vmin = ps.min(axis=0)
        vmax = ps.max(axis=0)
        original_center = (vmax + vmin) / 2
        original_half_size = np.max(vmax - vmin) / 2

        scale_factor = target_half_size / original_half_size
        self.vertices = (ps - original_center) * scale_factor + target_center

        ps_new = np.array(self.vertices)
        vmin_new = ps_new.min(axis=0)
        vmax_new = ps_new.max(axis=0)
        self.bbox_center = (vmax_new + vmin_new) / 2
        self.bbox_half_r = np.max(vmax_new - vmin_new) / 2

    def _build_arrays(self):
        if isinstance(self.mtl, list):
            self.mtl = MTL(*self.mtl)

        positions = []
        indices = []
        for face in self.faces:
            verts, _, _, material = face
            base = len(positions)
            for vid in verts:
                positions.append(self.vertices[vid - 1])
            for k in range(1, len(verts) - 1):
                indices.extend([base, base + k, base + k + 1])
        self._positions = np.array(positions, dtype="f4") if positions else np.zeros((0, 3), dtype="f4")
        self._indices = np.array(indices, dtype="i4") if indices else np.zeros((0,), dtype="i4")

    def create_gl_list(self):
        """Kept for API compatibility; precomputes vertex/index arrays."""
        self._build_arrays()

    def _ensure_gpu(self, ctx, programs):
        if self._vao is None:
            if self._positions is None:
                self._build_arrays()
            if len(self._positions) == 0:
                return
            self._vbo = ctx.buffer(self._positions.tobytes())
            self._ibo = ctx.buffer(self._indices.tobytes())
            self._vao = ctx.vertex_array(
                programs["solid"], [(self._vbo, "3f", "in_pos")], self._ibo
            )

    def render1(self, ctx, vp, programs, color, model, linewidth):
        self._ensure_gpu(ctx, programs)
        if self._vao is None:
            return
        program = programs["solid"]
        mvp = vp @ model
        program["mvp"].write(mat_to_bytes(mvp))
        program["u_color"].value = tuple(color)
        self._vao.render()

