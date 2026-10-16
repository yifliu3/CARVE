"""Convert uploaded meshes into the local preview geometry contract.

Adapted from the asset importer: dataset delivery/catalog code is not included.
"""

import hashlib
import math
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .settings import CFG, write

FORMATS = {".glb", ".gltf", ".obj", ".ply", ".stl", ".npz"}
MAX_EXPANDED = 768 * 1024 * 1024
MAX_TRIANGLES = CFG["max_triangles"]


def validate_geometry(vertices, colors):
    vertices = np.asarray(vertices, dtype=np.float32)
    colors = np.asarray(colors)
    if vertices.ndim != 3 or vertices.shape[1:] != (3, 3) or not 0 < len(vertices) <= MAX_TRIANGLES:
        raise ValueError(f"需要三角网格，最多 {MAX_TRIANGLES:,} 个三角形")
    if not np.isfinite(vertices).all() or np.abs(vertices).max() > 1e7:
        raise ValueError("网格坐标必须为有限数值且在支持的范围内")
    if colors.shape == (len(vertices), 3):
        colors = np.repeat(colors[:, None, :], 3, axis=1)
    if colors.shape != vertices.shape or not np.isfinite(colors).all():
        raise ValueError("颜色须与三角形或顶点对应")
    if colors.min() < 0 or colors.max() > 255:
        raise ValueError("颜色值必须在 0–255 之间")
    return vertices, colors.astype(np.uint8)


def save_environment(out, vertices_m, colors, *, scale, origin, K, metadata):
    vertices, colors = validate_geometry(vertices_m, colors)
    origin = np.asarray(origin, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if not math.isfinite(scale) or scale <= 0 or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Invalid environment normalization")
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Invalid camera intrinsics")
    env = out / "environment"
    env.mkdir(exist_ok=True, parents=True)
    ((vertices.astype(np.float64) - origin) / scale).astype("<f4").tofile(env / "positions.bin")
    colors.tofile(env / "colors.bin")
    bounds = np.stack([vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))])
    scene = {**metadata, "width": 960, "height": 544, "K": K.tolist(),
             "origin_m": origin.tolist(), "normalization_scale_m": float(scale),
             "initial_pose": [0, 0, 0, 0, 0], "triangles": len(vertices),
             "bounds_m": bounds.tolist(), "source_type": "3d", "units": "meters",
             "axes": "X right, Y forward, Z up", "proxy_renderer": "cwm_demo_webgl_v1"}
    np.savez_compressed(env / "geometry.npz", vertices=vertices, colors=colors, K=K)
    write(env / "scene.json", scene)
    return scene


def unpack_zip(path, directory):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 1024 or sum(i.file_size for i in members) > MAX_EXPANDED:
            raise ValueError("压缩包过大：最多 1024 个文件、解压后 768 MB")
        for member in members:
            relative = PurePosixPath(member.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or ":" in member.filename or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("压缩包包含不安全的路径或符号链接")
            target = directory.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
    candidates = [p for p in directory.rglob("*") if p.suffix.lower() in FORMATS and "__MACOSX" not in p.parts]
    preferred = [p for p in candidates if p.suffix.lower() in {".gltf", ".glb"}]
    candidates = preferred or candidates
    if len(candidates) != 1:
        raise ValueError("ZIP 需包含一个主网格文件及其材质、纹理资源")
    return candidates[0]


class LocalResolver:
    """Resource paths in glTF/OBJ stay inside the uploaded package."""
    def __init__(self, root, base=None):
        self.root = Path(root).resolve()
        self.base = Path(base or root).resolve()

    def get(self, key):
        from urllib.parse import unquote
        key = unquote(str(key)).replace("\\", "/")
        if ":" in key or key.startswith("/"):
            raise ValueError("网格只允许引用上传包内的资源")
        path = (self.base / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("网格资源路径超出上传包")
        return path.read_bytes()

    def __getitem__(self, key):
        return self.get(key)

    def namespaced(self, namespace):
        base = (self.base / str(namespace)).resolve()
        if not base.is_relative_to(self.root):
            raise ValueError("Invalid resource namespace")
        return LocalResolver(self.root, base)

    def keys(self):
        return [str(p.relative_to(self.base)) for p in self.base.rglob("*") if p.is_file()]


def npz_mesh(path):
    # np.load's pickle loader remains disabled for uploaded archives.
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > MAX_EXPANDED:
            raise ValueError("NPZ 解压后过大")
    with np.load(path, allow_pickle=False) as data:
        v = data["vertices"]
        faces = data["faces"] if "faces" in data else None
        if v.ndim == 2 and v.shape[1] == 3 and faces is not None:
            if faces.ndim != 2 or faces.shape[1] != 3 or not 0 < len(faces) <= MAX_TRIANGLES or faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= len(v):
                raise ValueError("Invalid triangle indices")
            triangles = v[faces]
        elif v.ndim == 3 and v.shape[1:] == (3, 3):
            triangles = v
        else:
            raise ValueError("NPZ 需有 vertices[T,3,3]，或 vertices[N,3] 和 faces[T,3]")
        c = data["colors"] if "colors" in data else np.tile([99, 130, 158], (len(triangles), 1))
        if c.ndim == 2 and c.shape[1] == 4:
            c = c[:, :3]
        if faces is not None and c.shape == v.shape:
            c = c[faces]
        return validate_geometry(triangles, c)


def read_mesh(path, package_root):
    if path.suffix.lower() == ".npz":
        return npz_mesh(path)
    import trimesh
    # A file-like object plus a restrictive resolver avoids arbitrary local/remote reads.
    with path.open("rb") as stream:
        scene = trimesh.load_scene(stream, file_type=path.suffix[1:].lower(),
            resolver=LocalResolver(package_root, path.parent), allow_remote=False, process=False)
    vertices, colors = [], []
    count = 0
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        count += len(mesh.faces)
        if count > MAX_TRIANGLES:
            raise ValueError(f"资产超过 {MAX_TRIANGLES:,} 个三角形，请先简化")
        visual = mesh.visual
        if visual.kind == "texture":
            if visual.uv is None:
                # glTF frequently stores a solid PBR baseColorFactor without UVs.
                rgba = np.asarray(visual.material.main_color)
                c = np.tile(rgba[:3], (len(mesh.faces), 3, 1))
            else:
                c = np.asarray(visual.to_color().vertex_colors)[:, :3][mesh.faces]
        else:
            c = np.asarray(visual.vertex_colors)[:, :3][mesh.faces]
        if mesh.visual.kind is None:
            c[:] = [99, 130, 158]
        v = trimesh.transform_points(mesh.vertices, transform)[mesh.faces]
        vertices.append(v)
        colors.append(c)
    if not vertices:
        raise ValueError("文件中没有可渲染的三角网格")
    return validate_geometry(np.concatenate(vertices), np.concatenate(colors))


def load_upload(path, out, *, up_axis, units, progress, reference=None):
    if up_axis not in {"y", "z"} or units not in {"m", "cm", "mm"}:
        raise ValueError("未知的坐标轴或单位")
    progress("asset", "读取并转换上传的 3D 资产")
    package_root = path.parent
    if path.suffix.lower() == ".zip":
        package_root = out / "unpacked"
        package_root.mkdir()
        path = unpack_zip(path, package_root)
    if path.suffix.lower() not in FORMATS:
        raise ValueError("支持 GLB、glTF ZIP、OBJ、PLY、STL、NPZ")
    v, colors = read_mesh(path, package_root)
    factor = {"m": 1., "cm": .01, "mm": .001}[units]
    v = v.astype(np.float64) * factor
    if up_axis == "y":
        v = v[..., [0, 2, 1]] * [1, -1, 1]
    lower, upper = v.min(axis=(0, 1)), v.max(axis=(0, 1))
    center = (lower + upper) / 2
    extents = upper - lower
    if extents.max() < 1e-5:
        raise ValueError("网格尺寸过小")
    if extents.max() > 500:
        raise ValueError("资产跨度超过 500 米，请检查所选单位或先裁剪场景")
    focal = 480 / math.tan(math.radians(60) / 2)
    distance = max(extents[0] / 2 * focal / 480, extents[2] / 2 * focal / 272, .2) * 1.15 + extents[1] / 2
    camera = center - [0, distance, 0]
    local = v - camera
    scale = max(float(np.linalg.norm(extents)) / 2, 1)
    scene = save_environment(out, local, colors, scale=scale, origin=center-camera,
        K=[[focal, 0, 480], [0, focal, 272], [0, 0, 1]],
        metadata={"name": path.stem, "asset_source": "upload", "upload_units": units,
                  "upload_up_axis": up_axis, "reference_source": "uploaded image" if reference else "not supplied",
                  "geometry_kind": "mesh with vertex/material colors"})
    if reference:
        with Image.open(reference) as source:
            if source.width * source.height > 40000000:
                raise ValueError("参考图像素过多")
            source.convert("RGB").save(out / "reference.png")
    write(out / "asset_receipt.json", {"source_type": "3d", "asset_source": "upload",
        "filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "triangles": scene["triangles"], "units": units, "up_axis": up_axis,
        "reference_source": scene["reference_source"], "texture_conversion": "sampled vertex colors; no PBR lighting",
        "image_reconstruction": False})
    return scene
