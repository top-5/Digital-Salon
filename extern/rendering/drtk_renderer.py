"""Headless CUDA previews and ControlNet conditioning using pinned DRTK.

OBJ meshes and Digital Salon/USC binary strands share one orthographic camera.
Strands become camera-facing ribbons; output is a shaded RGB image on white.
"""
from pathlib import Path
import argparse
import struct

import numpy as np
from PIL import Image
import torch
import drtk


def load_obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == "f":
            indices = [int(x.split("/")[0]) for x in fields[1:]]
            indices = [i - 1 if i > 0 else len(vertices) + i for i in indices]
            faces.extend((indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1))
    if not vertices or not faces:
        raise ValueError(f"No mesh triangles in {path}")
    return np.asarray(vertices, np.float32), np.asarray(faces, np.int32)


def load_strands(path):
    with open(path, "rb") as stream:
        def count():
            return struct.unpack("<i", stream.read(4))[0]
        n = count()
        if not 0 < n < 10_000_000:
            raise ValueError("Invalid strand count")
        strands = []
        for _ in range(n):
            length = count()
            if not 0 < length < 1_000_000:
                raise ValueError("Invalid strand length")
            points = np.frombuffer(stream.read(length * 12), dtype="<f4").reshape(length, 3)
            if length > 1:
                strands.append(points)
    if not strands:
        raise ValueError("No strand segments")
    return strands


def strand_mesh(strands, radius):
    starts = np.concatenate([s[:-1] for s in strands])
    ends = np.concatenate([s[1:] for s in strands])
    delta = ends - starts
    side = np.stack((-delta[:, 1], delta[:, 0], np.zeros(len(delta))), axis=1)
    norm = np.linalg.norm(side, axis=1, keepdims=True)
    side[norm[:, 0] < 1e-10] = (1, 0, 0)
    side = side / np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-10) * radius
    vertices = np.stack((starts - side, starts + side, ends + side, ends - side), axis=1).reshape(-1, 3)
    base = np.arange(len(starts), dtype=np.int32)[:, None] * 4
    faces = np.concatenate((base + (0, 1, 2), base + (0, 2, 3)))
    return vertices, faces


@torch.inference_mode()
def render_scene(mesh=None, hair=None, size=512, radius=None, yaw=0):
    """Return uint8 HWC RGB. No display, model weights, or OpenGL context required."""
    if size < 16 or size > 4096:
        raise ValueError("Image size must be between 16 and 4096")
    if not torch.cuda.is_available():
        raise RuntimeError("DRTK rendering requires a CUDA GPU")
    parts = []
    if mesh:
        v, f = load_obj(mesh)
        parts.append((v, f, (0.73, 0.57, 0.45)))
    if hair:
        strands = load_strands(hair)
        extent = np.ptp(np.concatenate(strands), axis=0).max()
        v, f = strand_mesh(strands, radius if radius is not None else extent / 1500)
        parts.append((v, f, (0.12, 0.065, 0.035)))
    if not parts:
        raise ValueError("Supply a mesh or hair file")
    vertices, faces, colors, offset = [], [], [], 0
    for v, f, color in parts:
        vertices.append(v)
        faces.append(f + offset)
        colors.extend([color] * len(f))
        offset += len(v)
    v = np.concatenate(vertices)
    angle = np.deg2rad(yaw)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]], np.float32)
    v = v @ rotation.T
    v -= (v.min(0) + v.max(0)) / 2
    extent = np.ptp(v, axis=0).max()
    if not np.isfinite(v).all() or extent <= 0:
        raise ValueError("Invalid scene geometry")
    v /= extent
    f = np.concatenate(faces)
    # OBJ +Y is up; image +Y is down. Camera looks toward -Z.
    projected = v.copy()
    projected[:, 0] = (v[:, 0] * 0.86 + 0.5) * size - 0.5
    projected[:, 1] = (-v[:, 1] * 0.86 + 0.5) * size - 0.5
    projected[:, 2] = 2 - v[:, 2]
    normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    light = np.array([0.3, 0.4, 1], np.float32)
    light /= np.linalg.norm(light)
    shading = 0.35 + 0.65 * np.abs(normals @ light)
    face_colors = torch.as_tensor(np.asarray(colors) * shading[:, None], device="cuda", dtype=torch.float32)
    positions = torch.as_tensor(projected[None], device="cuda").contiguous()
    triangles = torch.as_tensor(f, device="cuda", dtype=torch.int32).contiguous()
    indices = drtk.rasterize(positions, triangles, size, size)[0]
    rgb = torch.where((indices >= 0)[..., None], face_colors[indices.clamp_min(0).long()], 1.0)
    return (rgb.clamp(0, 1) * 255).byte().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--hair", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--yaw", type=float, default=0)
    args = parser.parse_args()
    image = render_scene(args.mesh, args.hair, args.size, args.radius, args.yaw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output)
    print(f"DRTK {drtk.__version__}: wrote {args.output} ({args.size}x{args.size})")


if __name__ == "__main__":
    main()
