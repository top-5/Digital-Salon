"""Short, model-free CUDA rendering check; run with .venv/bin/python."""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import drtk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extern/rendering"))
sys.path.insert(0, str(ROOT / "extern/controlnet"))
from drtk_renderer import render_scene
from canny_controlnet_api import conditioning_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Two overlapping triangles: smaller positive camera Z must win.
    positions = torch.tensor([[[2, 2, 2], [14, 2, 2], [2, 14, 2],
                               [2, 2, 1], [14, 2, 1], [2, 14, 1]]],
                             device="cuda", dtype=torch.float32)
    triangles = torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32)
    indices = drtk.rasterize(positions, triangles, 16, 16)
    assert indices[0, 4, 4].item() == 1, "DRTK depth ordering failed"
    assert indices[0, 15, 15].item() == -1, "DRTK background mask failed"
    print(f"DRTK {drtk.__version__}: CUDA occlusion/background checks passed")
    for name, options in (("head", {"mesh": ROOT / "data/meshes/head.obj"}),
                          ("hair", {"hair": ROOT / "data/hair/Jewfro.data", "yaw": 25})):
        image = render_scene(**options, size=512)
        foreground = np.any(image < 250, axis=2)
        assert image.shape == (512, 512, 3) and image.dtype == np.uint8
        assert 1000 < foreground.sum() < 512 * 512 * 0.9, "Empty or clipped render"
        path = args.output_dir / f"{name}-drtk.png"
        Image.fromarray(image).save(path)
        with Image.open(path) as saved:
            assert saved.size == (512, 512) and saved.format == "PNG"
        print(f"{path}: {int(foreground.sum())} foreground pixels")
    canny = conditioning_image(mesh=ROOT / "data/meshes/head.obj")
    assert np.asarray(canny).max() == 255 and np.asarray(canny).min() == 0
    canny.save(args.output_dir / "conditioning-drtk.png")
    print("DRTK -> ControlNet Canny conditioning passed; no model weights loaded")


if __name__ == "__main__":
    main()
