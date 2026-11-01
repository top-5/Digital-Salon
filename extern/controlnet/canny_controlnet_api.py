"""Local diffusers ControlNet Canny inference; never downloads model weights."""
import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]


def conditioning_image(image_path=None, mesh=None, hair=None, size=512):
    if mesh or hair:
        sys.path.insert(0, str(ROOT / "extern/rendering"))
        from drtk_renderer import render_scene
        image = render_scene(mesh=mesh, hair=hair, size=size)
    else:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        side = min(h, w)
        image = image[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]
        image = cv2.resize(image, (size, size))
    edges = cv2.Canny(image, 100, 200)
    return Image.fromarray(np.repeat(edges[..., None], 3, axis=2))


def controlnet_api(image, gender, hairstyle, headpose, misc_prompt="", steps=20, seed=12345):
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    sd_path = Path(os.environ.get("DIGITAL_SALON_SD_MODEL", ROOT / "data/models/stable-diffusion-v1-5"))
    control_path = Path(os.environ.get("DIGITAL_SALON_CONTROLNET_MODEL", ROOT / "data/models/controlnet-canny"))
    for path in (sd_path, control_path):
        if not path.is_dir():
            raise FileNotFoundError(f"Supply a local diffusers model directory: {path}. See INSTALL.md.")
    controlnet = ControlNetModel.from_pretrained(control_path, torch_dtype=torch.float16, local_files_only=True)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        sd_path, controlnet=controlnet, torch_dtype=torch.float16, local_files_only=True)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")
    prompt = f"a {gender} with {hairstyle} hair, {headpose}, {misc_prompt}, photorealistic, soft lighting"
    return pipeline(prompt, image=image, num_inference_steps=steps,
                    negative_prompt="lowres, bad anatomy, cropped, worst quality",
                    generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gender", "hairstyle", "headpose", "misc"):
        parser.add_argument(name)
    parser.add_argument("--input", type=Path, default=ROOT / "tmp/controlnet/input.png")
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--hair", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/controlnet/output.png")
    parser.add_argument("--conditioning-only", action="store_true")
    args = parser.parse_args()
    image = conditioning_image(args.input, args.mesh, args.hair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output.parent / "conditioning.png")
    if args.conditioning_only:
        image.save(args.output)
    else:
        controlnet_api(image, args.gender, args.hairstyle, args.headpose, args.misc).save(args.output)


if __name__ == "__main__":
    main()
