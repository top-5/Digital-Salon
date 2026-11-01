# Installing Digital Salon

## Linux (tested on Ubuntu 24.04, RTX 4090)

Use CUDA Toolkit 12.8, CMake 3.24 or newer, a C++17 compiler, and Python 3.12.
No driver or toolkit installation is performed by the project scripts.

```bash
sudo apt install libeigen3-dev libglm-dev nlohmann-json3-dev libassimp-dev \
  libboost-program-options-dev libboost-system-dev libboost-filesystem-dev \
  libfreeimage-dev libglfw3-dev libglew-dev freeglut3-dev libegl-dev \
  python3.12-dev python3.12-venv ninja-build
bash scripts/build.sh
bash scripts/setup_python.sh
```

The build defaults to `/usr/local/cuda-12.8/bin/nvcc` and `sm_89`. Override for another host:

```bash
bash scripts/build.sh -DCMAKE_CUDA_COMPILER=/path/to/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
```

`BUILD_JOBS` controls C++ build concurrency; `MAX_JOBS` controls Python CUDA extension builds (both default to 4).
The working vendored docking-enabled ImGui 1.90.2 WIP is retained. Linux uses system
Assimp, GLM, GLFW, GLEW and FreeImage rather than bundled headers or Windows binaries.
The viewer uses GLEW; a separate glad dependency is unnecessary.

Python is installed in `.venv` using pip, including torch 2.9.1+cu128, torchvision
0.24.1+cu128, numpy 2.5.3, transformers 4.57.1 and diffusers 0.35.2.
DRTK 0.1.0 is pinned to commit `43c3095d01d0ff435c798518e97e387c784b567f`
with a verified archive hash in `requirements.txt`. Build it with CUDA 12.8 and
`TORCH_CUDA_ARCH_LIST=8.9`; the setup script sets these defaults and installs torch
before building DRTK with `--no-build-isolation`. Setuptools 70.2.0 is needed by DRTK's
build script. No seed source tree is modified.

## Assets and viewer

The licensed head mesh must be obtained separately and is ignored by Git:

```bash
curl -fL https://www.avatarneo.com/PinscreenGenericHeadModel.obj -o data/meshes/head.obj
cd bin
./DigitalSalon
```

Run from `bin` because the viewer uses relative data/shader paths. The default
`data/hair/Jewfro.data` is already included. A desktop display is required for the
interactive viewer. Without DISPLAY it exits at GLFW initialization with a diagnostic.

The binary also offers a short, display-free EGL pbuffer context/readback probe:

```bash
env -u DISPLAY -u WAYLAND_DISPLAY bin/DigitalSalon --egl-smoke
```

This checks offscreen OpenGL only; it does not run the interactive scene or simulation.

## DRTK headless rendering

Python rendering uses DRTK's CUDA rasterizer without an OpenGL context or display.
OBJ meshes are triangulated; binary strands use camera-facing ribbons. The preview
is shaded RGB on white using an auto-framed orthographic camera. Meshes and strands
must use the same coordinates when supplied together. `--yaw`, `--radius` and
`--size` control view angle, ribbon half-width in source units, and output resolution.

```bash
.venv/bin/python extern/rendering/drtk_renderer.py \
  --mesh data/meshes/head.obj --output tmp/head-drtk.png
.venv/bin/python extern/rendering/drtk_renderer.py \
  --hair data/hair/Jewfro.data --yaw 25 --output tmp/hair-drtk.png
```

Run `.venv/bin/python scripts/smoke_render.py` for CUDA occlusion checks, head
and hair PNGs, and DRTK-to-Canny conditioning without model weights.

This is a lightweight preview/conditioning renderer, with flat shading and no
physically based materials or differentiable camera interface.

## Copilot data

Obtain the original project's [s3.zip](https://yaleedu-my.sharepoint.com/:u:/g/personal/chengan_he_yale_edu/EUT5lUdAZbFHhvnlyRYF29cBImNMupDEKDMqSnhSF36qgw?e=i4vri5)
and [usc-hair-resampled.zip](https://yaleedu-my.sharepoint.com/:u:/g/personal/chengan_he_yale_edu/EZUbewJcyzpCo7iMjwZr7MQBPr_gLmKFzP8_l5rQDbYioQ?e=2bwevv).
They are not fetched by setup. Extract so these files exist:

- `data/copilot/s3/label/codes_usc.json` (each entry has `name` and `code`).
- `data/copilot/usc-hair-resampled/<name>.data` for each codebook hairstyle.

Override roots with `DIGITAL_SALON_COPILOT_DATA` and `DIGITAL_SALON_HAIR_DATA`.
Copilot uses Hugging Face `openai/clip-vit-base-patch32`; supply/cache its tokenizer
and weights before using Copilot offline. No models are fetched by the build or smoke checks.
Chosen hairstyle previews are rendered from strands with DRTK instead of copied
precomputed JPEGs. The viewer starts the project `.venv` Python directly.

## ControlNet models

The legacy external ControlNet checkout and separate environment are replaced with
the pinned diffusers pipeline. Supply **diffusers-format directories**, including
configs, tokenizer, scheduler and weights:

- Stable Diffusion 1.5 in `data/models/stable-diffusion-v1-5/`.
- SD 1.5 Canny ControlNet (`lllyasviel/control_v11p_sd15_canny`) in `data/models/controlnet-canny/`.

Override with `DIGITAL_SALON_SD_MODEL` and `DIGITAL_SALON_CONTROLNET_MODEL`.
Old `.ckpt`/`.pth` files alone are not sufficient; convert them outside this setup
or obtain the corresponding diffusers exports. Model loading uses `local_files_only=True`.
No weights are downloaded or inference/training runs performed by setup.

The GUI still provides its OpenGL screenshot at `tmp/controlnet/input.png`.
For a fully headless DRTK mesh/strand conditioning path, run from the repo root:

```bash
.venv/bin/python extern/controlnet/canny_controlnet_api.py \
  female curly front "soft lighting" --mesh data/meshes/head.obj \
  --conditioning-only --output tmp/controlnet/canny-smoke.png
```

Remove `--conditioning-only` once local models are available to generate an image.
`--hair` may be supplied with or without `--mesh`. Canny conditioning is written to
`conditioning.png` beside the requested output. Model-backed Copilot retrieval and
ControlNet inference require the owner's assets and have not been validated by the
asset-free smoke checks.

## Windows

The original `cmake/windows.cmake` remains available but has not been validated by
the Linux upgrade. Adjust its local dependency paths before configuring. The new
EGL probe is Linux-only.
