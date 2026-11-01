import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(Path(__file__).resolve().parent / "codeBook"))
import shutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rendering"))
from drtk_renderer import render_scene
from PIL import Image



data_root = os.environ.get("DIGITAL_SALON_COPILOT_DATA", str(Path(__file__).resolve().parents[2] / "data/copilot/s3")) + "/"
hair_data_root = os.environ.get("DIGITAL_SALON_HAIR_DATA", str(Path(__file__).resolve().parents[2] / "data/copilot/usc-hair-resampled")) + "/"
tmp_folder = str(Path(__file__).resolve().parents[2] / "tmp/copilot") + "/"

if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)
label_folder = data_root + "label/"

# features_fn = label_folder + "features.json"
features_fn = label_folder + "codes_usc.json"

test_fn = label_folder + "test_prompts.json"


if not Path(features_fn).is_file():
    raise FileNotFoundError(f"Supply the Copilot codebook at {features_fn}; see INSTALL.md")

with open(features_fn, "r") as f:
    code_book = json.load(f)

from featureFetcher import device, dist, get_feature

## prepare the code matrix

code_len = len(code_book[0]["code"][0])
code_num = len(code_book)

# title_matrix = torch.ones(code_num, code_len).to(device)
description_matrix = torch.ones(code_num, code_len).to(device)

for i in range(code_num):
    # c = torch.FloatTensor(code_book[i]["title_code"]).to(device).view(-1)
    # title_matrix[i] = c
    c = torch.FloatTensor(code_book[i]["code"]).to(device).view(-1)
    description_matrix[i] = c


print("finish loading the code matrix.t")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_string>")
        sys.exit(1)

    input_str = sys.argv[1]

    text_feature, _ = get_feature(input_str)
    similarities = dist(description_matrix, text_feature).view(
        -1
    )  # Resulting in a tensor of size N
    topk_values, topk_indices = torch.topk(similarities, 3, largest=True)
    print(topk_indices.shape)
    for i in range(topk_indices.shape[0]):
        print(i)
        k = topk_indices[i]

        hairname = code_book[k]["name"]
        print(str(i + 1), hairname)

        data_fn = f"{hair_data_root}/{hairname}.data"

        preview = render_scene(hair=data_fn, yaw=25)
        Image.fromarray(preview).save(tmp_folder + "tmp" + str(i + 1) + ".jpg")
        shutil.copyfile(data_fn, tmp_folder + "tmp" + str(i + 1) + ".data")
