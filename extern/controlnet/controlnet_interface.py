from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def process_string(input_str):
    # The viewer sends four quoted prompt fields.
    fields = shlex.split(input_str)
    if len(fields) != 4:
        raise ValueError("Expected gender, hairstyle, head pose, and miscellaneous prompt")
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "extern/controlnet/canny_controlnet_api.py"), *fields], cwd=ROOT / "bin", check=True)
    return f"Processed: {input_str}"
