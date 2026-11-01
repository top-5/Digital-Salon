from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def process_string(input_str):
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "extern/copilot/text_to_hair.py"), input_str], cwd=ROOT / "bin", check=True)
    return f"Processed: {input_str}"
