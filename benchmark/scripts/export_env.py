#!/usr/bin/env python3
"""
export_env.py
Export environment and hardware info for reproducibility:
- python version
- platform
- pip freeze
- torch + cuda info
Outputs text files into out_dir.
"""
from __future__ import annotations
import argparse, platform, subprocess, sys
from pathlib import Path

def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"ERROR: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "env_python.txt").write_text(sys.version.replace("\n", " ") + "\n", encoding="utf-8")
    (out / "env_platform.txt").write_text(platform.platform() + "\n", encoding="utf-8")
    (out / "env_pip_freeze.txt").write_text(run([sys.executable, "-m", "pip", "freeze"]) + "\n", encoding="utf-8")

    try:
        import torch
        lines = []
        lines.append(f"torch={torch.__version__}")
        lines.append(f"cuda_available={torch.cuda.is_available()}")
        lines.append(f"cuda_version={torch.version.cuda}")
        if torch.cuda.is_available():
            lines.append(f"gpu_name={torch.cuda.get_device_name(0)}")
            lines.append(f"gpu_count={torch.cuda.device_count()}")
        (out / "env_torch.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        (out / "env_torch.txt").write_text(f"ERROR: {e}\n", encoding="utf-8")

    print("[OK] Wrote env files to:", out)

if __name__ == "__main__":
    main()
