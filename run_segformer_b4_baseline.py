from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CONFIG_NAME = "segformer_b4_baseline.yaml"


def main() -> None:
    root = Path(__file__).resolve().parent
    train_path = root / "src" / "train.py"
    config_path = root / "configs" / CONFIG_NAME

    if not train_path.exists():
        raise FileNotFoundError(f"train.py not found: {train_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cmd = [
        sys.executable,
        str(train_path),
        "--config",
        str(config_path),
    ]

    print("\n### Running SegFormer-B4 Baseline ###", flush=True)
    print("Command:", " ".join(cmd), flush=True)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
