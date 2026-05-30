from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CONFIG_ORDER = [
    "baseline.yaml",
    "boundary.yaml",
    "context.yaml",
    "boundary_context.yaml",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    for config_name in CONFIG_ORDER:
        config_path = root / "configs" / config_name
        print(f"\n### Running {config_path} ###", flush=True)
        subprocess.run([sys.executable, str(root / "src" / "train.py"), "--config", str(config_path)], check=True)


if __name__ == "__main__":
    main()
