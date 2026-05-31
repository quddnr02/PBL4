from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    config_path = root / "configs" / "boundary_context.yaml"
    aug_config_path = root / "configs" / "augmentation_extra.yaml"
    subprocess.run(
        [
            sys.executable,
            str(root / "src" / "train.py"),
            "--config",
            str(config_path),
            "--aug-config",
            str(aug_config_path),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
