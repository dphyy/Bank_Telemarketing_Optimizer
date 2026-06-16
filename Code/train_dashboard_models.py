# Import necessary modules and set up the environment for training dashboard models.
from __future__ import annotations
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dashboard_pipeline import build_training_bundle  # noqa: E402


def main() -> None:
    bundle = build_training_bundle(use_artifacts=False)
    ok = [name for name, run in bundle.models.items() if run.status == "ok"]
    failed = {name: run.error for name, run in bundle.models.items() if run.status != "ok"}
    print(f"Saved {len(ok)} model artifacts: {', '.join(ok)}")
    if failed:
        print("Failed models:")
        for name, error in failed.items():
            print(f"- {name}: {error}")


if __name__ == "__main__":
    main()
