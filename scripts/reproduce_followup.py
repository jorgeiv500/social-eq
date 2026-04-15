from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from social_eq.followup import cli


if __name__ == "__main__":
    raise SystemExit(cli(["--root", str(ROOT)]))
