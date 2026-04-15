from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "inputs"
REFERENCE_DIR = ROOT / "data" / "reference" / "paper_v6_experiments"

CANDIDATES = {
    "utam_shapefile.zip": [
        Path("/Users/jorgeiv500/Downloads/a. Shapefile UTAM.zip"),
        Path("/Users/jorgeiv500/Downloads/2.PublicacionSIMUR/EODH/03_Zonificacion/a. Shapefile UTAM.zip"),
    ],
    "priority_utam_scores.csv": [
        Path("/Users/jorgeiv500/Downloads/priority_utam_outputs_enriched (1)/priority_utam_scores.csv"),
    ],
    "temporal_attention_by_utam.csv": [
        Path("/Users/jorgeiv500/Downloads/priority_utam_outputs_enriched (1)/temporal_attention_by_utam.csv"),
    ],
    "learning_loss_history.csv": [
        Path("/Users/jorgeiv500/Downloads/priority_utam_outputs_enriched (1)/learning_loss_history.csv"),
    ],
    "utam_social_vulnerability.csv": [
        Path("/Users/jorgeiv500/Library/CloudStorage/GoogleDrive-jorgeiv500@gmail.com/Mi unidad/0. archivos para colab_sdewes/parta_outputs/social_equity_followup_outputs/utam_social_vulnerability.csv"),
    ],
    "utam_travel_time_matrix.csv": [
        Path("/Users/jorgeiv500/Library/CloudStorage/GoogleDrive-jorgeiv500@gmail.com/Mi unidad/0. archivos para colab_sdewes/parta_outputs/social_equity_followup_outputs/utam_travel_time_matrix.csv"),
    ],
}
REPO_NATIVE_INPUTS = [
    "bogota_official_ev_growth.csv",
]

REFERENCE_SOURCE = Path(
    "/Users/jorgeiv500/Library/CloudStorage/GoogleDrive-jorgeiv500@gmail.com/Mi unidad/0. archivos para colab_sdewes/parta_outputs/social_equity_followup_outputs/paper_v6_experiments"
)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> int:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"inputs": {}, "reference_dir": None}

    missing = []
    for target_name, candidates in CANDIDATES.items():
        source = first_existing(candidates)
        target = INPUT_DIR / target_name
        if source is None:
            missing.append(target_name)
            continue
        copy_file(source, target)
        manifest["inputs"][target_name] = {"source": str(source), "staged_to": str(target)}

    if REFERENCE_SOURCE.exists():
        if REFERENCE_DIR.exists():
            shutil.rmtree(REFERENCE_DIR)
        shutil.copytree(REFERENCE_SOURCE, REFERENCE_DIR)
        manifest["reference_dir"] = {"source": str(REFERENCE_SOURCE), "staged_to": str(REFERENCE_DIR)}

    for target_name in REPO_NATIVE_INPUTS:
        target = INPUT_DIR / target_name
        if target.exists():
            manifest["inputs"][target_name] = {"source": "repo_native", "staged_to": str(target)}

    (INPUT_DIR / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if missing:
        raise SystemExit(f"Missing staged files: {', '.join(missing)}")

    print("Staged inputs in:", INPUT_DIR)
    if manifest["reference_dir"]:
        print("Staged reference outputs in:", REFERENCE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
