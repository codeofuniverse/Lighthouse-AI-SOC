"""Promote the SHAP-validated k-fold models to the canonical production paths.

CIC: the 18-feature V2 model (the 17 flow features + dst_port) is fully Suricata-
reproducible and roughly tripled Web-Attack recall while cutting benign FP ~70%,
so it replaces the legacy 17-feature production model. The live bridge already
slices the feature vector to the loaded model's feature count, so serving picks up
18 parameters with no further change.

UNSW: the 28-feature SHAP-discovered model is NOT promoted to live serving here —
it needs proto/service/ct_*/sttl that a Suricata flow record cannot supply (that's
the Zeek track). It is kept as the high-accuracy offline/benchmark model. The
existing 11-feature production UNSW model is left in place for Suricata-only serving.

Each overwrite archives the previous file to <name>.bak first.

Usage:
    python scripts/promote_canonical.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import joblib

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

MODELS = Path("data/models")
CIC_SRC = MODELS / "cic2017_kfold_v2_pipeline.joblib"
CIC_DST = MODELS / "cic2017_pipeline_smote.joblib"


def _archive(dst: Path) -> None:
    if dst.exists():
        bak = dst.with_suffix(dst.suffix + ".bak")
        shutil.copy2(dst, bak)
        print(f"  archived {dst.name} -> {bak.name}")


def main() -> None:
    print("=" * 70)
    print("  Promote canonical models")
    print("=" * 70)
    if not CIC_SRC.exists():
        raise FileNotFoundError(f"{CIC_SRC} missing — run scripts/train_kfold_cic.py --features v2")

    pipe = joblib.load(CIC_SRC)
    assert pipe["features"][-1] == "dst_port" and len(pipe["features"]) == 18, \
        "CIC source is not the 18-feature V2 model"
    assert {"scaler", "stage1_model", "stage2_model", "fam_encoder", "features"} <= set(pipe), \
        "CIC source missing production-schema keys"

    _archive(CIC_DST)
    joblib.dump(pipe, CIC_DST)
    print(f"  promoted V2 -> {CIC_DST.name}  ({len(pipe['features'])} features, "
          f"val macro-F1={pipe['meta'].get('val_macro_f1', 'n/a')})")
    print("\n  UNSW: 28-feature model kept as offline benchmark "
          "(unsw_kfold_pipeline.joblib); live UNSW path unchanged (needs Zeek).")
    print("  Done.")


if __name__ == "__main__":
    main()
