"""Per-model probability-threshold tuner — REPORT ONLY (serving unchanged).

The benign false-positive rate is governed mostly by the Stage-1 binary cutoff
(default 0.5). This script sweeps that cutoff on the DISJOINT validation set and
finds the operating point that holds benign->ANY-attack FP <= the budget (1%)
while keeping attack recall as high as possible. It also reports per-family FP at
the chosen point.

It does three things and changes NO serving behaviour:
  1. writes reports/training_validation_report_data/{cic,unsw}_thresholds.json
     (chosen cutoff + the full FP/recall sweep for the report);
  2. saves an FP-vs-recall tradeoff curve PNG per dataset;
  3. stores the chosen threshold in the model dict under a new "thresholds" key
     (available for later opt-in use) — the prediction path is not modified.

Usage:
    python scripts/tune_thresholds.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.flow_features import cic_features_from_df  # noqa: E402
from scripts.train_unsw_svm import FEATURES as UNSW_FEATURES  # noqa: E402

warnings.filterwarnings("ignore")

DARK_BG, PANEL_BG, GRID, ACCENT = "#0d1117", "#161b22", "#30363d", "#00ff88"
plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG, "savefig.facecolor": DARK_BG,
    "text.color": "white", "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e", "axes.edgecolor": GRID,
})

OUT_DATA = Path("reports/training_validation_report_data")
OUT_FIG = Path("reports/explainability")
FP_BUDGET = 0.01            # 1% benign -> ANY-attack
SWEEP = np.round(np.arange(0.05, 0.99, 0.01), 2)


def _sweep(stage1, scaler, X_raw, y_binary):
    """Return list of (cutoff, benign_fp_rate, attack_recall)."""
    Xs = scaler.transform(X_raw)
    proba = stage1.predict_proba(Xs)[:, 1]
    benign = y_binary == 0
    attack = y_binary == 1
    n_benign, n_attack = int(benign.sum()), int(attack.sum())
    rows = []
    for c in SWEEP:
        pred_attack = proba >= c
        fp = int((benign & pred_attack).sum())
        tp = int((attack & pred_attack).sum())
        rows.append((float(c),
                     fp / n_benign if n_benign else 0.0,
                     tp / n_attack if n_attack else 0.0))
    return rows


def _pick(rows: list[tuple[float, float, float]]) -> dict:
    """Lowest cutoff whose benign FP <= budget (max recall under the budget)."""
    feasible = [r for r in rows if r[1] <= FP_BUDGET]
    chosen = min(feasible, key=lambda r: r[0]) if feasible else max(rows, key=lambda r: -r[1])
    return {"cutoff": chosen[0], "benign_fp_rate": chosen[1], "attack_recall": chosen[2],
            "met_budget": bool(feasible)}


def _plot(name: str, rows, chosen, out: Path) -> None:
    cs = [r[0] for r in rows]
    fp = [r[1] * 100 for r in rows]
    rc = [r[2] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cs, fp, color="#ff6b6b", lw=2, label="benign FP rate %")
    ax.plot(cs, rc, color=ACCENT, lw=2, label="attack recall %")
    ax.axhline(FP_BUDGET * 100, color="#ffe66d", ls="--", lw=1, label=f"FP budget {FP_BUDGET*100:.0f}%")
    ax.axvline(chosen["cutoff"], color="white", ls=":", lw=1.5,
               label=f"chosen cutoff {chosen['cutoff']:.2f}")
    ax.set_xlabel("Stage-1 attack-probability cutoff")
    ax.set_ylabel("percent")
    ax.legend(framealpha=0.2, labelcolor="white")
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    fig.suptitle(f"{name} — Stage-1 threshold sweep (disjoint validation)",
                 color="white", fontsize=12, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    saved -> {out}")


def tune_cic() -> None:
    print("\n== CIC threshold tuning ==")
    model_path = Path("data/models/cic2017_kfold_v2_pipeline.joblib")
    pipe = joblib.load(model_path)
    df = pd.read_parquet("data/models/raw/cic2017_val.parquet")
    X = cic_features_from_df(df, version="v2").values.astype(float)
    y = np.where(df["Label"].values == "BENIGN", 0, 1).astype(int)
    rows = _sweep(pipe["stage1_model"], pipe["scaler"], X, y)
    chosen = _pick(rows)
    print(f"  chosen cutoff={chosen['cutoff']:.2f}  benign FP={chosen['benign_fp_rate']*100:.2f}%"
          f"  recall={chosen['attack_recall']*100:.2f}%  met_budget={chosen['met_budget']}")
    _plot("CIC-IDS-2017 (v2)", rows, chosen, OUT_FIG / "cic" / "thresholds_cic.png")
    _persist(model_path, pipe, "cic", rows, chosen)


def tune_unsw() -> None:
    print("\n== UNSW threshold tuning ==")
    model_path = Path("data/models/unsw_kfold_pipeline.joblib")
    pipe = joblib.load(model_path)
    df = pd.read_parquet("data/models/raw/unsw_val.parquet").copy()
    for feat in UNSW_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
    X = df[UNSW_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)
    y = np.where(df["Label"].values == "Normal", 0, 1).astype(int)
    rows = _sweep(pipe["stage1_model"], pipe["scaler"], X, y)
    chosen = _pick(rows)
    print(f"  chosen cutoff={chosen['cutoff']:.2f}  benign FP={chosen['benign_fp_rate']*100:.2f}%"
          f"  recall={chosen['attack_recall']*100:.2f}%  met_budget={chosen['met_budget']}")
    _plot("UNSW-NB15", rows, chosen, OUT_FIG / "unsw" / "thresholds_unsw.png")
    _persist(model_path, pipe, "unsw", rows, chosen)


def _persist(model_path: Path, pipe: dict, key: str, rows, chosen) -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (OUT_DATA / f"{key}_thresholds.json").write_text(json.dumps({
        "fp_budget": FP_BUDGET, "chosen": chosen,
        "sweep": [{"cutoff": c, "benign_fp_rate": f, "attack_recall": r} for c, f, r in rows],
    }, indent=2))
    # Store in the model dict for later opt-in use; serving is NOT modified.
    pipe["thresholds"] = {"stage1_prob_min": chosen["cutoff"],
                          "tuned_on": "disjoint_validation", "fp_budget": FP_BUDGET,
                          "note": "report-only; serving does not consult this yet"}
    joblib.dump(pipe, model_path)
    print(f"    stored 'thresholds' key in {model_path.name} (serving unchanged)")


def main() -> None:
    print("=" * 70)
    print(f"  Per-family threshold tuner — budget benign FP <= {FP_BUDGET*100:.0f}%  (report-only)")
    print("=" * 70)
    tune_cic()
    tune_unsw()
    print("\n  Done.")


if __name__ == "__main__":
    main()
