"""Explainable-AI report: SHAP + LIME for EVERY attack type, both datasets.

Loads the k-fold models (cic2017_kfold_pipeline.joblib, unsw_kfold_pipeline.joblib)
and their proportional subsets, then for each dataset produces:

  SHAP
    * Stage-1 (XGBoost binary) global beeswarm — what drives BENIGN-vs-attack.
    * Stage-2 (LightGBM multiclass) per-attack-family mean |SHAP| bars — which of
      the flow features shape each attack (CIC: 6 families incl. Web Attack;
      UNSW: 6 incl. DoS / Reconnaissance / Shellcode / Worms).
    * CIC only: per Web-Attack SUBTYPE (Brute Force / XSS / SQLi) attribution,
      using the preserved LabelFine column.

  LIME
    * One local explanation per attack type (a correctly-flagged representative).
    * A benign FALSE-POSITIVE case study: a benign flow the Stage-1 model flags as
      attack, explained feature-by-feature — shows WHY benign traffic flips.

Figures -> reports/explainability/{cic,unsw}/

Usage:
    python scripts/explain_attacks.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.flow_features import cic_features_from_df  # noqa: E402
from scripts.train_unsw_svm import FEATURES as UNSW_FEATURES  # noqa: E402

warnings.filterwarnings("ignore")

# Dark theme — match scripts/generate_model_reports.py
DARK_BG, PANEL_BG, GRID = "#0d1117", "#161b22", "#30363d"
ACCENT = "#00ff88"
plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
    "savefig.facecolor": DARK_BG, "text.color": "white",
    "axes.labelcolor": "#8b949e", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "axes.edgecolor": GRID,
})

OUT_ROOT = Path("reports/explainability")
SAMPLE_CAP = 4000        # cap rows fed to SHAP for speed
LIME_PER_CLASS = 1


def _slug(name: str) -> str:
    """Filesystem-safe ASCII slug (CIC's 'Web Attack  XSS' has non-ASCII separators)."""
    out = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name))
    return "_".join(filter(None, out.split("_")))


def _savefig(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.suptitle(title, color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    saved -> {path}")


def _shap_stage1(stage1, X_scaled: np.ndarray, features: list[str], out: Path) -> None:
    expl = shap.TreeExplainer(stage1)
    sv = expl.shap_values(X_scaled)
    sv = sv[1] if isinstance(sv, list) else sv
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_scaled, feature_names=features, show=False, color_bar=True)
    _savefig(out, "Stage 1 (BENIGN vs Attack) — SHAP feature impact")


def _shap_stage2_per_family(stage2, X_att: pd.DataFrame, class_names: list[str],
                            out_dir: Path, tag: str) -> None:
    """One mean|SHAP| bar chart per attack family."""
    expl = shap.TreeExplainer(stage2)
    sv = expl.shap_values(X_att)            # list (per class) or 3D array
    feats = list(X_att.columns)
    for ci, cls in enumerate(class_names):
        if isinstance(sv, list):
            if ci >= len(sv):
                continue
            cls_sv = sv[ci]
        else:
            cls_sv = sv[:, :, ci] if sv.ndim == 3 else sv
        mean_abs = np.abs(cls_sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        plt.figure(figsize=(9, 6))
        plt.barh([feats[i] for i in order][::-1], mean_abs[order][::-1], color=ACCENT)
        plt.xlabel("mean |SHAP value|")
        safe = _slug(cls)
        _savefig(out_dir / f"shap_{tag}_{safe}.png",
                 f"{cls} — feature attribution (mean |SHAP|)")


def _lime_case(explainer: LimeTabularExplainer, predict_fn, x_row: np.ndarray,
               features: list[str], out: Path, title: str, n_classes: int) -> None:
    exp = explainer.explain_instance(x_row, predict_fn,
                                     num_features=min(10, len(features)),
                                     labels=list(range(n_classes)))
    label = exp.available_labels()[0]
    pairs = exp.as_list(label=label)
    names = [p[0] for p in pairs][::-1]
    weights = [p[1] for p in pairs][::-1]
    colors = [ACCENT if w >= 0 else "#ff6b6b" for w in weights]
    plt.figure(figsize=(9, 6))
    plt.barh(names, weights, color=colors)
    plt.axvline(0, color=GRID, lw=1)
    plt.xlabel("LIME local weight (green=supports prediction, red=opposes)")
    _savefig(out, title)


# ─────────────────────────────────────────────────────────────────────────────
def explain_cic() -> None:
    print("\n== CIC-IDS-2017 explainability ==")
    pipe = joblib.load("data/models/cic2017_kfold_v2_pipeline.joblib")
    stage1, stage2 = pipe["stage1_model"], pipe["stage2_model"]
    scaler, features = pipe["scaler"], list(pipe["features"])
    fam_encoder = pipe["fam_encoder"]
    out_dir = OUT_ROOT / "cic"

    df = pd.read_parquet("data/models/raw/cic2017_subset_10pct.parquet")
    # Use the same feature version the model was trained on (v2 = +dst_port).
    fv = "v2" if "dst_port" in features else "v1"
    X = cic_features_from_df(df, version=fv).values.astype(float)
    Xs = scaler.transform(X)
    labels = df["Label"].values
    labels_fine = df["LabelFine"].values

    rng = np.random.default_rng(42)
    idx = rng.choice(len(Xs), min(SAMPLE_CAP, len(Xs)), replace=False)
    print("  SHAP Stage 1 ...")
    _shap_stage1(stage1, Xs[idx], features, out_dir / "shap_cic_stage1.png")

    print("  SHAP Stage 2 per family ...")
    att = labels != "BENIGN"
    Xatt = pd.DataFrame(Xs[att], columns=features)
    a_idx = rng.choice(len(Xatt), min(SAMPLE_CAP, len(Xatt)), replace=False)
    _shap_stage2_per_family(stage2, Xatt.iloc[a_idx], list(fam_encoder.classes_),
                            out_dir, "cic")

    # CIC-only: per Web-Attack subtype
    print("  SHAP Web-Attack subtypes ...")
    web_mask = labels == "Web Attack"
    if web_mask.sum() > 0:
        expl = shap.TreeExplainer(stage2)
        Xweb = pd.DataFrame(Xs[web_mask], columns=features)
        sv = expl.shap_values(Xweb)
        wa_ci = list(fam_encoder.classes_).index("Web Attack")
        for sub in pd.unique(labels_fine[web_mask]):
            sub_rows = labels_fine[web_mask] == sub
            cls_sv = (sv[wa_ci] if isinstance(sv, list) else sv[:, :, wa_ci])[sub_rows]
            if len(cls_sv) == 0:
                continue
            mean_abs = np.abs(cls_sv).mean(axis=0)
            order = np.argsort(mean_abs)[::-1]
            plt.figure(figsize=(9, 6))
            plt.barh([features[i] for i in order][::-1], mean_abs[order][::-1], color="#f093fb")
            plt.xlabel("mean |SHAP value| (toward Web Attack)")
            safe = _slug(sub)
            _savefig(out_dir / f"shap_cic_webattack_{safe}.png",
                     f"Web Attack subtype: {sub} — feature attribution")

    # LIME: stage-2 per family + a benign FP case study from stage-1
    print("  LIME per family + benign FP case study ...")
    explainer2 = LimeTabularExplainer(
        Xs[att], feature_names=features, class_names=list(fam_encoder.classes_),
        discretize_continuous=True, random_state=42)
    predict2 = lambda d: stage2.predict_proba(pd.DataFrame(d, columns=features))
    for cls in fam_encoder.classes_:
        rows = np.where(labels == cls)[0]
        if len(rows) == 0:
            continue
        x = Xs[rows[0]]
        safe = _slug(cls)
        _lime_case(explainer2, predict2, x, features,
                   out_dir / f"lime_cic_{safe}.png",
                   f"LIME — why this flow is classified {cls}", len(fam_encoder.classes_))

    # Benign FP case study: a benign row Stage 1 calls an attack
    benign_idx = np.where(labels == "BENIGN")[0]
    s1pred = stage1.predict(Xs[benign_idx])
    fp_rows = benign_idx[s1pred == 1]
    if len(fp_rows):
        explainer1 = LimeTabularExplainer(
            Xs, feature_names=features, class_names=["BENIGN", "ATTACK"],
            discretize_continuous=True, random_state=42)
        predict1 = lambda d: stage1.predict_proba(d)
        _lime_case(explainer1, predict1, Xs[fp_rows[0]], features,
                   out_dir / "lime_cic_benign_false_positive.png",
                   "LIME — benign flow FALSELY flagged as attack (FP case study)", 2)
        print(f"    benign FP case study row: {fp_rows[0]} ({len(fp_rows)} benign FPs total)")


def explain_unsw() -> None:
    print("\n== UNSW-NB15 explainability ==")
    pipe = joblib.load("data/models/unsw_kfold_pipeline.joblib")
    stage1, stage2 = pipe["stage1_model"], pipe["stage2_model"]
    scaler, features = pipe["scaler"], list(pipe["features"])
    cat_encoder = pipe["cat_encoder"]
    out_dir = OUT_ROOT / "unsw"

    df = pd.read_parquet("data/models/raw/unsw_subset_10pct.parquet").copy()
    for feat in UNSW_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
    X = df[UNSW_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)
    Xs = scaler.transform(X)
    labels = df["Label"].values

    rng = np.random.default_rng(42)
    idx = rng.choice(len(Xs), min(SAMPLE_CAP, len(Xs)), replace=False)
    print("  SHAP Stage 1 ...")
    _shap_stage1(stage1, Xs[idx], features, out_dir / "shap_unsw_stage1.png")

    print("  SHAP Stage 2 per category ...")
    att = labels != "Normal"
    Xatt = pd.DataFrame(Xs[att], columns=features)
    a_idx = rng.choice(len(Xatt), min(SAMPLE_CAP, len(Xatt)), replace=False)
    _shap_stage2_per_family(stage2, Xatt.iloc[a_idx], list(cat_encoder.classes_),
                            out_dir, "unsw")

    print("  LIME per category + benign FP case study ...")
    explainer2 = LimeTabularExplainer(
        Xs[att], feature_names=features, class_names=list(cat_encoder.classes_),
        discretize_continuous=True, random_state=42)
    predict2 = lambda d: stage2.predict_proba(pd.DataFrame(d, columns=features))
    for cls in cat_encoder.classes_:
        rows = np.where(labels == cls)[0]
        if len(rows) == 0:
            continue
        safe = _slug(cls)
        _lime_case(explainer2, predict2, Xs[rows[0]], features,
                   out_dir / f"lime_unsw_{safe}.png",
                   f"LIME — why this flow is classified {cls}", len(cat_encoder.classes_))

    benign_idx = np.where(labels == "Normal")[0]
    s1pred = stage1.predict(Xs[benign_idx])
    fp_rows = benign_idx[s1pred == 1]
    if len(fp_rows):
        explainer1 = LimeTabularExplainer(
            Xs, feature_names=features, class_names=["Normal", "ATTACK"],
            discretize_continuous=True, random_state=42)
        predict1 = lambda d: stage1.predict_proba(d)
        _lime_case(explainer1, predict1, Xs[fp_rows[0]], features,
                   out_dir / "lime_unsw_benign_false_positive.png",
                   "LIME — normal flow FALSELY flagged as attack (FP case study)", 2)
        print(f"    normal FP case study row: {fp_rows[0]} ({len(fp_rows)} normal FPs total)")


def main() -> None:
    print("=" * 70)
    print("  Explainable AI — SHAP + LIME per attack type (CIC + UNSW)")
    print("=" * 70)
    explain_cic()
    explain_unsw()
    print(f"\n  All figures under: {OUT_ROOT.resolve()}")
    print("  Done.")


if __name__ == "__main__":
    main()
