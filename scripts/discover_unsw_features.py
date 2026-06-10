"""Explainable-AI feature discovery for UNSW-NB15 — find what lifts the weak classes.

DoS, Reconnaissance, Shellcode and Worms underperform in the current UNSW model,
which only uses 18 of the 42 available columns. Rather than guess which of the
unused 24 columns help, this script lets SHAP + LIME *discover* them:

  1. Build the full feature matrix from ALL 42 UNSW columns (proto/service/state
     label-encoded).
  2. Train a LightGBM multiclass model (6 merged attack categories + Normal) on
     the full set.
  3. SHAP (TreeExplainer) ranks every column PER attack category — with a focus on
     the four weak classes (DoS, Reconnaissance, Shellcode, Worms).
  4. LIME explains a representative row of each weak class.
  5. Emit unsw_feature_ranking.json: global + per-class column rankings, and a
     recommended feature set = current 18 ∪ the new high-value columns.

Outputs:
    reports/feature_discovery/unsw/*.png
    reports/feature_discovery/unsw_feature_ranking.json

Usage:
    python scripts/discover_unsw_features.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import LabelEncoder

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_unsw_svm import (  # noqa: E402
    ATTACK_CATEGORIES, CATEGORY_MERGE, COL_ALIASES, FEATURES as CURRENT_FEATURES,
    TRAIN_CSV, _RAW_CATEGORIES,
)

warnings.filterwarnings("ignore")

DARK_BG, PANEL_BG, GRID, ACCENT = "#0d1117", "#161b22", "#30363d", "#00ff88"
plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG, "savefig.facecolor": DARK_BG,
    "text.color": "white", "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e", "axes.edgecolor": GRID,
})

OUT_DIR = Path("reports/feature_discovery/unsw")
RANK_JSON = Path("reports/feature_discovery/unsw_feature_ranking.json")
WEAK = ["DoS", "Reconnaissance", "Shellcode", "Worms"]
ALL_LABELS = ["Normal"] + ATTACK_CATEGORIES
DROP = {"id", "attack_cat", "Attack_cat", "label", "Label", "LabelFine"}
CATEGORICAL = ["proto", "service", "state"]


def _savefig(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.suptitle(title, color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    saved -> {path}")


def _load() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(TRAIN_CSV, low_memory=False, encoding="utf-8", encoding_errors="replace")
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COL_ALIASES)
    for col in ["attack_cat", "Attack_cat"]:
        if col in df.columns:
            df = df.rename(columns={col: "Label"})
            break
    df["Label"] = df["Label"].astype(str).str.strip()
    df["Label"] = df["Label"].apply(lambda x: x if x in _RAW_CATEGORIES else "Generic")
    df["Label"] = df["Label"].replace(CATEGORY_MERGE)
    df = df[df["Label"].isin(ALL_LABELS)].reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in DROP]
    X = df[feat_cols].copy()
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    keep = [c for c in feat_cols if X[c].std() > 0]
    return X[keep], df["Label"].values, keep


def main() -> None:
    print("=" * 70)
    print("  UNSW-NB15 feature discovery (SHAP + LIME over ALL 42 columns)")
    print("=" * 70)
    X, labels, feat_cols = _load()
    print(f"  Rows: {len(X):,}   columns considered: {len(feat_cols)}")
    print(f"  Currently used ({len(CURRENT_FEATURES)}): {CURRENT_FEATURES}")

    enc = LabelEncoder().fit(ALL_LABELS)
    enc.classes_ = np.array(ALL_LABELS)
    y = enc.transform(labels)

    clf = lgb.LGBMClassifier(
        n_estimators=400, max_depth=10, num_leaves=127, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
        objective="multiclass", num_class=len(ALL_LABELS),
        random_state=42, n_jobs=-1, verbose=-1, class_weight="balanced",
    )
    clf.fit(X.values, y, feature_name=feat_cols)

    print("  SHAP per-class ranking ...")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(6000, len(X)), replace=False)
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(X.iloc[idx])   # list per class OR 3D array

    def _class_sv(ci: int) -> np.ndarray:
        if isinstance(sv, list):
            return sv[ci]
        return sv[:, :, ci] if sv.ndim == 3 else sv

    per_class_ranking: dict[str, list] = {}
    for ci, cls in enumerate(ALL_LABELS):
        mean_abs = np.abs(_class_sv(ci)).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        per_class_ranking[cls] = [
            {"feature": feat_cols[i], "mean_abs_shap": float(mean_abs[i]),
             "currently_used": feat_cols[i] in CURRENT_FEATURES} for i in order[:20]
        ]
        if cls in WEAK or cls == "Normal":
            top = order[:15]
            color = "#ff6b6b" if cls in WEAK else ACCENT
            plt.figure(figsize=(9, 6))
            plt.barh([feat_cols[i] for i in top][::-1], mean_abs[top][::-1], color=color)
            plt.xlabel(f"mean |SHAP| toward {cls}")
            _savefig(OUT_DIR / f"shap_unsw_{cls}.png",
                     f"UNSW {cls} — top columns (SHAP, * = currently unused)")

    # Global ranking (mean over classes)
    if isinstance(sv, list):
        glob = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        glob = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
    gorder = np.argsort(glob)[::-1]

    # LIME for each weak class
    print("  LIME for weak classes ...")
    lime_expl = LimeTabularExplainer(
        X.values, feature_names=feat_cols, class_names=ALL_LABELS,
        discretize_continuous=True, random_state=42)
    for cls in WEAK:
        rows = np.where(labels == cls)[0]
        if len(rows) == 0:
            continue
        ci = ALL_LABELS.index(cls)
        exp = lime_expl.explain_instance(X.values[rows[0]], clf.predict_proba,
                                         num_features=12, labels=[ci])
        pairs = exp.as_list(label=ci)[::-1]
        names, weights = [p[0] for p in pairs], [p[1] for p in pairs]
        colors = [ACCENT if w >= 0 else "#ff6b6b" for w in weights]
        plt.figure(figsize=(10, 6))
        plt.barh(names, weights, color=colors)
        plt.axvline(0, color=GRID, lw=1)
        plt.xlabel(f"LIME local weight (green supports {cls})")
        _savefig(OUT_DIR / f"lime_unsw_{cls}.png", f"LIME — why this flow is classified {cls}")

    # Recommended feature set: current 18 ∪ top new columns that help the weak classes
    new_for_weak: list[str] = []
    for cls in WEAK:
        for item in per_class_ranking[cls][:8]:
            f = item["feature"]
            if f not in CURRENT_FEATURES and f not in new_for_weak:
                new_for_weak.append(f)
    recommended = list(CURRENT_FEATURES) + new_for_weak

    payload = {
        "n_features_considered": len(feat_cols),
        "currently_used": list(CURRENT_FEATURES),
        "global_ranking": [
            {"rank": r + 1, "feature": feat_cols[i], "mean_abs_shap": float(glob[i]),
             "currently_used": feat_cols[i] in CURRENT_FEATURES}
            for r, i in enumerate(gorder)
        ],
        "per_class_ranking": per_class_ranking,
        "new_columns_for_weak_classes": new_for_weak,
        "recommended_feature_set": recommended,
    }
    RANK_JSON.parent.mkdir(parents=True, exist_ok=True)
    RANK_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    joblib.dump({"model": clf, "features": feat_cols, "encoder": enc},
                OUT_DIR / "discovery_model.joblib")

    print(f"\n  New high-value columns for weak classes: {new_for_weak}")
    print(f"  Recommended feature set ({len(recommended)}): {recommended}")
    print(f"  Wrote {RANK_JSON}")
    print("  Done.")


if __name__ == "__main__":
    main()
