"""Generate evaluation reports for CIC 2017 and UNSW-NB15 models.

Produces figures saved to reports/model_evaluation/:

CIC 2017 (XGBoost + LightGBM):
  cic_validation_metrics.png        — 5-fold CV accuracy + log-loss
  cic_roc_stage1.png                — Stage 1 binary ROC
  cic_roc_stage2.png                — Stage 2 attack-family ROC
  cic_confusion_matrix.png          — End-to-end confusion matrix
  cic_classification_report.png     — Precision / Recall / F1 / Support

UNSW-NB15 — XGBoost+LightGBM (default):
  unsw_confusion_matrix.png / unsw_classification_report.png / ...

UNSW-NB15 — SVM (if model file present):
  unsw_svm_confusion_matrix.png / unsw_svm_classification_report.png / ...

UNSW-NB15 — Random Forest (if model file present):
  unsw_rf_confusion_matrix.png / unsw_rf_classification_report.png / ...

CV correctness guarantee:
  SMOTE and StandardScaler are both fitted on the training fold only and
  applied to the validation fold — no data leakage.

Usage:
    python scripts/generate_model_reports.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

warnings.filterwarnings("ignore")

OUT_DIR = Path("reports/model_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dark theme ────────────────────────────────────────────────────────────────
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
GRID_COLOR = "#30363d"
ACCENT     = "#00ff88"
COLORS = [
    "#00ff88", "#ff6b6b", "#4ecdc4", "#ffe66d",
    "#a8edea", "#f093fb", "#ffeaa7", "#fd79a8", "#74b9ff", "#55efc4",
]


def _style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color="#8b949e", fontsize=9)
    ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.spines[:].set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_acc_loss(fig_title: str, fold_accs: list[float], fold_losses: list[float],
                  out_path: Path, smote_note: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
    subtitle = "SMOTE + StandardScaler fitted inside each fold (no data leakage)"
    if smote_note:
        subtitle = smote_note
    fig.suptitle(f"{fig_title}\n{subtitle}", color="white", fontsize=13, fontweight="bold")
    folds = list(range(1, len(fold_accs) + 1))

    for ax_idx, (vals, color, label, ylabel) in enumerate([
        ([v * 100 for v in fold_accs], ACCENT, "Val Accuracy", "Accuracy (%)"),
        (fold_losses, "#ff6b6b", "Val Log-Loss", "Log-Loss"),
    ]):
        ax = axes[ax_idx]
        _style_ax(ax, f"{'Validation Accuracy' if ax_idx == 0 else 'Validation Log-Loss'} per Fold",
                  "Fold", ylabel)
        mean_v = np.mean(vals)
        std_v  = np.std(vals)
        fmt = ".2f%" if ax_idx == 0 else ".4f"
        ax.plot(folds, vals, color=color, lw=2, marker="o", markersize=8, label=label)
        ax.axhline(mean_v, color="#ffe66d", lw=1.5, linestyle="--",
                   label=f"Mean = {mean_v:{fmt[:-1] if '%' not in fmt else '6.2f'}}"
                         + ("%" if ax_idx == 0 else ""))
        ax.fill_between(folds, [mean_v - std_v] * len(folds), [mean_v + std_v] * len(folds),
                        alpha=0.15, color=color)
        pad = max(std_v * 3, (0.5 if ax_idx == 0 else mean_v * 0.05))
        if ax_idx == 0:
            ax.set_ylim(max(0, mean_v - pad), min(100, mean_v + pad))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        else:
            ax.set_ylim(max(0, mean_v - pad), mean_v + pad)
        ax.set_xticks(folds)
        ax.legend(framealpha=0.2, labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_stage1_roc(fig_title: str, y_binary: np.ndarray, s1_proba: np.ndarray,
                    out_path: Path, model_label: str = "Stage 1") -> None:
    """Stage 1 binary ROC only — Normal/BENIGN vs Attack."""
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=DARK_BG)
    fig.suptitle(fig_title, color="white", fontsize=13, fontweight="bold")
    _style_ax(ax, f"Stage 1 — Binary: BENIGN vs ATTACK\n({model_label})",
              "False Positive Rate", "True Positive Rate")
    fpr, tpr, _ = roc_curve(y_binary, s1_proba)
    auc_val = roc_auc_score(y_binary, s1_proba)
    ax.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"{model_label}  AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], color=GRID_COLOR, lw=1, linestyle="--", label="Random baseline")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(framealpha=0.2, labelcolor="white", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_stage2_roc(fig_title: str, y_cat: np.ndarray, s2_proba: np.ndarray,
                    class_names: list[str], out_path: Path) -> None:
    """Stage 2 attack-family ROC — attack rows only, no Normal/BENIGN column.

    y_cat: integer labels 0..N-1 for attack categories only (Normal rows excluded)
    s2_proba: shape (n_attack_rows, N) — raw Stage 2 predict_proba output
    class_names: list of N attack category names (no Normal/BENIGN)
    """
    n_classes = len(class_names)
    lb = LabelBinarizer()
    lb.fit(list(range(n_classes)))
    y_bin = lb.transform(y_cat)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=DARK_BG)
    fig.suptitle(fig_title, color="white", fontsize=13, fontweight="bold")
    _style_ax(ax,
              "Stage 2 — Attack-Family ROC (One-vs-Rest)\n"
              "Attack rows only — Normal/BENIGN excluded",
              "False Positive Rate", "True Positive Rate")
    ax.plot([0, 1], [0, 1], color=GRID_COLOR, lw=1, linestyle="--", label="Random")

    plotted = 0
    for i, cls in enumerate(class_names):
        if i >= y_bin.shape[1] or y_bin[:, i].sum() == 0:
            continue
        try:
            fpr_c, tpr_c, _ = roc_curve(y_bin[:, i], s2_proba[:, i])
            auc_c = roc_auc_score(y_bin[:, i], s2_proba[:, i])
            ax.plot(fpr_c, tpr_c, color=COLORS[plotted % len(COLORS)], lw=1.8,
                    label=f"{cls:<16} AUC {auc_c:.3f}")
            plotted += 1
        except Exception:
            pass

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(framealpha=0.2, labelcolor="white", fontsize=9, loc="lower right",
              prop={"family": "monospace"})
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_confusion_matrix(fig_title: str, y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list[str], out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    # Row-normalise for readability (show recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    n = len(class_names)
    fig_h = max(7, n * 0.7 + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_h * 2.2, fig_h), facecolor=DARK_BG)
    fig.suptitle(fig_title, color="white", fontsize=13, fontweight="bold")

    for ax, data, fmt, title in [
        (axes[0], cm,      "d",    "Confusion Matrix (counts)"),
        (axes[1], cm_norm, ".2f",  "Confusion Matrix (row-normalised recall)"),
    ]:
        ax.set_facecolor(PANEL_BG)
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", color="#8b949e", fontsize=8)
        ax.set_yticklabels(class_names, color="#8b949e", fontsize=8)
        ax.set_xlabel("Predicted", color="#8b949e"); ax.set_ylabel("True", color="#8b949e")
        ax.set_title(title, color="white", fontsize=10, pad=8)
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                text = f"{val:{fmt}}"
                color = "white" if (fmt == ".2f" and val < 0.5) or (fmt == "d" and val < cm.max() * 0.5) else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_classification_report(fig_title: str, y_true: np.ndarray, y_pred: np.ndarray,
                                class_names: list[str], out_path: Path) -> None:
    report = classification_report(y_true, y_pred, labels=class_names,
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)

    metrics = ["precision", "recall", "f1-score", "support"]
    rows = class_names + ["accuracy", "macro avg", "weighted avg"]
    data = []
    for r in rows:
        if r == "accuracy":
            acc = report.get("accuracy", 0.0)
            total = sum(report[c]["support"] for c in class_names if c in report)
            data.append([acc, acc, acc, total])
        elif r in report:
            data.append([report[r][m] for m in metrics])
        else:
            data.append([0, 0, 0, 0])

    fig_h = max(6, len(rows) * 0.42 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h), facecolor=DARK_BG)
    fig.suptitle(fig_title, color="white", fontsize=13, fontweight="bold")
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    col_labels = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    table_data = []
    for r, vals in zip(rows, data):
        if r in ("accuracy", "macro avg", "weighted avg"):
            table_data.append([r, f"{vals[0]:.4f}", f"{vals[1]:.4f}", f"{vals[2]:.4f}",
                                f"{int(vals[3]):,}"])
        else:
            table_data.append([r, f"{vals[0]:.4f}", f"{vals[1]:.4f}", f"{vals[2]:.4f}",
                                f"{int(vals[3]):,}"])

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Style header
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#1f6feb")
        cell.set_text_props(color="white", fontweight="bold")

    # Style rows — colour by F1 value
    for i, (r, vals) in enumerate(zip(rows, data), 1):
        f1 = vals[2]
        if r in ("accuracy", "macro avg", "weighted avg"):
            bg = "#1c2128"
        elif f1 >= 0.90:
            bg = "#0d3321"
        elif f1 >= 0.70:
            bg = "#1a2a1a"
        elif f1 >= 0.50:
            bg = "#2a2010"
        else:
            bg = "#2a1010"
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            cell.set_facecolor(bg)
            cell.set_text_props(color="white")
            cell.set_edgecolor(GRID_COLOR)

    # Header edge
    for j in range(len(col_labels)):
        tbl[0, j].set_edgecolor(GRID_COLOR)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# CIC 2017
# ═════════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.flow_features import (  # noqa: E402
    cic_features_from_df, unsw_features_from_df,
    CIC_FLOW_FEATURES, UNSW_FLOW_FEATURES,
)

CIC_CSV_DIR = Path("data/models/raw/cic 2017/MachineLearningCSV/MachineLearningCVE")
CIC_MODEL   = Path("data/models/cic2017_pipeline_smote.joblib")
CIC_LABEL_MAP = {
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS", "DoS Slowhttptest": "DoS",
    "FTP-Patator": "Brute Force", "SSH-Patator": "Brute Force",
    "Web Attack  Brute Force": "Web Attack",
    "Web Attack  XSS": "Web Attack",
    "Web Attack  Sql Injection": "Web Attack",
}
CIC_ATTACK_FAMILIES = ["Bot", "Brute Force", "DDoS", "DoS", "PortScan", "Web Attack"]
CIC_ALL_CLASSES     = ["BENIGN"] + CIC_ATTACK_FAMILIES
CIC_EXCLUDE         = {"Heartbleed", "Infiltration"}


def _load_cic(benign_cap: int = 50_000, attack_cap: int = 8_000,
              features: list[str] | None = None,
              version: str = "v1") -> tuple[np.ndarray, np.ndarray]:
    """Load CIC CSVs using the shared flow_features module (same as training).

    version selects which feature set to compute so it MATCHES the model being
    evaluated: "v1" -> 17 features, "v2" -> 18 features (adds dst_port). Feeding a
    v1 (17) vector to a v2 (18) model silently mis-scores every class (this was the
    bug that crushed the reported Web-Attack recall).
    """
    # Auto-detect version from the feature list if provided.
    if features is not None and "dst_port" in features:
        version = "v2"
    print(f"  Loading CIC 2017 CSVs (shared flow_features module, {version})...")
    frames = []
    csv_files = sorted(CIC_CSV_DIR.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CIC CSVs found under {CIC_CSV_DIR}")

    for f in csv_files:
        df = pd.read_csv(f, low_memory=False, encoding="utf-8", encoding_errors="replace")
        df.columns = df.columns.str.strip()
        df["Label"] = df["Label"].astype(str).str.strip()
        for k, v in CIC_LABEL_MAP.items():
            df.loc[df["Label"] == k, "Label"] = v
        df.loc[df["Label"].str.contains("Web Attack", na=False), "Label"] = "Web Attack"
        df = df[~df["Label"].isin(CIC_EXCLUDE)]
        df = df[df["Label"].isin(CIC_ALL_CLASSES)].copy()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(42)

    parts = []
    for cls in CIC_ALL_CLASSES:
        rows = combined[combined["Label"] == cls]
        cap = benign_cap if cls == "BENIGN" else attack_cap
        if len(rows) > cap:
            rows = rows.iloc[rng.choice(len(rows), cap, replace=False)]
        parts.append(rows)
    combined = pd.concat(parts, ignore_index=True).reset_index(drop=True)

    # Compute the feature set that matches the model (v1=17, v2=18).
    X = cic_features_from_df(combined, version=version).values.astype(float)
    return X, combined["Label"].values


def evaluate_cic(model_path: Path = CIC_MODEL, prefix: str = "cic",
                 title: str = "CIC 2017") -> None:
    print(f"\n-- {title} Model Evaluation ------------------------------")
    if not model_path.exists():
        print(f"  [SKIP] model not found: {model_path}")
        return
    pipeline = joblib.load(model_path)
    stage1   = pipeline["stage1_model"]
    stage2   = pipeline["stage2_model"]
    encoder  = pipeline["fam_encoder"]
    cic_features: list[str] = list(pipeline["features"])  # CIC_FLOW_FEATURES from shared module
    print(f"  Model: {model_path.name}")
    print(f"  Feature set: {pipeline.get('meta', {}).get('feature_set', 'unknown')}")
    print(f"  Features ({len(cic_features)}): {cic_features}")

    X_raw, labels = _load_cic(features=cic_features)
    y_binary = np.where(labels == "BENIGN", 0, 1).astype(int)

    # Map attack labels -> family integer index
    known = set(encoder.classes_)
    safe_attack: list[str] = []
    for lbl in labels:
        if lbl in known:
            safe_attack.append(lbl)
        else:
            safe_attack.append(next((a for a in CIC_ATTACK_FAMILIES if a.lower() in str(lbl).lower()), "DDoS"))
    y_family = np.where(y_binary == 0, -1,
                        np.array([encoder.transform([s])[0] for s in safe_attack]))
    # Combined: 0=BENIGN, 1..N = family_idx+1
    y_combined = np.where(y_binary == 0, 0, y_family + 1).astype(int)
    print(f"  Loaded {len(labels):,} samples")

    # ── 80/20 stratified split (raw features — no leakage) ────────
    X_tr, X_te, yb_tr, yb_te, yc_tr, yc_te, lbl_tr, lbl_te = train_test_split(
        X_raw, y_binary, y_combined, labels,
        test_size=0.2, stratify=y_binary, random_state=42,
    )

    # ── 5-fold CV: SMOTE + scaling INSIDE each fold ────────────────
    print("  Running 5-fold CV (SMOTE + StandardScaler inside each fold)...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Cap for speed — use stratified subsample of training split
    max_cv = min(len(yb_tr), 60_000)
    rng = np.random.default_rng(42)
    cv_idx = rng.choice(len(yb_tr), max_cv, replace=False)
    X_cv_raw, y_cv = X_tr[cv_idx], yb_tr[cv_idx]

    fold_accs, fold_losses = [], []
    for fold, (tr, va) in enumerate(kf.split(X_cv_raw, y_cv), 1):
        # 1. Fit scaler on training fold only
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_cv_raw[tr])
        X_va_s = sc.transform(X_cv_raw[va])

        # 2. SMOTE on training fold only (binary: benign vs attack)
        counts = dict(zip(*np.unique(y_cv[tr], return_counts=True)))
        minority = counts.get(1, 0)
        if minority >= 10:
            k = min(5, minority - 1)
            try:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_tr_s, y_tr_sm = sm.fit_resample(X_tr_s, y_cv[tr])
            except Exception:
                y_tr_sm = y_cv[tr]
        else:
            y_tr_sm = y_cv[tr]

        # 3. Train Stage 1 on this fold
        s1 = _clone_model_for_cv(stage1)
        s1.fit(X_tr_s, y_tr_sm)
        fold_accs.append(accuracy_score(y_cv[va], s1.predict(X_va_s)))
        fold_losses.append(log_loss(y_cv[va], s1.predict_proba(X_va_s)))
        print(f"    Fold {fold}: acc={fold_accs[-1]:.4f}  loss={fold_losses[-1]:.4f}"
              f"  (SMOTE minority: {minority}->{int((y_tr_sm==1).sum())})")

    # ── Scale using the model's own scaler (same transform as training) ──
    model_scaler = pipeline["scaler"]
    X_tr_scaled = model_scaler.transform(X_tr)
    X_te_scaled = model_scaler.transform(X_te)

    # ── Stage 1 binary ROC ─────────────────────────────────────────
    print("  Computing ROC curves...")
    s1_proba_te = stage1.predict_proba(X_te_scaled)[:, 1]
    s1_pred_te  = stage1.predict(X_te_scaled)

    plot_stage1_roc(
        f"{title} — Stage 1 Binary ROC (held-out 20%)",
        yb_te, s1_proba_te,
        OUT_DIR / f"{prefix}_roc_stage1.png",
    )

    # ── Stage 2 ROC: attack rows only, no BENIGN column ───────────
    true_att_mask = yb_te == 1   # ground-truth attack rows
    pred_att_mask = s1_pred_te == 1
    # Use rows that are truly attacks for Stage 2 evaluation
    X_true_att = X_te_scaled[true_att_mask]
    y_fam_te   = yc_te[true_att_mask] - 1   # 0-indexed family labels

    if true_att_mask.sum() > 0:
        X_att_df = pd.DataFrame(X_true_att, columns=cic_features)
        s2_proba_te = stage2.predict_proba(X_att_df)
        attack_class_names = list(encoder.classes_)
        plot_stage2_roc(
            f"{title} — Stage 2 Attack-Family ROC (attack rows only, held-out 20%)",
            y_fam_te, s2_proba_te, attack_class_names,
            OUT_DIR / f"{prefix}_roc_stage2.png",
        )

    # ── End-to-end predictions for confusion matrix ────────────────
    print("  Computing confusion matrix and classification report...")
    e2e_pred: list[str] = ["BENIGN" if p == 0 else "Unknown" for p in s1_pred_te]
    if pred_att_mask.any():
        X_att_pred = pd.DataFrame(X_te_scaled[pred_att_mask], columns=cic_features)
        fam_preds  = encoder.inverse_transform(stage2.predict(X_att_pred))
        for i, name in zip(np.where(pred_att_mask)[0], fam_preds):
            e2e_pred[i] = str(name)

    plot_confusion_matrix(
        f"{title} — End-to-End Confusion Matrix (held-out 20%)",
        lbl_te, np.array(e2e_pred), CIC_ALL_CLASSES,
        OUT_DIR / f"{prefix}_confusion_matrix.png",
    )
    plot_classification_report(
        f"{title} — Classification Report: Precision / Recall / F1 / Support (held-out 20%)",
        lbl_te, np.array(e2e_pred), CIC_ALL_CLASSES,
        OUT_DIR / f"{prefix}_classification_report.png",
    )
    plot_acc_loss(
        f"{title} — 5-Fold Cross-Validation Metrics",
        fold_accs, fold_losses,
        OUT_DIR / f"{prefix}_validation_metrics.png",
        smote_note="SMOTE + StandardScaler fitted inside each fold — no data leakage",
    )

    # Print report to console too
    print(f"\n  Classification Report ({title}):")
    print(classification_report(lbl_te, np.array(e2e_pred),
                                labels=CIC_ALL_CLASSES, zero_division=0))
    # Surface Web-Attack recall specifically (the headline number)
    rep = classification_report(lbl_te, np.array(e2e_pred), labels=CIC_ALL_CLASSES,
                                output_dict=True, zero_division=0)
    wa = rep.get("Web Attack", {})
    print(f"  >>> {title} Web-Attack recall = {wa.get('recall', 0)*100:.2f}%  "
          f"precision = {wa.get('precision', 0)*100:.2f}%")


# ═════════════════════════════════════════════════════════════════════════════
# UNSW-NB15  (shared across XGBoost, SVM, and Random Forest models)
# ═════════════════════════════════════════════════════════════════════════════

UNSW_TRAIN = Path("data/models/raw/unsbw15/Training and Testing Sets/UNSW_NB15_training-set.csv")
UNSW_TEST  = Path("data/models/raw/unsbw15/Training and Testing Sets/UNSW_NB15_testing-set.csv")

# All known UNSW models — (model_file, file_prefix, display_name, model_label)
UNSW_MODELS = [
    (
        Path("data/models/unsw_nb15_pipeline.joblib"),
        "unsw",
        "UNSW-NB15 (11-feat flow, Suricata-servable)",
        "XGBoost-11",
    ),
    (
        # The SHAP-discovered, real-traffic-recalibrated 28-feature model (Zeek).
        Path("data/models/unsw_kfold_pipeline.joblib"),
        "unsw28",
        "UNSW-NB15 (28-feat SHAP+recalibrated, Zeek)",
        "XGBoost-28",
    ),
    (
        Path("data/models/unsw_nb15_svm_pipeline.joblib"),
        "unsw_svm",
        "UNSW-NB15 (SVM)",
        "SVM",
    ),
    (
        Path("data/models/unsw_nb15_rf_pipeline.joblib"),
        "unsw_rf",
        "UNSW-NB15 (Random Forest)",
        "Random Forest",
    ),
]

_UNSW_COL_ALIASES = {
    "smean": "smeansz", "dmean": "dmeansz",
    "sinpkt": "sintpkt", "dinpkt": "dintpkt",
    "ct_src_ ltm": "ct_src_ltm",
}

# Canonical class order for display (superset — models may use 6 or 9)
UNSW_CLASS_ORDER = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
]

# Merges applied in the 6-class models (SVM / RF / new XGBoost)
_UNSW_MERGE = {
    "Fuzzers":  "Generic",
    "Analysis": "Reconnaissance",
    "Backdoor": "Exploits",
}


_RAW_UNSW_CATS = {"Normal","Generic","Exploits","Fuzzers","DoS",
                   "Reconnaissance","Analysis","Backdoor","Shellcode","Worms"}

def _load_unsw(path: Path, features: list[str],
               apply_merge: bool = False,
               freq_maps: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, low_memory=False, encoding="utf-8", encoding_errors="replace")
    df.columns = df.columns.str.strip()
    df = df.rename(columns=_UNSW_COL_ALIASES)
    for col in ["attack_cat", "Attack_cat"]:
        if col in df.columns:
            df = df.rename(columns={col: "Label"})
            break
    df["Label"] = df["Label"].astype(str).str.strip()
    df["Label"] = df["Label"].apply(lambda x: x if x in _RAW_UNSW_CATS else "Generic")
    if apply_merge:
        df["Label"] = df["Label"].replace(_UNSW_MERGE)
    # Use shared feature module for UNSW_FLOW_FEATURES models, raw cols otherwise
    if set(features) <= set(UNSW_FLOW_FEATURES):
        X = unsw_features_from_df(df).values.astype(float)
    else:
        # 28-feature model: proto/service are strings -> frequency-encode them with
        # the SAME maps the model was trained with (stored in the model dict). This
        # is the serving-reproducible encoding; without it those columns are wrong.
        if freq_maps:
            for c in ("proto", "service"):
                if c in df.columns and c in features:
                    df[c] = df[c].astype(str).map(freq_maps.get(c, {})).fillna(0.0)
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0.0
        X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)
    y = np.where(df["Label"].values == "Normal", 0, 1).astype(int)
    return X, y, df["Label"].values


def _has_proba(model: object) -> bool:
    """Return True if the model supports predict_proba."""
    return hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba"))


def _predict_proba_binary(model: object, X: np.ndarray) -> np.ndarray | None:
    """Return positive-class probabilities, or None if not supported."""
    if _has_proba(model):
        try:
            return model.predict_proba(X)[:, 1]  # type: ignore[union-attr]
        except Exception:
            pass
    return None


def _clone_model_for_cv(model: object) -> object:
    """Clone an estimator while stripping fit-time early-stopping state.

    Some saved XGBoost estimators keep `early_stopping_rounds` or callback
    configuration in their params. Those settings require an evaluation set at
    fit time, which we do not provide inside fold-local CV training.
    """
    params = model.get_params(deep=True)
    for key in ("early_stopping_rounds", "callbacks", "eval_set", "eval_metric"):
        params.pop(key, None)
    return type(model)(**params)


def evaluate_unsw_model(model_path: Path, prefix: str, display_name: str,
                        model_label: str) -> None:
    print(f"\n-- {display_name} Evaluation --")
    try:
        pipeline = joblib.load(model_path)
    except ModuleNotFoundError as exc:
        if exc.name == "cuml" or "cuml" in str(exc):
            print(f"  [SKIP] {display_name} model was saved with cuML objects, but cuML is not installed here.")
            print(f"         To evaluate this model, open the same RAPIDS/Docker environment used for training.")
            return
        raise
    except Exception as exc:
        if "cuml" in str(exc).lower():
            print(f"  [SKIP] {display_name} model requires cuML to unpickle: {exc}")
            print(f"         Run this report generator inside the RAPIDS environment used for training.")
            return
        raise
    stage1   = pipeline["stage1_model"]
    stage2   = pipeline["stage2_model"]
    encoder  = pipeline.get("cat_encoder") or pipeline.get("fam_encoder")
    features = pipeline["features"]
    attack_class_names: list[str] = list(encoder.classes_)

    # Determine whether this model uses the 6-class merge
    uses_merge = set(attack_class_names) <= {"Generic", "Exploits", "DoS",
                                              "Reconnaissance", "Shellcode", "Worms"}

    fmaps = pipeline.get("freq_maps")   # 28-feature model carries proto/service maps
    X_train_raw, y_train, _ = _load_unsw(UNSW_TRAIN, features, apply_merge=uses_merge,
                                         freq_maps=fmaps)
    X_test_raw, y_test, test_labels = _load_unsw(UNSW_TEST, features, apply_merge=uses_merge,
                                                 freq_maps=fmaps)
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")
    print(f"  Attack classes: {attack_class_names}")
    print(f"  Model type: {pipeline.get('meta', {}).get('model_type', model_label)}")

    # ── 5-fold CV: SMOTE + scaling INSIDE each fold ────────────────
    print("  Running 5-fold CV (SMOTE + StandardScaler inside each fold)...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    max_cv = min(len(y_train), 60_000)
    rng = np.random.default_rng(42)
    cv_idx = rng.choice(len(y_train), max_cv, replace=False)
    X_cv_raw, y_cv = X_train_raw[cv_idx], y_train[cv_idx]

    fold_accs: list[float] = []
    fold_losses: list[float] = []
    supports_proba = _has_proba(stage1)

    for fold, (tr, va) in enumerate(kf.split(X_cv_raw, y_cv), 1):
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_cv_raw[tr])
        X_va_s = sc.transform(X_cv_raw[va])

        counts  = dict(zip(*np.unique(y_cv[tr], return_counts=True)))
        minority = counts.get(1, 0)
        if minority >= 10:
            k = min(5, minority - 1)
            try:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_tr_s, y_tr_sm = sm.fit_resample(X_tr_s, y_cv[tr])
            except Exception:
                y_tr_sm = y_cv[tr]
        else:
            y_tr_sm = y_cv[tr]

        try:
            s1_fold = _clone_model_for_cv(stage1)
        except Exception:
            s1_fold = type(stage1)()
        s1_fold.fit(X_tr_s, y_tr_sm)
        preds = s1_fold.predict(X_va_s)
        fold_accs.append(accuracy_score(y_cv[va], preds))

        if _has_proba(s1_fold):
            try:
                fold_losses.append(log_loss(y_cv[va], s1_fold.predict_proba(X_va_s)))
            except Exception:
                fold_losses.append(float("nan"))
        else:
            fold_losses.append(float("nan"))

        loss_str = f"{fold_losses[-1]:.4f}" if not np.isnan(fold_losses[-1]) else "n/a"
        print(f"    Fold {fold}: acc={fold_accs[-1]:.4f}  loss={loss_str}"
              f"  (SMOTE minority: {minority}->{int((y_tr_sm == 1).sum())})")

    # Replace nan losses with mean of valid ones for plotting
    valid_losses = [v for v in fold_losses if not np.isnan(v)]
    if not valid_losses:
        fold_losses = [0.0] * len(fold_accs)
    else:
        mean_loss = float(np.mean(valid_losses))
        fold_losses = [v if not np.isnan(v) else mean_loss for v in fold_losses]

    # ── Scale using the model's own scaler ────────────────────────
    X_test = pipeline["scaler"].transform(X_test_raw)

    # ── Stage 1 binary ROC (only when predict_proba is available) ─
    s1_pred_te = stage1.predict(X_test)
    s1_proba_te = _predict_proba_binary(stage1, X_test)

    if s1_proba_te is not None:
        print("  Computing Stage 1 ROC curve...")
        plot_stage1_roc(
            f"{display_name} — Stage 1 Binary ROC (held-out test set)",
            y_test, s1_proba_te,
            OUT_DIR / f"{prefix}_roc_stage1.png",
            model_label=model_label,
        )
    else:
        print(f"  Skipping Stage 1 ROC — {model_label} does not support predict_proba")

    # ── Stage 2 ROC (only when predict_proba is available) ────────
    true_att_mask = y_test == 1
    if true_att_mask.sum() > 0 and _has_proba(stage2):
        X_true_att  = X_test[true_att_mask]
        true_att_labels = test_labels[true_att_mask]
        cat_to_idx  = {c: i for i, c in enumerate(attack_class_names)}
        y_cat_te    = np.array([cat_to_idx.get(lbl, 0) for lbl in true_att_labels])
        try:
            s2_input    = pd.DataFrame(X_true_att, columns=features)
            s2_proba_te = stage2.predict_proba(s2_input)
            print("  Computing Stage 2 ROC curve...")
            plot_stage2_roc(
                f"{display_name} — Stage 2 Attack-Category ROC (attack rows only)",
                y_cat_te, s2_proba_te, attack_class_names,
                OUT_DIR / f"{prefix}_roc_stage2.png",
            )
        except Exception as exc:
            print(f"  Skipping Stage 2 ROC — {exc}")
    else:
        print(f"  Skipping Stage 2 ROC — {model_label} does not support predict_proba")

    # ── End-to-end predictions ─────────────────────────────────────
    print("  Computing confusion matrix and classification report...")
    e2e_pred: list[str] = ["Normal" if p == 0 else "Unknown" for p in s1_pred_te]
    pred_att_mask = s1_pred_te == 1
    if pred_att_mask.any():
        X_att_pred = X_test[pred_att_mask]
        try:
            s2_input = pd.DataFrame(X_att_pred, columns=features)
            cat_preds = encoder.inverse_transform(stage2.predict(s2_input))
        except Exception:
            cat_preds = encoder.inverse_transform(stage2.predict(X_att_pred))
        for i, name in zip(np.where(pred_att_mask)[0], cat_preds):
            e2e_pred[i] = str(name)

    present = sorted(
        set(test_labels.tolist()) | set(e2e_pred),
        key=lambda x: UNSW_CLASS_ORDER.index(x) if x in UNSW_CLASS_ORDER else 99,
    )

    plot_confusion_matrix(
        f"{display_name} — End-to-End Confusion Matrix (held-out test set)",
        test_labels, np.array(e2e_pred), present,
        OUT_DIR / f"{prefix}_confusion_matrix.png",
    )
    plot_classification_report(
        f"{display_name} — Classification Report: Precision / Recall / F1 / Support (held-out test set)",
        test_labels, np.array(e2e_pred), present,
        OUT_DIR / f"{prefix}_classification_report.png",
    )
    plot_acc_loss(
        f"{display_name} — 5-Fold Cross-Validation Metrics",
        fold_accs, fold_losses,
        OUT_DIR / f"{prefix}_validation_metrics.png",
        smote_note="SMOTE + StandardScaler fitted inside each fold — no data leakage",
    )

    print(f"\n  Classification Report ({display_name}):")
    print(classification_report(test_labels, np.array(e2e_pred),
                                labels=present, zero_division=0))
    # Surface the weak-class recalls the 28-feature model targets.
    rep = classification_report(test_labels, np.array(e2e_pred), labels=present,
                                output_dict=True, zero_division=0)
    weak = [c for c in ("DoS", "Reconnaissance", "Shellcode", "Worms") if c in rep]
    if weak:
        print("  >>> weak-class recall: " + "  ".join(
            f"{c}={rep[c]['recall']*100:.1f}%" for c in weak))


def evaluate_webattack_v3() -> None:
    """The Web-Attack SPECIALIST (v3): a binary BENIGN-vs-Web-Attack booster trained
    on PCAP-extracted HTTP features. It is NOT the main CIC model (it can't classify
    Bot/DDoS/DoS/PortScan) — it runs alongside the v2 7-class model to rescue the one
    class v2 is weak on. This panel shows the recall it actually achieves."""
    print("\n-- CIC Web-Attack v3 SPECIALIST (binary booster) --------------")
    model_path = Path("data/models/cic2017_webattack_v3_http.joblib")
    rich = Path("data/models/raw/cic2017_pcap_rich.parquet")
    if not model_path.exists() or not rich.exists():
        print(f"  [SKIP] need {model_path.name} + {rich.name} "
              "(run train_cic_v3_webattack.py / extract_pcap_features.py)")
        return
    try:
        from scripts.train_cic_v3_webattack import _derive
    except Exception as exc:
        print(f"  [SKIP] cannot import v3 feature builder: {exc}")
        return

    pipe = joblib.load(model_path)
    feats = pipe["features"]
    df = _derive(pd.read_parquet(rich))
    X = df[feats].values.astype(float)
    y_true = np.where(df["Label"].values == "Web Attack", "Web Attack", "BENIGN")
    Xs = pipe["scaler"].transform(X)
    pred = pipe["model"].predict(Xs)
    y_pred = np.where(pred == 1, "Web Attack", "BENIGN")

    classes = ["BENIGN", "Web Attack"]
    plot_confusion_matrix(
        "CIC Web-Attack v3 SPECIALIST — Confusion Matrix (PCAP, BENIGN vs Web Attack)",
        y_true, y_pred, classes, OUT_DIR / "cic_v3_webattack_confusion_matrix.png")
    plot_classification_report(
        f"CIC Web-Attack v3 SPECIALIST — {pipe.get('tier','')} "
        "(binary booster, runs alongside the v2 7-class model)",
        y_true, y_pred, classes, OUT_DIR / "cic_v3_webattack_classification_report.png")

    rep = classification_report(y_true, y_pred, labels=classes,
                                output_dict=True, zero_division=0)
    wa = rep["Web Attack"]
    print(f"  >>> v3 Web-Attack recall = {wa['recall']*100:.2f}%  "
          f"precision = {wa['precision']*100:.2f}%  "
          f"(vs ~17% for the v2 7-class flow-only model — this is the booster's job)")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Lighthouse -- Model Evaluation Report Generator")
    print("=" * 60)

    # CIC: compare v1 (17-feat) vs v2 (18-feat, dst_port) — same data, correct
    # features for each, so the dst_port impact on Web-Attack is visible and honest.
    evaluate_cic(Path("data/models/cic2017_kfold_v1_pipeline.joblib"), "cic_v1",
                 "CIC 2017 v1 (17-feat)")
    evaluate_cic(CIC_MODEL, "cic", "CIC 2017 v2 (18-feat, production)")
    evaluate_webattack_v3()

    for model_path, prefix, display_name, model_label in UNSW_MODELS:
        if not model_path.exists():
            print(f"\n  [SKIP] {display_name} — model not found: {model_path}")
            continue
        evaluate_unsw_model(model_path, prefix, display_name, model_label)

    print(f"\nAll graphs saved to: {OUT_DIR.resolve()}")
    print("=" * 60)
