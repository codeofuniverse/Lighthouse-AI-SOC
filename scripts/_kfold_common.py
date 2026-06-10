"""Shared helpers for the stratified-k-fold trainers (CIC + UNSW).

Holds the three things both trainers need identically:

  1. resampling_plan()       — the FP fix: decide per-class whether to use
                               class-weighting (ultra-rare) or capped SMOTE.
  2. apply_resampling()      — execute that plan inside ONE fold (leakage-free).
  3. FoldRecorder            — accumulate per-fold, per-class metrics for the report.
  4. fp_rate_by_class()      — false-positive rate per attack type (the headline metric).

The FP fix rationale: SMOTE'ing a 21-row class up to a flat 10k target synthesises
a dense blob that bleeds across the benign region of feature space, which is what
manufactures false positives. So:
    - class has < CW_THRESHOLD real rows  -> NO SMOTE, lean on class_weight.
    - class in [CW_THRESHOLD, target)      -> SMOTE capped at min(target, SMOTE_CAP_MULT * real).
    - class >= target                      -> untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_fscore_support

CW_THRESHOLD = 500       # below this many real rows: class-weight, never SMOTE
SMOTE_TARGET = 10_000    # nominal per-class target for mid-size classes
SMOTE_CAP_MULT = 5       # never synthesise more than 5x the real rows
SMOTE_MIN_REAL = CW_THRESHOLD


@dataclass
class ResamplingPlan:
    smote_strategy: dict[int, int]            # class_idx -> SMOTE target count
    class_weight: dict[int, float]            # class_idx -> weight (for the small classes)
    decisions: dict[int, str]                 # class_idx -> "smote(x3.2)" | "class_weight" | "untouched"


def resampling_plan(y: np.ndarray, class_names: list[str]) -> ResamplingPlan:
    """Decide, per class, between capped-SMOTE and class-weighting. Pure function."""
    vals, cnts = np.unique(y, return_counts=True)
    counts = dict(zip(vals.tolist(), cnts.tolist()))
    total = int(cnts.sum())
    n_classes = len(counts)

    smote_strategy: dict[int, int] = {}
    class_weight: dict[int, float] = {}
    decisions: dict[int, str] = {}

    for cls, cnt in counts.items():
        # Inverse-frequency weight (balanced-style); always available as a fallback signal.
        class_weight[cls] = total / (n_classes * cnt)
        if cnt < CW_THRESHOLD:
            decisions[cls] = "class_weight"          # FP fix: do NOT synthesise tiny classes
            continue
        target = min(SMOTE_TARGET, cnt * SMOTE_CAP_MULT)
        if cnt >= target:
            decisions[cls] = "untouched"
            continue
        smote_strategy[cls] = target
        decisions[cls] = f"smote(x{target / cnt:.1f})"
    return ResamplingPlan(smote_strategy, class_weight, decisions)


def apply_resampling(X: np.ndarray, y: np.ndarray, plan: ResamplingPlan,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Run the capped-SMOTE portion of the plan on ONE fold's training data."""
    if not plan.smote_strategy:
        return X, y
    # k must be < the smallest real class being synthesised
    counts = dict(zip(*np.unique(y, return_counts=True)))
    min_real = min(counts[c] for c in plan.smote_strategy)
    k = max(1, min(5, int(min_real) - 1))
    try:
        sm = SMOTE(sampling_strategy=plan.smote_strategy, k_neighbors=k, random_state=42)
        Xr, yr = sm.fit_resample(X, y)
        return np.asarray(Xr), np.asarray(yr)
    except ValueError:
        return X, y      # too few neighbours — leave the fold as-is, class_weight carries it


def sample_weight_from_plan(y: np.ndarray, plan: ResamplingPlan) -> np.ndarray:
    """Per-row sample weights: class-weighted classes get their weight, others get 1.0.

    After SMOTE the synthesised classes are balanced, so only the class_weight
    classes (the ones we deliberately did NOT synthesise) need up-weighting.
    """
    cw = {c: w for c, w in plan.class_weight.items() if plan.decisions.get(c) == "class_weight"}
    return np.array([cw.get(int(lbl), 1.0) for lbl in y], dtype=float)


def fp_rate_by_class(y_true_label: np.ndarray, y_pred_label: np.ndarray,
                     benign_name: str, attack_names: list[str]) -> dict[str, dict]:
    """For each attack family: how many benign rows were predicted as that family.

    Returns {family: {"fp": int, "benign_total": int, "fp_rate": float}}.
    This is the headline 'huge false positives' metric the user cares about.
    """
    benign_mask = y_true_label == benign_name
    benign_total = int(benign_mask.sum())
    out: dict[str, dict] = {}
    for fam in attack_names:
        fp = int((benign_mask & (y_pred_label == fam)).sum())
        out[fam] = {
            "fp": fp,
            "benign_total": benign_total,
            "fp_rate": (fp / benign_total) if benign_total else 0.0,
        }
    # Aggregate: any benign predicted as ANY attack
    any_fp = int((benign_mask & (y_pred_label != benign_name)).sum())
    out["__ANY__"] = {
        "fp": any_fp, "benign_total": benign_total,
        "fp_rate": (any_fp / benign_total) if benign_total else 0.0,
    }
    return out


@dataclass
class FoldRecorder:
    """Accumulates per-fold metrics into tidy rows for the markdown report."""
    rows: list[dict] = field(default_factory=list)

    def record(self, dataset: str, fold: int, class_names: list[str],
               y_true: np.ndarray, y_pred: np.ndarray,
               stage1_auc: float, log_loss_val: float,
               plan_decisions: dict[int, str]) -> None:
        p, r, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(class_names))),
            average=None, zero_division=0)
        for i, name in enumerate(class_names):
            self.rows.append({
                "dataset": dataset, "fold": fold, "class": name,
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "support": int(sup[i]),
                "stage1_auc": float(stage1_auc), "log_loss": float(log_loss_val),
                "resampling": plan_decisions.get(i, ""),
            })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
