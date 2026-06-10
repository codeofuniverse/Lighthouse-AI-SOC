"""Build proportional (stratified) subsets + DISJOINT validation sets for both
CIC-IDS-2017 and UNSW-NB15.

Why this exists
---------------
The user wants to train on a *small* dataset that is still *proportional* to the
full one (so a 60/20/20 logic on 2M rows behaves the same on a 200k draw), and
they want validation metrics computed on data that is **disjoint** from anything
used in the 80/20 train/test subset.

Strategy (per dataset)
----------------------
1. SUBSET  — for each class keep ``round(0.10 * full_count)`` rows, EXCEPT any
   class with fewer than ``RARE_FLOOR`` rows which is kept in FULL (so the tiny
   Web-Attack / Bot / Worms / Shellcode families survive 5-fold CV + SHAP/LIME).
2. VALIDATION — drawn from rows that are NOT in the subset, so it is provably
   disjoint:
     * CIC  : the complement pool (the ~90% of rows not chosen into the subset).
     * UNSW : the official *testing-set* partition (disjoint from the training
              set the subset is built from, and schema-compatible — the raw
              UNSW-NB15_1..4.csv files use an incompatible NetFlow column set
              that lacks the model's engineered FEATURES, so the testing-set is
              the correct disjoint source).

Outputs (data/models/raw/):
    cic2017_subset_10pct.parquet   cic2017_val.parquet
    unsw_subset_10pct.parquet      unsw_val.parquet

Usage:
    python scripts/sample_proportional.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Windows consoles default to cp1252; force UTF-8 so box-drawing/em-dash output is safe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse the production label maps / feature config — do NOT redefine them here.
from scripts.retrain_cic_smote import (  # noqa: E402
    CSV_DIR as CIC_CSV_DIR,
    EXCLUDE_LABELS as CIC_EXCLUDE,
    LABEL_MAP as CIC_LABEL_MAP,
    FINAL_LABELS as CIC_FINAL_LABELS,
)
from scripts.train_unsw_svm import (  # noqa: E402
    TRAIN_CSV as UNSW_TRAIN_CSV,
    TEST_CSV as UNSW_TEST_CSV,
    COL_ALIASES as UNSW_COL_ALIASES,
    CATEGORY_MERGE as UNSW_MERGE,
    ATTACK_CATEGORIES as UNSW_ATTACK_CATEGORIES,
    _RAW_CATEGORIES as UNSW_RAW_CATEGORIES,
)

warnings.filterwarnings("ignore")

OUT_DIR = Path("data/models/raw")
SEED = 42
SUBSET_FRACTION = 0.10
RARE_FLOOR = 2_000          # classes smaller than this are kept in FULL in the subset
VAL_FRACTION = 0.10         # proportional validation draw (CIC: from complement pool)


# ─────────────────────────────────────────────────────────────────────────────
# Generic proportional draw
# ─────────────────────────────────────────────────────────────────────────────
def _proportional_indices(labels: np.ndarray, fraction: float, rng: np.random.Generator,
                          rare_floor: int, candidate_mask: np.ndarray | None = None,
                          ) -> np.ndarray:
    """Return row indices for a proportional, rare-class-floored draw.

    For each class: keep min(available, max(rare_floor_keep, round(fraction*N)))
    where rare_floor_keep = available if N < rare_floor else round(fraction*N).
    Only rows where candidate_mask is True are eligible.
    """
    n = len(labels)
    eligible = np.ones(n, dtype=bool) if candidate_mask is None else candidate_mask.copy()
    chosen: list[np.ndarray] = []
    for cls in np.unique(labels):
        cls_idx = np.where((labels == cls) & eligible)[0]
        full_count = int((labels == cls).sum())
        if full_count < rare_floor:
            keep = len(cls_idx)                       # floor: take ALL eligible rare rows
        else:
            keep = min(len(cls_idx), int(round(fraction * full_count)))
        if keep <= 0:
            continue
        if keep < len(cls_idx):
            cls_idx = rng.choice(cls_idx, keep, replace=False)
        chosen.append(cls_idx)
    return np.concatenate(chosen) if chosen else np.array([], dtype=int)


def _partition_subset_val(labels: np.ndarray, fraction: float, val_fraction: float,
                          rng: np.random.Generator, rare_floor: int,
                          rare_val_fraction: float = 0.30,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Per-class DISJOINT (subset_idx, val_idx).

    Large classes (>= rare_floor): subset = round(fraction*N), validation =
    round(val_fraction*N), both drawn without overlap.
    Rare classes (< rare_floor): can't be at 10% and still validate, so reserve
    rare_val_fraction of the rows for validation and put the rest in the subset.
    This guarantees every class appears in BOTH the subset and the disjoint val.
    """
    subset_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)
        if n_cls < rare_floor:
            n_val = max(1, int(round(rare_val_fraction * n_cls))) if n_cls > 1 else 0
            n_sub = n_cls - n_val
        else:
            n_sub = int(round(fraction * n_cls))
            n_val = int(round(val_fraction * n_cls))
            n_val = min(n_val, n_cls - n_sub)         # keep disjoint
        subset_parts.append(cls_idx[:n_sub])
        val_parts.append(cls_idx[n_sub:n_sub + n_val])
    subset_idx = np.concatenate(subset_parts) if subset_parts else np.array([], dtype=int)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)
    return subset_idx, val_idx


def _dist_table(title: str, full: pd.Series, subset: pd.Series, val: pd.Series) -> str:
    classes = sorted(set(full.index) | set(subset.index) | set(val.index))
    lines = [f"\n{title}", f"  {'class':<16}{'full':>10}{'subset':>10}{'subset%':>9}{'val':>10}"]
    for c in classes:
        f = int(full.get(c, 0)); s = int(subset.get(c, 0)); v = int(val.get(c, 0))
        pct = (100.0 * s / f) if f else 0.0
        lines.append(f"  {str(c):<16}{f:>10,}{s:>10,}{pct:>8.1f}%{v:>10,}")
    lines.append(f"  {'TOTAL':<16}{int(full.sum()):>10,}{int(subset.sum()):>10,}"
                 f"{'':>9}{int(val.sum()):>10,}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CIC-IDS-2017
# ─────────────────────────────────────────────────────────────────────────────
def _load_cic_full() -> pd.DataFrame:
    frames = []
    for f in sorted(CIC_CSV_DIR.glob("*.csv")):
        print(f"  {f.name} ...", flush=True)
        df = pd.read_csv(f, encoding="utf-8", encoding_errors="replace", low_memory=False)
        df.columns = df.columns.str.strip()
        df["Label"] = df["Label"].astype(str).str.strip()
        # Preserve the fine-grained Web-Attack subtype BEFORE collapsing (for SHAP/LIME)
        df["LabelFine"] = df["Label"]
        for k, v in CIC_LABEL_MAP.items():
            df.loc[df["Label"] == k, "Label"] = v
        df.loc[df["Label"].str.contains("Web Attack", na=False), "Label"] = "Web Attack"
        df = df[~df["Label"].isin(CIC_EXCLUDE)].copy()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["Label"] = combined["Label"].str.strip()
    combined = combined[combined["Label"].isin(CIC_FINAL_LABELS)].reset_index(drop=True)
    return combined


def build_cic_subset() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== CIC-IDS-2017: loading full dataset ===")
    df = _load_cic_full()
    labels = df["Label"].values
    rng = np.random.default_rng(SEED)

    # Per-class disjoint partition so even rare classes (Bot, Web Attack) appear
    # in BOTH the subset and the disjoint validation set.
    subset_idx, val_idx = _partition_subset_val(
        labels, SUBSET_FRACTION, VAL_FRACTION, rng, RARE_FLOOR)

    subset = df.iloc[subset_idx].reset_index(drop=True)
    val = df.iloc[val_idx].reset_index(drop=True)
    print(_dist_table("CIC-IDS-2017 distribution", df["Label"].value_counts(),
                      subset["Label"].value_counts(), val["Label"].value_counts()))
    assert not (set(subset_idx) & set(val_idx)), "CIC val/subset overlap!"
    print(f"  Disjointness: subset-intersect-val = 0 indices  "
          f"(subset={len(subset):,}, val={len(val):,})")
    return subset, val


# ─────────────────────────────────────────────────────────────────────────────
# UNSW-NB15
# ─────────────────────────────────────────────────────────────────────────────
def _load_unsw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=UNSW_COL_ALIASES)
    for col in ["attack_cat", "Attack_cat"]:
        if col in df.columns:
            df = df.rename(columns={col: "Label"})
            break
    df["Label"] = df["Label"].astype(str).str.strip()
    df["Label"] = df["Label"].apply(lambda x: x if x in UNSW_RAW_CATEGORIES else "Generic")
    df["LabelFine"] = df["Label"]                      # pre-merge subtype, for SHAP/LIME
    df["Label"] = df["Label"].replace(UNSW_MERGE)
    df = df[df["Label"].isin(["Normal"] + UNSW_ATTACK_CATEGORIES)].reset_index(drop=True)
    return df


def build_unsw_subset() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== UNSW-NB15: loading training-set (subset source) + testing-set (val source) ===")
    df_train = _load_unsw(UNSW_TRAIN_CSV)
    df_test = _load_unsw(UNSW_TEST_CSV)        # official disjoint partition -> validation

    rng = np.random.default_rng(SEED)
    labels = df_train["Label"].values
    subset_idx = _proportional_indices(labels, SUBSET_FRACTION, rng, RARE_FLOOR)
    subset = df_train.iloc[subset_idx].reset_index(drop=True)

    # Validation = proportional, rare-floored draw from the testing-set partition.
    val_idx = _proportional_indices(df_test["Label"].values, VAL_FRACTION, rng, RARE_FLOOR)
    val = df_test.iloc[val_idx].reset_index(drop=True)

    full_counts = pd.concat([df_train["Label"], df_test["Label"]]).value_counts()
    print(_dist_table("UNSW-NB15 distribution (train+test as 'full')", full_counts,
                      subset["Label"].value_counts(), val["Label"].value_counts()))
    print("  Disjointness: validation is the official UNSW testing-set partition,")
    print("                structurally separate from the training-set subset source.")
    return subset, val


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("  Proportional subset + disjoint validation builder")
    print(f"  fraction={SUBSET_FRACTION}  rare_floor={RARE_FLOOR}  seed={SEED}")
    print("=" * 70)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cic_subset, cic_val = build_cic_subset()
    cic_subset.to_parquet(OUT_DIR / "cic2017_subset_10pct.parquet", index=False)
    cic_val.to_parquet(OUT_DIR / "cic2017_val.parquet", index=False)

    unsw_subset, unsw_val = build_unsw_subset()
    unsw_subset.to_parquet(OUT_DIR / "unsw_subset_10pct.parquet", index=False)
    unsw_val.to_parquet(OUT_DIR / "unsw_val.parquet", index=False)

    print("\n  Saved:")
    for name in ["cic2017_subset_10pct", "cic2017_val", "unsw_subset_10pct", "unsw_val"]:
        p = OUT_DIR / f"{name}.parquet"
        print(f"    {p}  ({p.stat().st_size/1e6:.1f} MB)")
    print("  Done.")


if __name__ == "__main__":
    main()
