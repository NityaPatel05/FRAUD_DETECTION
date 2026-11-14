from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from loguru import logger


# ─────────────────────────────────────────────────────────
# 1. Group V columns by NaN pattern
# ─────────────────────────────────────────────────────────

def group_v_cols_by_nan(
    df: pd.DataFrame,
    v_prefix: str = "V",
) -> Dict[int, List[str]]:
    """
    Group V columns by their NaN count (proxy: same NaN count = same sensor block).
    Returns {nan_count: [col1, col2, ...]}
    """
    v_cols = [c for c in df.columns if c.startswith(v_prefix) and c[1:].isdigit()]
    nan_counts = df[v_cols].isna().sum()

    groups: Dict[int, List[str]] = defaultdict(list)
    for col in v_cols:
        groups[int(nan_counts[col])].append(col)

    logger.info(
        "V columns: {} total, grouped into {} NaN-pattern blocks.",
        len(v_cols), len(groups),
    )
    for k, v in sorted(groups.items())[:5]:
        logger.debug("  NaN={}: {} cols (e.g. {})", k, len(v), v[:3])

    return dict(groups)


# ─────────────────────────────────────────────────────────
# 2. Reduce by correlation within each group (tree models)
# ─────────────────────────────────────────────────────────

def reduce_v_by_correlation(
    df: pd.DataFrame,
    groups: Dict[int, List[str]],
    corr_threshold: float = 0.9,
) -> List[str]:
    """
    Within each NaN group, keep only columns whose pairwise Pearson correlation
    with all already-kept columns is below corr_threshold.

    Returns the list of kept V column names.
    """
    kept_cols: List[str] = []

    for nan_count, group in sorted(groups.items()):
        if len(group) == 1:
            kept_cols.append(group[0])
            continue

        # Fill NaN with median (0 inflates correlation for columns with same NaN pattern)
        sub = df[group].fillna(df[group].median())
        corr_matrix = sub.corr().abs()

        # Greedy: always keep first col, skip later ones that are too similar
        group_kept = [group[0]]
        for col in group[1:]:
            # Check against all already-kept cols in this group
            max_corr = corr_matrix.loc[col, group_kept].max()
            if max_corr < corr_threshold:
                group_kept.append(col)

        kept_cols.extend(group_kept)
        logger.debug(
            "NaN={}: {} cols → {} after corr reduction (threshold={}).",
            nan_count, len(group), len(group_kept), corr_threshold,
        )

    logger.info(
        "V correlation reduction: {} → {} columns (threshold={}).",
        sum(len(g) for g in groups.values()), len(kept_cols), corr_threshold,
    )
    return kept_cols


# ─────────────────────────────────────────────────────────
# 3. PCA within NaN groups (DL models — VAE / BiLSTM)
# ─────────────────────────────────────────────────────────

def fit_v_pca(
    train_df: pd.DataFrame,
    groups: Dict[int, List[str]],
    n_components: int = 3,
    min_group_size: int = 3,
) -> Dict[int, PCA]:
    """
    Fit one PCA per NaN group (only groups with >= min_group_size columns).
    Returns {nan_count: fitted_PCA}.

    IMPORTANT: Fit on train_df ONLY. Apply to val/test via apply_v_pca().
    """
    pcas: Dict[int, PCA] = {}

    for nan_count, group in sorted(groups.items()):
        if len(group) < min_group_size:
            continue
        X = train_df[group].fillna(0).values
        k = min(n_components, len(group))
        pca = PCA(n_components=k, random_state=42)
        pca.fit(X)
        pcas[nan_count] = pca
        logger.debug(
            "PCA fitted — NaN={}: {} cols → {} components  "
            "(explained var: {:.3f})",
            nan_count, len(group), k,
            pca.explained_variance_ratio_.sum(),
        )

    logger.info("Fitted {} PCA objects across V NaN groups.", len(pcas))
    return pcas


def apply_v_pca(
    df: pd.DataFrame,
    groups: Dict[int, List[str]],
    pcas: Dict[int, PCA],
    split_name: str = "data",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply pre-fitted PCAs to df.  Adds PCA component columns to df.
    Returns (modified_df, list_of_new_pca_col_names).
    """
    pca_cols: List[str] = []

    for nan_count, pca in pcas.items():
        group = groups[nan_count]
        X = df[group].fillna(0).values
        components = pca.transform(X)

        for i in range(components.shape[1]):
            col_name = f"V_pca_nan{nan_count}_pc{i}"
            df[col_name] = components[:, i]
            pca_cols.append(col_name)

    logger.info(
        "Applied V PCA to {} — added {} PCA component columns.",
        split_name, len(pca_cols),
    )
    return df, pca_cols


# ─────────────────────────────────────────────────────────
# 4. UID-level aggregations on top V columns
# ─────────────────────────────────────────────────────────

IMPORTANT_V_COLS = ["V258", "V257", "V201", "V189", "V187", "V294", "V83", "V76"]


def add_uid_v_aggregations(
    df: pd.DataFrame,
    v_cols: List[str] = IMPORTANT_V_COLS,
    uid_col: str = "uid1",
    train_idx: np.ndarray = None,
) -> pd.DataFrame:
    """
    Add uid-level mean/std of top V columns.
    If train_idx is given, statistics are computed on train rows only
    and filled in for all rows (correct — no leakage).
    """
    present_v = [c for c in v_cols if c in df.columns]
    if not present_v:
        logger.warning("None of the important V cols found in df. Skipping uid-V aggs.")
        return df

    if train_idx is not None:
        train_df = df.iloc[train_idx]
        for v in present_v:
            # Compute stats from train rows only, map back to all rows by uid key
            mean_map = train_df.groupby(uid_col)[v].mean()
            std_map  = train_df.groupby(uid_col)[v].std()
            df[f"{v}_uid_mean"] = df[uid_col].map(mean_map).fillna(np.nan)
            df[f"{v}_uid_std"]  = df[uid_col].map(std_map).fillna(np.nan)
    else:
        for v in present_v:
            df[f"{v}_uid_mean"] = df.groupby(uid_col)[v].transform("mean")
            df[f"{v}_uid_std"]  = df.groupby(uid_col)[v].transform("std")

    added = [f"{v}_uid_{stat}" for v in present_v for stat in ("mean", "std")]
    logger.info("Added {} uid-V aggregation columns.", len(added))
    return df


# ─────────────────────────────────────────────────────────
# 4. Deotte-style reduction (v2 tree pipeline)
# ─────────────────────────────────────────────────────────

def reduce_v_keep_most_unique(
    df: pd.DataFrame,
    groups: Dict[int, List[str]],
    corr_threshold: float = 0.75,
) -> List[str]:
    """
    Within each NaN-pattern group, cluster columns whose |corr| > threshold
    (connected components of the correlation graph) and keep, from every
    cluster, the column with the MOST unique values (most information).

    The legacy greedy version kept whichever column came first, which is
    arbitrary. Label-free, so fitting on train rows is enough.
    """
    kept: List[str] = []
    for _, group in sorted(groups.items()):
        if len(group) == 1:
            kept.extend(group)
            continue
        corr = df[group].corr().abs().fillna(0.0).to_numpy()
        n = len(group)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if corr[i, j] > corr_threshold:
                    parent[find(i)] = find(j)
        clusters: Dict[int, List[str]] = defaultdict(list)
        for i, col in enumerate(group):
            clusters[find(i)].append(col)
        for cols in clusters.values():
            kept.append(max(cols, key=lambda c: df[c].nunique()))
    kept = sorted(kept, key=lambda c: int(c[1:]))
    logger.info(
        "V reduction (most-unique per corr>{} cluster): {} → {} columns.",
        corr_threshold, sum(len(g) for g in groups.values()), len(kept),
    )
    return kept
