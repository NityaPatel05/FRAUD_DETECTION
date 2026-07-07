from __future__ import annotations

from typing import List, Optional

import pandas as pd
import numpy as np
from loguru import logger


D_COLS = ["D1","D2","D3","D4","D5","D6","D7","D8",
          "D9","D10","D11","D12","D13","D14","D15"]

# Priority D columns confirmed by top-5% solution
PRIORITY_D = ["D1_norm","D2_norm","D3_norm","D4_norm","D5_norm",
              "D10_norm","D11_norm","D15_norm"]


def build_d_features(
    df: pd.DataFrame,
    train_idx: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    1. Normalise all D columns: D{n}_norm = D{n} - day
    2. Keep originals (tree models pick the better one automatically)
    3. Add uid1 mean/std aggregations for priority D_norm columns

    Parameters
    ----------
    train_idx : indices of train rows within df. If None, uses all rows.
                ALWAYS pass train_idx when df contains val/test rows to avoid leakage.
    """
    present_d = [c for c in D_COLS if c in df.columns]

    for d in present_d:
        df[f"{d}_norm"] = df[d] - df["day"]

    logger.info(
        "D normalisation done: {} D_norm columns added.  "
        "Example D1_norm stats: mean={:.2f}, std={:.2f}.",
        len(present_d),
        df["D1_norm"].mean() if "D1_norm" in df.columns else float("nan"),
        df["D1_norm"].std()  if "D1_norm" in df.columns else float("nan"),
    )

    # UID aggregations for priority D_norm columns (train-only fit to prevent leakage)
    source = df.iloc[train_idx] if train_idx is not None else df
    present_priority = [c for c in PRIORITY_D if c in df.columns]
    for d in present_priority:
        grp = source.groupby("uid1")[d]
        df[f"{d}_uid_mean"] = df["uid1"].map(grp.mean()).fillna(-999)
        df[f"{d}_uid_std"]  = df["uid1"].map(grp.std()).fillna(-999)

    logger.info(
        "D uid aggregations added for {} priority columns.", len(present_priority)
    )
    return df
