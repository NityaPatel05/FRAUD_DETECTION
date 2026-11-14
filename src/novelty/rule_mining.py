"""
Contrast-pattern rule mining (Uber RADAR uses FP-Growth for the same job).

Given the rows inside an alert window and rows from a baseline window, find
feature=value combinations (1 or 2 items) that are strongly OVER-represented in
the alert window. Each becomes a candidate rule, to be backtested
(rules_engine.backtest) and approved by an analyst before enforcing.

Implemented directly with pandas (no extra dependency); itemsets are limited to
pairs, which is what analysts can read and act on anyway.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Sequence

import numpy as np
import pandas as pd


def _items(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: c + "=" + df[c].astype(str) for c in cols}, index=df.index)


def mine_contrast_rules(alert_rows: pd.DataFrame, base_rows: pd.DataFrame, cols: Sequence[str],
                        min_support: float = 0.05, min_lift: float = 3.0, max_rules: int = 20,
                        y_col: str | None = None) -> pd.DataFrame:
    """Return candidate rules ranked by lift × support.

    support_alert = share of alert-window rows matching the itemset
    support_base  = share of baseline rows matching it (smoothed)
    lift          = support_alert / support_base
    fraud_rate    = fraud rate among matching alert rows (if labels known yet)
    """
    cols = [c for c in cols if c in alert_rows.columns]
    A, B = _items(alert_rows, cols), _items(base_rows, cols)
    nA, nB = len(A), max(len(B), 1)
    out: List[dict] = []
    for k in (1, 2):
        for combo in combinations(cols, k):
            keyA = A[list(combo)].agg("&".join, axis=1) if k > 1 else A[combo[0]]
            keyB = B[list(combo)].agg("&".join, axis=1) if k > 1 else B[combo[0]]
            sa = keyA.value_counts() / nA
            sa = sa[sa >= min_support]
            if sa.empty:
                continue
            sb = keyB.value_counts().reindex(sa.index).fillna(0)
            sb = (sb + 1) / (nB + 1)                      # Laplace smoothing
            lift = sa / sb
            for item, l in lift[lift >= min_lift].items():
                m = keyA == item
                rec = {"rule": item, "n_items": k, "support_alert": float(sa[item]),
                       "support_base": float(sb[item]), "lift": float(l), "hits_alert": int(m.sum())}
                if y_col and y_col in alert_rows.columns:
                    yy = alert_rows.loc[m, y_col]
                    rec["fraud_rate_known"] = float(yy.mean()) if yy.notna().any() else np.nan
                out.append(rec)
    if not out:
        return pd.DataFrame(columns=["rule", "n_items", "support_alert", "support_base", "lift", "hits_alert"])
    res = pd.DataFrame(out)
    res["score"] = res["lift"] * res["support_alert"]
    # drop pairs that add nothing over one of their single items
    singles = res[res.n_items == 1].set_index("rule")["lift"].to_dict()
    def redundant(r):
        if r.n_items == 1:
            return False
        a, b = r.rule.split("&")
        return r.lift <= 1.1 * max(singles.get(a, 0), singles.get(b, 0))
    res = res[~res.apply(redundant, axis=1)]
    return res.sort_values("score", ascending=False).head(max_rules).reset_index(drop=True)


def rule_to_conditions(rule: str) -> List[dict]:
    """'ProductCD=C&card6=credit' → [{'col','op','value'}…] for rules_engine."""
    conds = []
    for part in rule.split("&"):
        col, val = part.split("=", 1)
        conds.append({"col": col, "op": "eq_str", "value": val})
    return conds
