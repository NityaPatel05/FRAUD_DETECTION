"""
Versioned, expiring, backtestable rules (Stripe Radar / Monzo / Nubank pattern).

A rule is a dict (stored in YAML):
  id: r_2026_11_14_productC_credit
  conditions: [{col: ProductCD, op: eq_str, value: C}, {col: card6, op: eq_str, value: credit}]
  action: review            # review | step_up | block | allow
  mode: shadow              # shadow (log only) → enforce
  expires_at: 1700000000    # unix seconds (TransactionDT units offline); rules MUST expire
  created_from: "attack alert ProductCD=C 2026-11-14"
  min_score: null           # optional: only fire when model score >= this

Lifecycle: mined → backtested → analyst approves → shadow → enforce → expires.
The same `evaluate_*` code runs offline (DataFrame) and online (single dict).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

OPS = {
    "eq_str": lambda s, v: s.astype(str) == str(v),
    "eq": lambda s, v: s == v,
    "ne": lambda s, v: s != v,
    "gt": lambda s, v: s > v,
    "ge": lambda s, v: s >= v,
    "lt": lambda s, v: s < v,
    "le": lambda s, v: s <= v,
    "in": lambda s, v: s.astype(str).isin([str(x) for x in v]),
}
ACTION_PRIORITY = {"allow": 4, "block": 3, "review": 2, "step_up": 1}   # allow overrides (Stripe)


def evaluate_rule_df(df: pd.DataFrame, rule: Dict, now_col: str = "TransactionDT",
                     score: Optional[np.ndarray] = None) -> np.ndarray:
    m = np.ones(len(df), dtype=bool)
    for c in rule["conditions"]:
        if c["col"] not in df.columns:
            return np.zeros(len(df), dtype=bool)
        m &= OPS[c["op"]](df[c["col"]], c["value"]).fillna(False).to_numpy()
    if rule.get("expires_at") is not None and now_col in df.columns:
        m &= df[now_col].to_numpy() < rule["expires_at"]
    if rule.get("min_score") is not None and score is not None:
        m &= score >= rule["min_score"]
    return m


def evaluate_rules_row(row: Dict, rules: List[Dict], now: float, score: Optional[float] = None,
                       enforce_only: bool = True) -> Dict:
    """Online path: returns {'action', 'fired': [ids], 'shadow_fired': [ids]}."""
    frame = pd.DataFrame([row])
    fired, shadow = [], []
    for r in rules:
        if r.get("expires_at") is not None and now >= r["expires_at"]:
            continue
        hit = evaluate_rule_df(frame, {**r, "expires_at": None},
                               score=None if score is None else np.array([score]))[0]
        if hit:
            (fired if r.get("mode") == "enforce" else shadow).append(r["id"])
    acts = [r["action"] for r in rules if r["id"] in fired]
    action = max(acts, key=lambda a: ACTION_PRIORITY[a]) if acts else None
    return {"action": action, "fired": fired, "shadow_fired": shadow}


def backtest(df: pd.DataFrame, rule: Dict, y_col: str = "isFraud",
             amount_col: str = "TransactionAmt", score: Optional[np.ndarray] = None) -> Dict:
    """Precision / recall / volume of a rule on labelled history (matured labels only)."""
    lab = df[y_col].notna().to_numpy()
    hit = evaluate_rule_df(df, {**rule, "expires_at": None}, score=score) & lab
    y = df[y_col].to_numpy()
    n_hit, tp = int(hit.sum()), int(np.nansum(y[hit]))
    n_fraud = int(np.nansum(y[lab]))
    amt = df[amount_col].to_numpy()
    return {
        "rule_id": rule.get("id"), "hits": n_hit, "hit_rate": n_hit / max(int(lab.sum()), 1),
        "precision": tp / n_hit if n_hit else float("nan"),
        "recall": tp / n_fraud if n_fraud else float("nan"),
        "fraud_amount_caught": float(np.nansum(amt[hit] * y[hit])),
        "legit_amount_affected": float(np.nansum(amt[hit] * (1 - y[hit]))),
    }


def load_rules(path: str | Path) -> List[Dict]:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or [] if p.exists() else []


def save_rules(rules: List[Dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(rules, sort_keys=False), encoding="utf-8")
