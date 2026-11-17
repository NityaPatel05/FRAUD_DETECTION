"""Row features available to a policy at decision time: the public pre-decision columns plus the engine's
causal ``hist_*`` aggregates. No ground truth, no labels. (src/sim_track/features.py needs a whole
frame with entity histories and label arrivals; inside the daily loop the engine already supplies the
causal aggregates, so this stays cheap and exactly causal.)"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

CAT_COLS = ["channel", "pos_entry_mode", "txn_type", "ip_type", "avs_result", "cvc_check", "three_ds_result",
            "email_domain", "mcc", "card_type", "card_network", "device_os", "browser"]
GEO_COLS = ["ip_country", "billing_country", "shipping_country", "merchant_country"]


class Featurizer:
    """Stateful only through the categorical vocabularies (codes assigned on first sight, stable after)."""

    def __init__(self):
        self.vocab: Dict[str, Dict[str, int]] = {c: {} for c in CAT_COLS}
        self.columns: List[str] | None = None

    def _code(self, col: str, s: pd.Series) -> np.ndarray:
        v = self.vocab[col]
        vals = s.astype("object").where(s.notna(), None).to_numpy()
        out = np.full(len(vals), -1.0, dtype=np.float32)
        for u in pd.unique(vals):
            if u is None or (isinstance(u, float) and np.isnan(u)):
                continue
            k = str(u)
            if k not in v:
                v[k] = len(v)
            out[vals == u] = v[k]
        return out

    def transform(self, f: pd.DataFrame) -> pd.DataFrame:
        n = len(f)
        X: Dict[str, np.ndarray] = {}
        num = lambda c: pd.to_numeric(f[c], errors="coerce").astype("float64").to_numpy() if c in f else np.full(n, np.nan)
        amt = num("amount_usd")
        X["log_amt"] = np.log1p(np.maximum(amt, 0))
        X["amount_usd"] = amt
        X["local_hour"] = num("local_hour")
        X["account_age_days"] = num("account_age_days")
        X["merchant_age_days"] = num("merchant_age_days")
        X["recurring"] = num("recurring")
        X["has_customer"] = f["customer_id"].notna().to_numpy().astype(float)
        X["has_device"] = f["device_id"].notna().to_numpy().astype(float)
        X["has_ip"] = f["ip"].notna().to_numpy().astype(float)
        X["reissued_card"] = f["prev_card_id"].notna().to_numpy().astype(float)
        card = f["card_country"].astype("object").to_numpy()
        for g in GEO_COLS:
            col = f[g].astype("object")
            X[f"{g}_mismatch"] = (col.notna().to_numpy() & (col.to_numpy() != card)).astype(float)
        lm = num("hist_card_logamt_mean")
        X["amt_z_card"] = (np.log(np.maximum(amt, 0.01)) - lm) / 0.9
        for c in f.columns:
            if c.startswith("hist_"):
                X[c] = num(c)
        for c in CAT_COLS:
            if c in f:
                X[c] = self._code(c, f[c])
        out = pd.DataFrame(X)
        if self.columns is None:
            self.columns = list(out.columns)
        return out.reindex(columns=self.columns).astype(np.float32)
