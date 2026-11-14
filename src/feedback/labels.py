"""
Label store with MATURITY (Dal Pozzolo et al.; Fraud Detection Handbook "delay period").

Labels arrive late and from several sources (chargeback weeks later, analyst decision in
hours, customer report). A transaction is a usable NEGATIVE only once its chargeback
window has closed; before that it is unlabelled, not legit. Training on immature
"negatives" teaches the model that recent fraud is legit.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = """CREATE TABLE IF NOT EXISTS labels (
  transaction_id INTEGER, event_time INTEGER, label_time INTEGER,
  is_fraud REAL, source TEXT, PRIMARY KEY (transaction_id, source))"""


class LabelStore:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path))
        self.con.execute(SCHEMA)

    def add(self, transaction_id: int, event_time: int, label_time: int, is_fraud: float, source: str) -> None:
        self.con.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?)",
                         (int(transaction_id), int(event_time), int(label_time), float(is_fraud), source))
        self.con.commit()

    def known_as_of(self, as_of: int) -> pd.DataFrame:
        """Labels that had ARRIVED by as_of; per transaction, a fraud verdict from any
        source wins over a legit one."""
        q = pd.read_sql("SELECT * FROM labels WHERE label_time <= ?", self.con, params=(int(as_of),))
        return (q.sort_values(["transaction_id", "is_fraud"]).groupby("transaction_id").tail(1)
                 .reset_index(drop=True))

    def training_labels(self, txns: pd.DataFrame, as_of: int, maturity_days: int,
                        id_col: str = "TransactionID", time_col: str = "TransactionDT") -> pd.Series:
        """Label usable for training at time ``as_of``:
            1    fraud label already known
            0    no fraud known AND the transaction is older than the maturity window
            NaN  still immature (or in the future) → exclude from training
        """
        known = self.known_as_of(as_of).set_index("transaction_id")["is_fraud"]
        y = txns[id_col].map(known)
        mature = (txns[time_col] <= as_of - maturity_days * 86400).to_numpy()
        y = pd.Series(np.where(y == 1, 1.0, np.where(mature, 0.0, np.nan)), index=txns.index)
        y[(txns[time_col] > as_of).to_numpy()] = np.nan
        return y
