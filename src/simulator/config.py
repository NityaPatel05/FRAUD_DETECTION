"""Simulator configuration. Defaults give ~1.3M transactions over 182 days."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Dict


def _default_shares() -> Dict[str, float]:
    # Share of the fraud-row budget for the TRAIN+VAL+TEST period. "bust_out" (the novel
    # scenario) is budgeted separately, against the test period only.
    return {
        "card_testing": 0.18,
        "stolen_card_cnp": 0.30,
        "account_takeover": 0.12,
        "terminal_cashout": 0.08,
        "skimming": 0.10,
        "mule_ring": 0.14,
        "friendly_fraud": 0.08,
    }


@dataclass
class SimConfig:
    seed: int = 42
    start_date: str = "2025-01-01"
    n_days: int = 182
    val_days: int = 30            # validation period = the 30 days before the test period
    test_days: int = 30           # test period = the last 30 days (novel scenario lives only here)

    n_customers: int = 50_000
    n_merchants: int = 3_000
    tx_rate_mean: float = 0.15    # legit transactions per customer per day (gamma-distributed)

    target_fraud_rate: float = 0.035          # IEEE-CIS train 3.51%, val 3.46%
    scenario_shares: Dict[str, float] = field(default_factory=_default_shares)
    novel_share_of_test_fraud: float = 0.10   # bust_out rows / all fraud rows in the test period
    n_gangs: int = 40

    # ── derived helpers ──
    @property
    def horizon_sec(self) -> int:
        return self.n_days * 86_400

    @property
    def test_start_sec(self) -> int:
        return (self.n_days - self.test_days) * 86_400

    @property
    def val_start_sec(self) -> int:
        return (self.n_days - self.test_days - self.val_days) * 86_400

    def validate(self) -> None:
        if self.n_days > 1000:
            raise ValueError("n_days must be <= 1000 (feature code assumes t < 1e8 seconds)")
        if self.val_days + self.test_days >= self.n_days:
            raise ValueError("val_days + test_days must leave a training period")
        if not 0 < self.target_fraud_rate < 0.5:
            raise ValueError("target_fraud_rate must be in (0, 0.5)")
        unknown = set(self.scenario_shares) - set(_default_shares())
        if unknown:
            raise ValueError(f"unknown scenarios in scenario_shares: {sorted(unknown)}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SimConfig":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown simulator config keys: {sorted(unknown)}")
        if "scenario_shares" in d:
            shares = _default_shares()
            shares.update(d["scenario_shares"] or {})
            d["scenario_shares"] = shares
        cfg = cls(**d)
        cfg.validate()
        return cfg

    @classmethod
    def small(cls, **overrides) -> "SimConfig":
        """A quick world for tests / smoke runs (~40k rows, a few seconds)."""
        base = dict(n_customers=3_000, n_merchants=400, n_days=90, val_days=15,
                    test_days=15, n_gangs=8)
        base.update(overrides)
        return cls.from_dict(base)
