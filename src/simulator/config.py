"""Simulator configuration and presets (v2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Dict

SCENARIOS = ["card_testing", "stolen_card_cnp", "account_takeover", "terminal_cashout", "skimming",
             "mule_ring", "synthetic_identity", "merchant_collusion", "transaction_laundering"]
NOVEL_SCENARIO = "bust_out"      # held out: generated only in the test period


def _default_shares() -> Dict[str, float]:
    # share of the (non-novel) fraud-row budget; friendly fraud is NOT a generated scenario in v2:
    # it is a share of disputes (``friendly_share_of_disputes``) on legit purchases.
    return {
        "card_testing": 0.22, "stolen_card_cnp": 0.27, "account_takeover": 0.10,
        "terminal_cashout": 0.04, "skimming": 0.06, "mule_ring": 0.09,
        "synthetic_identity": 0.06, "merchant_collusion": 0.10, "transaction_laundering": 0.06,
    }


def _report_prob() -> Dict[str, float]:
    """Gate B: P(an APPROVED fraud row is ever reported by a victim / merchant / analyst)."""
    return {"card_testing": 0.30, "stolen_card_cnp": 0.85, "account_takeover": 0.90,
            "terminal_cashout": 0.80, "skimming": 0.80, "mule_victim": 0.55, "mule_layer": 0.10,
            "mule_cashout": 0.25, "synthetic_identity": 0.70, "merchant_collusion": 0.75,
            "transaction_laundering": 0.65, "bust_out": 0.85}


def _report_delay() -> Dict[str, float]:
    """Median days from the fraud row until the victim notices and contacts the issuer."""
    return {"card_testing": 12, "stolen_card_cnp": 4, "account_takeover": 2, "terminal_cashout": 14,
            "skimming": 10, "mule_ring": 5, "synthetic_identity": 60, "merchant_collusion": 8,
            "transaction_laundering": 10, "bust_out": 25}


def _cb_median() -> Dict[str, float]:
    """Median chargeback delay (days) per scenario; missing -> ``cb_delay_median_days``."""
    return {"synthetic_identity": 80.0, "bust_out": 55.0}


@dataclass
class SimConfig:
    preset: str = "realistic"
    seed: int = 42
    start_date: str = "2025-01-01"
    n_days: int = 420
    val_days: int = 45            # validation period = the days before the test period
    test_days: int = 60           # test period = the last days (the novel scenario lives only here)

    n_customers: int = 30_000
    n_merchants: int = 3_000
    tx_rate_mean: float = 0.15    # legit purchases per customer per day (gamma-distributed)

    target_fraud_rate: float = 0.0015         # fraud ATTEMPT rows / all rows (is_fraud_true)
    scenario_shares: Dict[str, float] = field(default_factory=_default_shares)
    novel_share_of_test_fraud: float = 0.08   # bust_out rows / all fraud rows in the test period
    n_gangs: int = 40
    clean_attack_share: float = 0.55          # share of stolen-card / ATO incidents run by attackers who blend in (no row-level tells)

    # ── population & calendar ──
    new_account_frac: float = 0.25            # customers whose account opens during the horizon
    promo_days: int = 14
    promo_boost: float = 2.5                  # early-life activity multiplier
    subscription_share: float = 0.25          # customers with recurring billing
    guest_checkout_share: float = 0.12        # legit e-commerce rows without a customer id
    calendar: bool = True                     # weekly cycle, paydays, holidays
    campaign_waves: bool = True               # clustered (Hawkes-like) fraud campaigns
    mix_shift: bool = True                    # gradual drift of the fraud mix
    refund_rate: float = 0.025                # share of approved CNP legit purchases later refunded

    # ── label process (3 gates + corruption) ──
    label_mode: str = "realistic"             # "realistic" | "oracle" (every fraud row confirmed fast)
    label_window_days: int = 120
    cb_delay_median_days: float = 38.0
    cb_delay_sigma: float = 0.6
    representment_rate: float = 0.12          # chargebacks reversed after the merchant's reply
    declined_fraud_confirm_share: float = 0.15  # Gate A: declined fraud confirmed via cardholder contact
    friendly_share_of_disputes: float = 0.20  # friendly fraud: legit purchase disputed as fraud
    friendly_reversal_rate: float = 0.35
    report_prob: Dict[str, float] = field(default_factory=_report_prob)
    report_delay_days: Dict[str, float] = field(default_factory=_report_delay)
    cb_median_by_scenario: Dict[str, float] = field(default_factory=_cb_median)

    # ── closed loop ──
    adaptive_attackers: bool = True
    explore_rate: float = 0.02                # P(a would-be decline is let through), logged as propensity
    review_capacity_frac: float = 0.004       # analysts review at most this share of the day's rows
    review_accuracy: float = 0.90
    stepup_legit_pass: float = 0.93
    reissue_lag_days: tuple = (1.0, 5.0)
    account_updater_share: float = 0.70       # card-on-file credentials refreshed after a reissue
    history_days: int = 14                    # history buffer handed to the policy

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
        if self.label_mode not in ("realistic", "oracle"):
            raise ValueError("label_mode must be 'realistic' or 'oracle'")
        unknown = set(self.scenario_shares) - set(SCENARIOS)
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
        preset = d.get("preset", "realistic")
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        merged = dict(PRESETS[preset])
        merged.update(d)
        merged["preset"] = preset
        for key, base in (("scenario_shares", _default_shares()), ("report_prob", _report_prob()),
                          ("report_delay_days", _report_delay()), ("cb_median_by_scenario", _cb_median())):
            v = dict(base)
            v.update(merged.get(key) or {})
            merged[key] = v
        if isinstance(merged.get("reissue_lag_days"), list):
            merged["reissue_lag_days"] = tuple(merged["reissue_lag_days"])
        cfg = cls(**merged)
        cfg.validate()
        return cfg

    @classmethod
    def from_preset(cls, name: str = "realistic", **overrides) -> "SimConfig":
        return cls.from_dict({"preset": name, **overrides})

    @classmethod
    def small(cls, **overrides) -> "SimConfig":
        """A quick world for tests / smoke runs (~45k rows, ~10 s). Fraud prevalence is raised to
        1% so that every scenario occurs; use ``from_preset`` for faithful prevalence."""
        base = dict(n_customers=3_000, n_merchants=400, n_days=120, val_days=15, test_days=20,
                    n_gangs=8, target_fraud_rate=0.01)
        base.update(overrides)
        return cls.from_dict(base)


# Presets are overrides on the dataclass defaults (the defaults ARE the `realistic` preset).
PRESETS: Dict[str, dict] = {
    # ≈0.15% fraud attempts by count, CNP-heavy by value, delayed + selective labels, adaptive attackers
    "realistic": {},
    # ≈3.5% fraud, oracle-ish labels, static attackers, no calendar effects: the IEEE-CIS-shaped world
    "ieee_like": dict(target_fraud_rate=0.035, n_days=182, val_days=30, test_days=30, label_mode="oracle",
                      adaptive_attackers=False, calendar=False, campaign_waves=False, mix_shift=False,
                      friendly_share_of_disputes=0.08, explore_rate=0.0, refund_rate=0.0,
                      subscription_share=0.0, new_account_frac=0.08),
    # experiment-oriented: realistic mechanics but ~0.5% fraud and 2x customers so that val and test each
    # hold thousands of fraud rows (stable metrics); prevalence is NOT calibrated to the real world
    "realistic_dense": dict(target_fraud_rate=0.005, n_customers=60_000, n_merchants=5_000, n_gangs=60),
    # realistic prevalence, adaptive + faster-moving attackers, noisier labels
    "hard": dict(target_fraud_rate=0.002, friendly_share_of_disputes=0.30, representment_rate=0.20,
                 declined_fraud_confirm_share=0.05, explore_rate=0.01,
                 report_prob={"stolen_card_cnp": 0.70, "account_takeover": 0.75, "card_testing": 0.20,
                              "mule_victim": 0.40, "merchant_collusion": 0.60}),
}
