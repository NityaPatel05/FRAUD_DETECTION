"""
Agent-based synthetic payments world ("Track C", docs/IMPROVEMENT_PLAN.md Phase 9).

A SEPARATE simulated dataset with the real-world fields IEEE-CIS lacks (customer /
merchant / terminal ids, MCC, IP + geolocation, device ids, AVS / CVC / 3DS, ISO 8583
response codes, chargeback dates). It is never joined to IEEE-CIS rows.

Fraud is not stamped onto rows after the fact: attacker agents (gangs, rings,
crews) generate their own transactions, and the label is "which process produced the
row". Authorisation outcomes come from mechanisms (CVC fail -> N7, 3DS fail -> 05, ...)
that apply identically to legit and fraud rows.

Entry point: ``simulate(SimConfig(...))`` -> ``SimResult`` (see world.py), or
``scripts/run_simulator.py``.
"""
from .config import SimConfig
from .world import SimResult, simulate

__all__ = ["SimConfig", "SimResult", "simulate"]
