"""
Agent-based synthetic payments world (v2): a closed-loop stand-in for a real payment system.

Customers, merchants, terminals, devices, IPs; attacker agents (gangs, rings, crews, synthetic
identities, colluding merchants) that ADAPT to the decisions of a pluggable ``DecisionPolicy``;
delayed, selective, noisy labels; card reissue; an account-event stream. Never joined to IEEE-CIS.

Entry point: ``simulate(SimConfig.from_preset("realistic"), policy=...)`` -> ``SimResult`` (world.py),
or ``scripts/run_simulator.py``. See docs/SYNTHETIC_DATA.md.
"""
from .config import PRESETS, SimConfig
from .loop import (ACTIONS, AllApprovePolicy, CallablePolicy, DecisionPolicy, Decisions, EpochBatch,
                   RulesPolicy, ScorePolicy)
from .world import SimResult, simulate

__all__ = ["SimConfig", "PRESETS", "SimResult", "simulate", "DecisionPolicy", "Decisions", "EpochBatch",
           "RulesPolicy", "ScorePolicy", "CallablePolicy", "AllApprovePolicy", "ACTIONS"]
