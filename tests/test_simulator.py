"""Synthetic payments world (src/simulator). Small worlds only; no model training."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import SimConfig, simulate
from src.simulator.datacard import COLUMN_DOCS, render
from src.simulator.validate import leakage_scan, structure_checks, validate
from src.simulator.world import _COLUMN_ORDER, GROUND_TRUTH_COLS

ALL_SCENARIOS = {"legit", "card_testing", "stolen_card_cnp", "account_takeover", "terminal_cashout",
                 "skimming", "mule_ring", "bust_out", "friendly_fraud"}


@pytest.fixture(scope="module")
def res():
    return simulate(SimConfig.small(seed=7))


@pytest.fixture(scope="module")
def tx(res):
    return res.transactions


def _tiny(seed):
    return simulate(SimConfig.small(seed=seed, n_customers=800, n_merchants=150, n_days=40,
                                    val_days=8, test_days=8, n_gangs=4))


# ── schema & config ──

def test_schema(tx):
    assert list(tx.columns) == _COLUMN_ORDER
    assert set(COLUMN_DOCS) == set(_COLUMN_ORDER)
    for c in ("event_time", "amount", "amount_usd", "currency", "channel", "card_id", "card_bin",
              "merchant_id", "terminal_id", "mcc", "auth_response_code"):
        assert tx[c].notna().all(), c
    assert (tx["amount"] > 0).all()
    assert tx["event_time"].is_monotonic_increasing
    assert tx["transaction_id"].is_unique
    assert set(tx["channel"]) == {"CP", "CNP", "P2P"}
    assert tx["local_hour"].between(0, 23).all()


def test_config_validation():
    with pytest.raises(ValueError):
        SimConfig.from_dict({"no_such_key": 1})
    with pytest.raises(ValueError):
        SimConfig.from_dict({"scenario_shares": {"made_up": 0.1}})
    with pytest.raises(ValueError):
        SimConfig.from_dict({"n_days": 30, "val_days": 15, "test_days": 15}).validate()


def test_deterministic_given_seed():
    a, b, c = _tiny(3), _tiny(3), _tiny(4)
    pd.testing.assert_frame_equal(a.transactions, b.transactions)
    assert not a.transactions["amount"].equals(c.transactions["amount"])


# ── labels & scenarios ──

def test_fraud_rate_and_scenarios(tx):
    assert 0.025 < tx["is_fraud"].mean() < 0.05
    assert set(tx["fraud_scenario"]) == ALL_SCENARIOS
    assert (tx.loc[tx["fraud_scenario"] == "legit", "is_fraud"] == 0).all()
    assert (tx.loc[tx["fraud_scenario"] != "legit", "is_fraud"] == 1).all()


def test_structure_checks_pass(res):
    checks = structure_checks(res)
    assert all(checks.values()), checks


def test_label_delays(tx):
    fr = tx[tx["is_fraud"] == 1]
    delay = (fr["label_date"] - fr["event_time"]).dt.total_seconds() / 86_400
    assert (delay >= 0).all() and (delay <= 121).all()
    approved_card = fr[(fr["approved"] == 1) & (fr["channel"] != "P2P")]
    assert approved_card["chargeback_date"].notna().all()
    assert fr.loc[fr["channel"] == "P2P", "chargeback_date"].isna().all()
    # most chargebacks arrive weeks later, not instantly
    assert (approved_card["chargeback_date"] - approved_card["event_time"]).dt.days.median() >= 7


def test_novel_scenario_only_in_test_period(res, tx):
    test_start = pd.Timestamp(res.metadata["splits"]["test"][0])
    bo = tx[tx["fraud_scenario"] == "bust_out"]
    assert len(bo) > 0 and (bo["event_time"] >= test_start).all()
    # bust-out accounts had a legit history before (they may also have been victims of other fraud)
    hist = tx[(tx["customer_id"].isin(bo["customer_id"])) & (tx["event_time"] < test_start)]
    assert (hist["fraud_scenario"] == "legit").sum() > 0


def test_card_testing_shape(tx):
    ct = tx[tx["fraud_scenario"] == "card_testing"]
    assert (ct["channel"] == "CNP").all()
    assert (ct["amount_usd"] <= 5.5).mean() > 0.95
    invalid = ct["customer_id"].isna()
    assert invalid.any()
    assert ct.loc[invalid, "auth_response_code"].isin(["14", "54"]).mean() > 0.95
    assert ct["approved"].mean() < 0.4


def test_mule_rings_ground_truth(res, tx):
    gt = res.ground_truth
    rings, members = gt["rings"], gt["ring_members"]
    assert len(rings) > 0
    assert set(tx["ring_id"].dropna().unique()) <= set(rings["ring_id"])
    assert set(members["ring_id"]) <= set(rings["ring_id"])
    mule_ids = set(members["customer_id"])
    layer = tx[(tx["fraud_scenario"] == "mule_ring") & tx["customer_id"].isin(mule_ids) & (tx["channel"] == "P2P")]
    # members of a ring operate from a small shared set of devices
    per_ring = layer.groupby("ring_id")["device_id"].nunique()
    assert (per_ring <= 3).all()
    assert members["new_account"].any()


def test_ids_do_not_encode_creation_order(res):
    gt = res.ground_truth["customers"]
    new = (gt["account_kind"] == "mule_new").to_numpy()
    assert new.sum() >= 3
    auc = roc_auc_score(new, gt["customer_id"].to_numpy())
    assert 0.15 < auc < 0.85


# ── realism / leakage ──

def test_channel_specific_fields(tx):
    cp = tx[tx["channel"] == "CP"]
    assert cp["ip"].isna().all() and cp["device_id"].isna().all() and cp["avs_result"].isna().all()
    cnp = tx[tx["channel"] == "CNP"]
    assert cnp["avs_result"].notna().all() and cnp["cvc_check"].notna().all()
    assert cnp["three_ds_result"].notna().all()
    p2p = tx[tx["channel"] == "P2P"]
    assert p2p["counterparty_customer_id"].notna().all()
    assert (p2p["mcc"] == "4829").all()


def test_auth_is_mechanism_based_not_label_based(tx):
    """A CVC mismatch is declined N7 at the same rate whether or not the row is fraud."""
    # real cards only: an invented PAN is declined "14 invalid card" before CVC is even checked
    cvc_n = tx[(tx["cvc_check"] == "N") & tx["customer_id"].notna()]
    rate = cvc_n.groupby("is_fraud")["auth_response_code"].apply(lambda s: (s == "N7").mean())
    assert len(rate) == 2 and abs(rate[0] - rate[1]) < 0.15
    # and plenty of fraud gets approved (otherwise the label would be readable from the outcome)
    assert tx.loc[tx["is_fraud"] == 1, "approved"].mean() > 0.5


def test_no_single_column_gives_fraud_away(res, tx):
    val_start = pd.Timestamp(res.metadata["splits"]["val"][0])
    scan = leakage_scan(tx, val_start)
    assert not set(scan["column"]) & set(GROUND_TRUTH_COLS)
    assert scan["auc"].max() < 0.95, scan.head()


def test_legit_traffic_is_bursty_too(tx):
    """Velocity alone must not separate fraud: legit customers have multi-purchase sessions."""
    legit = tx[tx["is_fraud"] == 0].sort_values(["customer_id", "event_time"])
    gap = legit.groupby("customer_id")["event_time"].diff().dt.total_seconds()
    assert (gap < 3600).mean() > 0.10


def test_validation_and_datacard(res, tmp_path):
    report = validate(res)
    sig = pd.DataFrame(report["signals"]).set_index(["scenario", "signal"])["auc_vs_legit"]
    assert sig[("card_testing", "merchant_burst_1h")] > 0.9
    assert sig[("mule_ring", "device_distinct_customers_30d")] > 0.7
    assert abs(sig[("friendly_fraud", "new_device_for_customer")] - 0.5) < 0.1
    md = render(res.metadata, report)
    assert "Never use as features" in md and "bust_out" in md
    out = res.save(tmp_path / "sim")
    back = pd.read_parquet(out / "transactions.parquet")
    assert back.shape == res.transactions.shape
    assert (out / "ground_truth" / "rings.parquet").exists()
