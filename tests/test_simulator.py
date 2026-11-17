"""Synthetic payments world v2 (src/simulator). Small worlds only; no model training beyond the
row-only ceiling check (a HistGBDT on a ~50k-row table)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import (ACTIONS, AllApprovePolicy, CallablePolicy, DecisionPolicy, Decisions, RulesPolicy,
                           ScorePolicy, SimConfig, simulate)
from src.simulator.datacard import COLUMN_DOCS, render
from src.simulator.events import EVENT_COLUMNS
from src.simulator.scenarios import Tactics
from src.simulator.validate import (calibration_table, daily_acf, label_checks, leakage_scan, purity_check,
                                    row_only_ceiling, structure_checks, validate)
from src.simulator.world import _COLUMN_ORDER

DAY = 86_400


@pytest.fixture(scope="module")
def res():
    return simulate(SimConfig.small(seed=7))


@pytest.fixture(scope="module")
def tx(res):
    return res.transactions


def _tiny(seed, policy=None, **kw):
    base = dict(seed=seed, n_customers=600, n_merchants=120, n_days=45, val_days=6, test_days=10, n_gangs=4,
                target_fraud_rate=0.02)
    base.update(kw)
    return simulate(SimConfig.small(**base), policy=policy)


# ── schema, config, determinism ──

def test_schema(tx):
    assert list(tx.columns) == _COLUMN_ORDER
    assert set(COLUMN_DOCS) == set(_COLUMN_ORDER)
    for c in ("event_time", "amount", "amount_usd", "currency", "channel", "card_id", "card_bin",
              "merchant_id", "terminal_id", "mcc", "auth_response_code", "decision", "label_status"):
        assert tx[c].notna().all(), c
    assert (tx["amount"] > 0).all()
    assert tx["event_time"].is_monotonic_increasing
    assert tx["transaction_id"].is_unique
    assert set(tx["channel"]) == {"CP", "CNP", "P2P"}
    assert set(tx["txn_type"]) == {"purchase", "refund"}
    assert set(tx["decision"]) <= set(ACTIONS)
    assert tx["local_hour"].between(0, 23).all()


def test_config_validation_and_presets():
    with pytest.raises(ValueError):
        SimConfig.from_dict({"no_such_key": 1})
    with pytest.raises(ValueError):
        SimConfig.from_dict({"scenario_shares": {"made_up": 0.1}})
    with pytest.raises(ValueError):
        SimConfig.from_dict({"n_days": 30, "val_days": 15, "test_days": 15}).validate()
    with pytest.raises(ValueError):
        SimConfig.from_dict({"preset": "nope"})
    assert SimConfig().preset == "realistic" and SimConfig().target_fraud_rate < 0.005
    assert SimConfig.from_preset("ieee_like").target_fraud_rate > 0.03
    h = SimConfig.from_preset("hard")
    assert h.friendly_share_of_disputes == 0.30 and h.report_prob["card_testing"] == 0.20
    assert h.report_prob["skimming"] == SimConfig().report_prob["skimming"]      # partial override merges


def test_deterministic_given_seed():
    a, b, c = _tiny(3), _tiny(3), _tiny(4)
    pd.testing.assert_frame_equal(a.transactions, b.transactions)
    pd.testing.assert_frame_equal(a.events, b.events)
    assert not a.transactions["amount"].equals(c.transactions["amount"])


# ── leak removal & purity ──

def test_category_purity(tx):
    pur = purity_check(tx)
    assert pur.empty, pur
    # the checker itself fires on a planted pure category
    bad = tx.copy()
    bad.loc[bad["is_fraud_true"] == 1, "email_domain"] = "planted.example"
    assert not purity_check(bad).empty


def test_no_deterministic_leaks(tx):
    leg = tx[tx["is_fraud_true"] == 0]
    fr = tx[tx["is_fraud_true"] == 1]
    # every fraud email domain is also used by legit customers
    assert set(fr["email_domain"].dropna()) <= set(leg["email_domain"].dropna())
    for d in ("protonmail.com", "mail.ru", "yandex.ru", "guerrillamail.com", "tutanota.com"):
        assert d in set(leg["email_domain"]), d
    # null customer is not a fraud flag (guest checkouts), and invented PANs have owner ids
    assert leg["customer_id"].isna().sum() > 50
    assert fr["customer_id"].notna().mean() > 0.9
    assert tx.loc[tx["fraud_scenario"] == "card_testing", "customer_id"].notna().all()
    # decline codes 14 / 54 also happen to legit rows
    for code in ("14", "54"):
        assert (leg["auth_response_code"] == code).sum() > 10, code
    # legit Linux / headless traffic exists
    assert ((leg["device_os"] == "Linux")).sum() > 20 and (leg["browser"] == "Headless Chrome").sum() > 5


def test_row_only_ceiling(res, tx):
    """Row-level tells exist but are never near-deterministic. AUCs on tiny scenario samples are noisy, so
    individual scenarios are only asserted from n >= 40 (loose bound) and the n-weighted mean is held tighter."""
    r = row_only_ceiling(tx, pd.Timestamp(res.metadata["splits"]["val"][0]))
    assert r["per_scenario"], r
    easy = ("card_testing", "skimming")       # allowed to be easy (burst pattern / cross-border card-present)
    ps = {s: v for s, v in r["per_scenario"].items() if s not in easy}
    for s, v in ps.items():
        if v["n"] >= 40:
            assert v["auc"] < 0.97, (s, v)
    big = [v for v in ps.values() if v["n"] >= 12]
    if big:
        mean = sum(v["auc"] * v["n"] for v in big) / sum(v["n"] for v in big)
        assert mean < 0.95, (mean, ps)
    assert r["overall_auc"] < 0.95


def test_no_single_column_gives_fraud_away(res, tx):
    scan = leakage_scan(tx, pd.Timestamp(res.metadata["splits"]["val"][0]))
    assert scan["auc"].max() < 0.95, scan.head()
    assert not (set(scan["column"]) & {"is_fraud", "is_fraud_true", "reported_label", "label_status"})


# ── label process ──

def test_label_gates(res, tx):
    lab = label_checks(res)
    assert lab["unobserved_has_no_label"] and lab["label_date_not_before_event"]
    assert lab["declined_legit_all_unobserved_or_reviewed"]
    assert lab["chargebacks_inside_window"]
    assert 25 <= lab["chargeback_delay_median_days"] <= 55
    cfg = res.metadata["config"]
    declined_fraud = tx[(tx["approved"] == 0) & (tx["is_fraud_true"] == 1)]
    share = (declined_fraud["label_status"] != "unobserved").mean()
    assert share < 0.5 and share > 0.0          # configured contact share + analyst reviews, never all
    assert (declined_fraud["label_status"] == "unobserved").mean() > 0.5
    # no oracle: some approved fraud is never reported (looks legit), some is reversed
    fr = tx[tx["is_fraud_true"] == 1]
    assert (fr["label_status"] == "unreported").any() and (fr["reported_label"] == 0).any()
    assert (tx["label_status"] == "reversed").any()
    unrep = fr[fr["label_status"] == "unreported"]
    assert (unrep["label_date"] - unrep["event_time"]).dt.days.eq(cfg["label_window_days"]).all()
    # chargebacks only on approved rows; reversed rows have reported 0
    cb = tx[tx["chargeback_date"].notna()]
    assert (cb["approved"] == 1).all() and (cb["label_status"].isin(["chargeback", "reversed"])).all()
    assert (tx.loc[tx["label_status"] == "reversed", "reported_label"] == 0).all()


def test_friendly_fraud_and_backward_compat(tx):
    ff = tx[tx["fraud_scenario"] == "friendly_fraud"]
    assert len(ff) > 0
    assert (ff["is_fraud_true"] == 0).all()
    assert (ff.loc[ff["label_status"] != "reversed", "reported_label"] == 1).all()   # representment can reverse a dispute
    assert (ff["approved"] == 1).all() and ff["dispute_reason"].notna().all() is np.True_
    assert (tx["is_fraud"] == tx["is_fraud_true"]).all()
    disputes = tx[(tx["reported_label"] == 1) & tx["label_status"].isin(["chargeback", "reported", "reversed"])]
    friendly_share = (disputes["is_fraud_true"] == 0).mean()
    assert 0.05 <= friendly_share <= 0.35


def test_structure_checks_pass(res):
    checks = structure_checks(res)
    assert all(checks.values()), checks


# ── scenarios ──

def test_scenarios_present(tx):
    got = set(tx["fraud_scenario"])
    need = {"legit", "card_testing", "stolen_card_cnp", "account_takeover", "terminal_cashout", "skimming",
            "mule_ring", "synthetic_identity", "synthetic_identity_build", "merchant_collusion",
            "transaction_laundering", "bust_out", "friendly_fraud"}
    assert need <= got, need - got
    assert (tx.loc[tx["fraud_scenario"] == "legit", "is_fraud_true"] == 0).all()
    assert (tx.loc[tx["fraud_scenario"].isin(["friendly_fraud", "synthetic_identity_build"]), "is_fraud_true"] == 0).all()
    assert (tx.loc[~tx["fraud_scenario"].isin(["legit", "friendly_fraud", "synthetic_identity_build"]), "is_fraud_true"] == 1).all()


def test_novel_scenario_only_in_test_period(res, tx):
    test_start = pd.Timestamp(res.metadata["splits"]["test"][0])
    assert res.metadata["novel_scenarios"] == ["bust_out"]
    bo = tx[tx["fraud_scenario"] == "bust_out"]
    assert len(bo) > 0 and (bo["event_time"] >= test_start).all()
    hist = tx[(tx["customer_id"].isin(bo["customer_id"])) & (tx["event_time"] < test_start)]
    assert (hist["fraud_scenario"] == "legit").sum() > 0
    # the other first-party scenario (synthetic identity) is present in training
    si = tx[tx["fraud_scenario"].isin(["synthetic_identity", "synthetic_identity_build"])]
    assert (si["event_time"] < test_start).any()


def test_card_testing_shape(tx):
    ct = tx[tx["fraud_scenario"] == "card_testing"]
    assert (ct["channel"] == "CNP").all()
    assert (ct["amount_usd"] <= 5.5).mean() > 0.95
    assert ct["customer_id"].notna().all()                       # phantom owners, not null
    assert ct["approved"].mean() < 0.5


def test_mule_rings_ground_truth(res, tx):
    gt = res.ground_truth
    rings, members = gt["rings"], gt["ring_members"]
    assert len(rings) > 0
    assert set(tx["ring_id"].dropna().unique()) <= set(rings["ring_id"])
    mule_ids = set(members["customer_id"])
    layer = tx[(tx["fraud_scenario"] == "mule_ring") & tx["customer_id"].isin(mule_ids) & (tx["channel"] == "P2P")]
    per_ring = layer.groupby("ring_id")["device_id"].nunique()
    assert (per_ring <= 3).all()
    assert members["new_account"].any()


def test_synthetic_identity_ramp(tx):
    si = tx[tx["fraud_scenario"].isin(["synthetic_identity_build", "synthetic_identity"])]
    spree = si[si["fraud_scenario"] == "synthetic_identity"]
    build = si[si["fraud_scenario"] == "synthetic_identity_build"]
    assert len(spree) and len(build)
    first_spree = spree.groupby("customer_id")["event_time"].min()
    b = build[build["customer_id"].isin(first_spree.index)]
    last_build = b.groupby("customer_id")["event_time"].max()
    assert ((first_spree.reindex(last_build.index) - last_build).dt.total_seconds() > -DAY).mean() > 0.9
    # months of thin good behaviour, small tickets, then high-value spree
    span = (first_spree.reindex(last_build.index) - b.groupby("customer_id")["event_time"].min()).dt.days
    assert span.median() >= 30
    assert build["amount_usd"].median() < spree["amount_usd"].median()


def test_merchant_side_fraud_and_refunds(res, tx):
    mt = res.ground_truth["merchants"]
    assert (mt["merchant_kind"] == "collusive").any()
    col = tx[tx["fraud_scenario"] == "merchant_collusion"]
    assert (col["txn_type"] == "refund").any()                   # refund abuse
    assert col["merchant_id"].nunique() >= 1
    ref = tx[(tx["txn_type"] == "refund") & (tx["fraud_scenario"] == "legit")]
    assert len(ref) > 50
    orig = tx.set_index("transaction_id").loc[ref["original_transaction_id"].dropna().astype(int)]
    assert (orig["txn_type"] == "purchase").all() and (orig["approved"] == 1).all()
    assert (ref["approved"] == 1).all() and (ref["decision"] == "approve").all()


# ── closed loop ──

def test_card_reissue_linkage(res, tx):
    cards = res.entities["cards"].set_index("card_id")
    new = cards[cards["prev_card_id"].notna()]
    assert len(new) > 20
    prev = cards.loc[new["prev_card_id"]]
    assert prev["blocked_date"].notna().all()
    assert (new["issued_date"].to_numpy() > prev["blocked_date"].to_numpy()).all()
    assert (new["customer_id"].to_numpy() == prev["customer_id"].to_numpy()).all()
    # after a block: no later use of the blocked card by legit traffic except stale card-on-file (54)
    blocked = cards["blocked_date"].dropna()
    use = tx[tx["card_id"].isin(blocked.index)].copy()
    use["blocked"] = use["card_id"].map(blocked)
    late = use[(use["event_time"] > use["blocked"]) & (use["fraud_scenario"] == "legit")]
    assert (late["auth_response_code"] == "54").all() and (late["recurring"] == 1).all()
    # transactions on reissued cards carry the link
    on_new = tx[tx["card_id"].isin(new.index)]
    assert on_new["prev_card_id"].notna().all()


def test_policy_interface_and_propensity():
    seen = {"days": [], "label_rows": 0, "hist_cols": None, "frame_cols": None}

    class P(DecisionPolicy):
        name = "test"

        def observe(self, labels, day):
            seen["label_rows"] += len(labels)
            start = pd.Timestamp(self.start)
            assert (labels["label_date"] <= start + pd.Timedelta(days=day)).all()      # no label from the future
            assert set(labels.columns) == {"transaction_id", "label_date", "reported_label", "label_status", "dispute_reason"}

        def decide(self, batch):
            seen["days"].append(batch.day)
            seen["frame_cols"] = set(batch.frame.columns)
            h = batch.history.frame()
            seen["hist_cols"] = set(h.columns)
            f = batch.frame
            act = np.where((f["channel"] == "CNP") & (f["amount_usd"] > 400), "decline",
                           np.where((f["channel"] == "CNP") & (f["amount_usd"] > 200), "step_up_3ds", "approve"))
            return Decisions(act.astype(object), np.clip(f["amount_usd"].to_numpy() / 1000, 0, 1))

    p = P()
    p.start = "2025-01-01T00:00:00+00:00"
    r = _tiny(5, policy=p, explore_rate=0.3)
    tx = r.transactions
    assert seen["days"] == list(range(len(seen["days"]))) and len(seen["days"]) > 40
    assert seen["label_rows"] > 0
    assert any(c.startswith("hist_card_n_") for c in seen["frame_cols"])
    assert not (seen["hist_cols"] & {"reported_label", "label_status", "label_date", "is_fraud_true", "fraud_scenario"})
    assert not (seen["frame_cols"] & {"approved", "reported_label", "is_fraud_true", "label_date"})
    assert r.metadata["policy"] == "test"
    pur = tx[tx["txn_type"] == "purchase"]
    big = pur[(pur["channel"] == "CNP") & (pur["amount_usd"] > 400)]
    # a would-be decline let through by exploration is logged as approve with propensity = explore_rate
    let = big[big["declined_by"] != "policy"]
    assert (let[let["declined_by"] == "none"]["propensity"].round(3) == 0.3).all() or let[let["declined_by"] == "none"].empty
    pol = big[big["declined_by"] == "policy"]
    assert len(pol) > 0 and (pol["propensity"].round(3) == 0.7).all() and (pol["auth_response_code"] == "59").all()
    assert pol["label_status"].isin(["unobserved", "confirmed"]).all()
    su = pur[pur["three_ds_result"].isin(["C", "N"]) & (pur["decision"] == "step_up_3ds")]
    assert len(su) > 0 and su.loc[su["three_ds_result"] == "N", "declined_by"].isin(["sca", "issuer"]).mean() >= 0.7   # few rows; a natural 3DS N still slips through 5% of the time
    assert (su.loc[su["three_ds_result"] == "C", "liability_shift"].isin([0, 1])).all()


def test_policy_validation_rejects_bad_actions():
    bad = CallablePolicy(lambda f: (np.full(len(f), "block", dtype=object), np.zeros(len(f))))
    with pytest.raises(ValueError):
        _tiny(1, policy=bad, n_days=30, val_days=4, test_days=6)


def test_policy_changes_outcomes():
    base = _tiny(8, policy=AllApprovePolicy())
    strict = _tiny(8, policy=ScorePolicy(lambda f: (f["amount_usd"] > 300).astype(float), t_decline=0.5, t_review=2, t_stepup=2))
    assert (base.transactions["declined_by"] == "policy").sum() == 0
    assert (strict.transactions["declined_by"] == "policy").sum() > 0
    assert strict.transactions["approved"].mean() < base.transactions["approved"].mean()


def test_review_capacity_respected():
    r = _tiny(6, policy=CallablePolicy(lambda f: (np.full(len(f), "review", dtype=object), np.linspace(0, 1, len(f)))))
    tx = r.transactions
    rv = tx[tx["label_status"].isin(["review_confirmed", "review_cleared"])]
    per_day = rv.groupby(rv["event_time"].dt.floor("D")).size()
    day_rows = tx.groupby(tx["event_time"].dt.floor("D")).size().reindex(per_day.index)
    assert (per_day <= np.ceil(0.004 * day_rows) + 1).all() and len(rv) > 0


def test_attacker_bandit_unit():
    rng = np.random.default_rng(0)
    t = Tactics(2, True, rng)
    for _ in range(40):          # arm 2 is the only one that ever gets approved
        t.update(np.zeros(3, np.int64), np.array([[0, 1, 1, 0], [1, 1, 1, 0], [2, 1, 1, 0]]), np.array([0.0, 0.0, 1.0]))
    picks = [t.choose(0, "ip") for _ in range(300)]
    assert np.mean(np.array(picks) == 2) > 0.7
    other = [t.choose(1, "ip") for _ in range(300)]          # gang 1 learned nothing: stays on its prior
    assert np.mean(np.array(other) == 2) < 0.5


def test_attackers_adapt_to_decisions():
    kw = dict(n_customers=2500, n_merchants=250, n_days=100, val_days=10, test_days=14, n_gangs=3,
              target_fraud_rate=0.04, seed=11, clean_attack_share=0.0)   # isolate the bandit from blend-in attackers

    class NoHosting(DecisionPolicy):
        name = "no_hosting"

        def decide(self, batch):
            f = batch.frame
            bad = (f["ip_type"] == "hosting") & (f["channel"] == "CNP")
            return Decisions(np.where(bad, "decline", "approve").astype(object), bad.astype(float))

    def late_hosting_share(r):
        tx = r.transactions
        t0 = tx["event_time"].min()
        d = (tx["event_time"] - t0).dt.days
        m = (tx["is_fraud_true"] == 1) & tx["fraud_scenario"].isin(["stolen_card_cnp", "account_takeover"]) & tx["ip_type"].notna()
        early, late = m & (d < 40), m & (d >= 60)
        return (tx.loc[early, "ip_type"] == "hosting").mean(), (tx.loc[late, "ip_type"] == "hosting").mean(), int(late.sum())

    blocked = simulate(SimConfig.small(**kw), policy=NoHosting())
    free = simulate(SimConfig.small(**kw), policy=AllApprovePolicy())
    e_b, l_b, n_b = late_hosting_share(blocked)
    e_f, l_f, n_f = late_hosting_share(free)
    assert n_b > 40 and n_f > 40
    # row-level hosting shares are noisy with 3 gangs; the choices themselves (tactic attempts) are not
    def arm0_share(r):
        t = r.ground_truth["gang_tactics"]
        a = t[t["dim"] == "ip"].groupby("arm")["attempts"].sum()
        return a.loc[0] / a.sum()
    assert arm0_share(blocked) < arm0_share(free)        # gangs facing the hosting-IP rule move away from it
    tac = blocked.ground_truth["gang_tactics"]
    ip = tac[tac["dim"] == "ip"].groupby("arm")[["approvals", "attempts"]].sum()
    assert ip.loc[0, "approvals"] / max(ip.loc[0, "attempts"], 1) < ip.drop(0)["approvals"].sum() / max(ip.drop(0)["attempts"].sum(), 1)


# ── calendar, presets, events ──

def test_calendar_weekly_autocorrelation_and_paydays(res, tx):
    assert daily_acf(tx, res.metadata["config"]["n_days"], 7) > 0.3
    t = tx[(tx["is_fraud_true"] == 0) & (tx["recurring"] == 0) & (tx["txn_type"] == "purchase")]
    dow = t["event_time"].dt.dayofweek.value_counts(normalize=True)
    assert dow.max() / dow.min() > 1.15                       # not a flat weekly profile
    day = t["event_time"].dt.floor("D").value_counts()
    first = day[day.index.day.isin([1, 2, 15, 16])].mean()
    other = day[~day.index.day.isin([1, 2, 15, 16, 28, 29, 30, 31])].mean()
    assert first > other
    sub = tx[tx["recurring"] == 1]
    assert len(sub) > 500 and sub["amount_usd"].round(2).groupby(sub["customer_id"].astype(str) + sub["merchant_id"].astype(str)).nunique().max() <= 2     # at most one price change


def test_presets_prevalence_within_tolerance():
    real = simulate(SimConfig.from_preset("realistic", n_customers=3500, n_merchants=350, n_days=100, val_days=12,
                                          test_days=18, n_gangs=6, seed=2))
    r = real.transactions["is_fraud_true"].mean()
    assert 0.0006 <= r <= 0.012, r      # tiny worlds are lumpy: one collusion/ring incident is a large share
    ieee = simulate(SimConfig.from_preset("ieee_like", n_customers=1500, n_merchants=250, n_days=60, val_days=8,
                                          test_days=10, n_gangs=5, seed=2))
    r2 = ieee.transactions["is_fraud_true"].mean()
    assert 0.02 <= r2 <= 0.05, r2
    assert ieee.metadata["config"]["label_mode"] == "oracle"
    # oracle labels: every fraud row is labelled fraud; realistic ones are not
    assert (ieee.transactions.loc[ieee.transactions["is_fraud_true"] == 1, "reported_label"] == 1).all()
    assert (real.transactions.loc[real.transactions["is_fraud_true"] == 1, "reported_label"] == 1).mean() < 0.9
    # fraud is CNP-heavy by value in the realistic world
    cal = calibration_table(real)
    row = cal[cal["statistic"].str.startswith("CNP share")]
    assert len(row) == 1


def test_account_events(res, tx):
    ev = res.events
    assert list(ev.columns[:len(EVENT_COLUMNS)]) == EVENT_COLUMNS
    assert set(ev["event_type"]) >= {"login", "password_reset", "email_change", "phone_change", "device_enrolment", "new_payee"}
    assert ev["event_time"].is_monotonic_increasing and ev["event_id"].is_unique
    leg = ev[ev["is_attacker"] == 0]
    att = ev[ev["is_attacker"] == 1]
    # legit users also enrol devices, change emails, fail logins: the event type alone is not a fraud flag
    for et in ("device_enrolment", "email_change", "password_reset"):
        assert (leg["event_type"] == et).sum() > 5, et
    assert (~leg.loc[leg["event_type"] == "login", "success"]).any()
    # ATO: login from a NEW device -> credential change -> purchases after a dwell
    ato = tx[tx["fraud_scenario"] == "account_takeover"]
    assert len(ato) > 0
    first = ato.groupby("incident_id")["event_time"].min()
    a_login = att[(att["event_type"] == "login") & att["success"]].set_index("incident_id")["event_time"]
    j = pd.concat([first.rename("tx"), a_login.rename("login")], axis=1, join="inner")
    assert len(j) > 3 and ((j["tx"] - j["login"]).dt.total_seconds() > 0).all()
    assert att["event_type"].isin(["password_reset", "email_change", "phone_change"]).any()
    # attacker devices are new to the customer; legit login devices are known
    dev_first = tx[tx["fraud_scenario"] == "legit"].groupby("customer_id")["device_id"].agg(set)
    a = att[(att["event_type"] == "login") & att["success"] & att["device_id"].notna()]
    known = [d in dev_first.get(c, set()) for c, d in zip(a["customer_id"], a["device_id"])]
    assert np.mean(known) < 0.3


def test_validation_save_and_datacard(res, tmp_path):
    report = validate(res, ceiling=False)
    assert report["purity"]["pass"]
    assert report["calibration"] and 0 <= report["calibration_pass_rate"] <= 1
    sig = pd.DataFrame(report["signals"]).set_index(["scenario", "signal"])["auc_vs_legit"]
    assert sig[("card_testing", "merchant_burst_1h")] > 0.9
    assert sig[("mule_ring", "device_distinct_customers_30d")] > 0.7
    assert abs(sig[("friendly_fraud", "new_device_for_customer")] - 0.5) < 0.12
    md = render(res.metadata, report)
    assert "Calibration table" in md and "bust_out" in md and "Category purity" in md
    out = res.save(tmp_path / "sim")
    back = pd.read_parquet(out / "transactions.parquet")
    assert back.shape == res.transactions.shape
    assert (out / "events.parquet").exists() and (out / "ground_truth" / "rings.parquet").exists()
    assert (out / "ground_truth" / "gang_tactics.parquet").exists()


def test_ids_do_not_encode_creation_order(res):
    gt = res.ground_truth["customers"]
    new = (gt["account_kind"] != "regular").to_numpy()      # accounts appended after the base population
    assert new.sum() >= 3
    auc = roc_auc_score(new, gt["customer_id"].to_numpy())
    assert 0.15 < auc < 0.85
    cards = res.entities["cards"]
    reissued = cards["prev_card_id"].notna().to_numpy()
    assert 0.15 < roc_auc_score(reissued, cards["card_id"].to_numpy()) < 0.85


def test_channel_specific_fields(tx):
    cp = tx[tx["channel"] == "CP"]
    assert cp["ip"].isna().all() and cp["device_id"].isna().all() and cp["avs_result"].isna().all()
    cnp = tx[(tx["channel"] == "CNP") & (tx["txn_type"] == "purchase")]
    assert cnp["avs_result"].notna().all() and cnp["cvc_check"].notna().all()
    assert cnp["three_ds_result"].notna().all()
    p2p = tx[tx["channel"] == "P2P"]
    assert p2p["counterparty_customer_id"].notna().all() and (p2p["mcc"] == "4829").all()


def test_auth_is_mechanism_based_not_label_based(tx):
    """A CVC mismatch is declined N7 at the same rate whether or not the row is fraud."""
    cvc_n = tx[(tx["cvc_check"] == "N") & (tx["declined_by"].isin(["issuer", "none"]))]
    rate = cvc_n.groupby("is_fraud_true")["auth_response_code"].apply(lambda s: (s == "N7").mean())
    n_by = cvc_n.groupby("is_fraud_true").size()
    if len(rate) == 2 and n_by.min() >= 40:
        assert abs(rate[0] - rate[1]) < 0.2
    assert tx.loc[tx["is_fraud_true"] == 1, "approved"].mean() > 0.5


def test_legit_traffic_is_bursty_too(tx):
    legit = tx[(tx["is_fraud_true"] == 0) & tx["customer_id"].notna()].sort_values(["customer_id", "event_time"])
    gap = legit.groupby("customer_id")["event_time"].diff().dt.total_seconds()
    assert (gap < 3600).mean() > 0.10
