"""Phase 6 novelty layer — synthetic data only, no model training (except a tiny
Isolation Forest / KMeans in discovery, which uses no labels)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.novelty.attack_detection import detect_attacks, add_amount_band
from src.novelty.rule_mining import mine_contrast_rules, rule_to_conditions
from src.novelty.rules_engine import evaluate_rule_df, evaluate_rules_row, backtest, save_rules, load_rules
from src.novelty.drift import psi, psi_report
from src.novelty.discovery import discover_unusual_clusters

H = 3600


@pytest.fixture(scope="module")
def traffic():
    """20 days of normal traffic + a 3-hour card-testing burst on day 15."""
    r = np.random.default_rng(0)
    n = 40_000
    df = pd.DataFrame({
        "TransactionDT": np.sort(r.integers(0, 20 * 86400, n)),
        "ProductCD": r.choice(list("WCHRS"), n, p=[.6, .1, .1, .1, .1]),
        "card4": r.choice(["visa", "mastercard"], n),
        "card6": r.choice(["debit", "credit"], n, p=[.7, .3]),
        "TransactionAmt": r.lognormal(4, 1, n).round(2),
        "isFraud": (r.random(n) < 0.03).astype(float),
    })
    start = 15 * 86400 + 10 * H
    burst = pd.DataFrame({
        "TransactionDT": r.integers(start, start + 3 * H, 600),
        "ProductCD": "C", "card4": "visa", "card6": "credit",
        "TransactionAmt": r.uniform(1, 3, 600).round(2), "isFraud": 1.0,
    })
    df = pd.concat([df, burst], ignore_index=True).sort_values("TransactionDT").reset_index(drop=True)
    df["amount_band"] = add_amount_band(df)
    return df, start


def test_attack_detected_in_right_segment_and_time(traffic):
    df, start = traffic
    alerts = detect_attacks(df, ["ProductCD", "card6", "amount_band"], period_sec=H,
                            z_threshold=6, min_periods=2, min_volume=20)
    assert len(alerts) > 0
    hit = alerts[(alerts.segment == "ProductCD") & (alerts.value == "C")]
    assert len(hit) == 1
    assert hit.iloc[0].start_time <= start + H and hit.iloc[0].end_time >= start + 2 * H
    # baseline is past-only: nothing fires before the burst
    assert (alerts.start_time >= start - H).all()


def test_contrast_rules_find_injected_pattern(traffic):
    df, start = traffic
    alert_rows = df[(df.TransactionDT >= start) & (df.TransactionDT < start + 3 * H) & (df.ProductCD == "C")]
    base = df[(df.TransactionDT >= start - 7 * 86400) & (df.TransactionDT < start)]
    rules = mine_contrast_rules(alert_rows, base, ["ProductCD", "card4", "card6", "amount_band"],
                                min_support=0.3, min_lift=3, y_col="isFraud")
    top = " ".join(rules.rule.head(3))
    assert "amount_band=[0.0, 10.0)" in top or "card6=credit" in top
    assert rules.iloc[0].lift >= 3


def test_rules_engine_eval_backtest_expiry_and_priority(tmp_path, traffic):
    df, start = traffic
    rule = {"id": "r1", "conditions": rule_to_conditions("ProductCD=C&card6=credit"),
            "action": "review", "mode": "enforce", "expires_at": start + 86400}
    m = evaluate_rule_df(df, rule)
    exp = (df.ProductCD == "C") & (df.card6 == "credit") & (df.TransactionDT < start + 86400)
    assert (m == exp.to_numpy()).all()
    bt = backtest(df, rule)
    assert bt["hits"] == int(((df.ProductCD == "C") & (df.card6 == "credit")).sum())
    assert 0 < bt["precision"] <= 1 and 0 < bt["recall"] <= 1
    row = {"ProductCD": "C", "card6": "credit", "TransactionAmt": 2.0}
    allow = {"id": "a1", "conditions": [{"col": "TransactionAmt", "op": "lt", "value": 5}],
             "action": "allow", "mode": "enforce", "expires_at": None}
    shadow = {"id": "s1", "conditions": [{"col": "card6", "op": "eq_str", "value": "credit"}],
              "action": "block", "mode": "shadow", "expires_at": None}
    res = evaluate_rules_row(row, [rule, allow, shadow], now=start)
    assert res["action"] == "allow" and set(res["fired"]) == {"r1", "a1"} and res["shadow_fired"] == ["s1"]
    assert evaluate_rules_row(row, [rule], now=start + 2 * 86400)["action"] is None   # expired
    save_rules([rule], tmp_path / "rules.yaml")
    assert load_rules(tmp_path / "rules.yaml")[0]["id"] == "r1"


def test_psi_behaviour():
    r = np.random.default_rng(1)
    a = r.normal(0, 1, 20_000)
    assert psi(a, r.normal(0, 1, 20_000)) < 0.02
    assert psi(a, r.normal(1.0, 1, 20_000)) > 0.2
    b = a.copy(); b[:5000] = np.nan
    assert psi(a, b) > 0.1                                   # missingness shift is detected
    rep = psi_report(pd.DataFrame({"x": a}), pd.DataFrame({"x": a + 1}), ["x"])
    assert rep.loc[0, "level"] == "investigate"


def test_discovery_returns_clusters(traffic):
    df, _ = traffic
    out = discover_unusual_clusters(df.head(5000), ["TransactionAmt"], ["ProductCD"], top_frac=0.02, n_clusters=3)
    assert 1 <= len(out) <= 3 and "top_ProductCD" in out.columns


def test_hmm_features_are_finite_and_causal():
    from src.feature_engineering.hmm_features import add_hmm_features
    r = np.random.default_rng(3)
    n = 3000
    df = pd.DataFrame({"uid1": r.choice([f"u{i}" for i in range(150)] + [None], n),
                       "card1_addr1": r.choice([f"c{i}" for i in range(40)], n),
                       "TransactionAmt": r.lognormal(4, 1, n)})
    t = np.sort(r.integers(0, 60 * 86400, n)).astype(np.int64)
    is_train = t < 40 * 86400
    out = add_hmm_features(df, t, is_train, n_states=3)
    s = out["hmm_client_amt_surprise"]
    known = df["uid1"].notna().to_numpy()
    assert np.isfinite(s[known]).all() and np.isnan(s[~known]).all()
    # changing a FUTURE amount must not change earlier rows (HMM fitted on train rows only)
    df2 = df.copy(); df2.loc[n - 1, "TransactionAmt"] = 1e6
    s2 = add_hmm_features(df2, t, is_train, n_states=3)["hmm_client_amt_surprise"]
    np.testing.assert_allclose(s[: n - 1], s2[: n - 1], equal_nan=True)
