"""Track C features on a small simulated world — causality checks, no training."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim_track.features import build_sim_features, label_arrival_counts, GROUND_TRUTH, OUTCOMES


@pytest.fixture(scope="module")
def world():
    from src.simulator.config import SimConfig
    from src.simulator.world import simulate
    res = simulate(SimConfig.small())
    df = res.transactions if hasattr(res, "transactions") else res.df
    return df.sort_values(["event_time", "transaction_id"]).reset_index(drop=True)


def test_features_build_and_exclude_ground_truth(world):
    out, feats = build_sim_features(world)
    assert len(out) == len(world)
    assert not set(feats) & set(GROUND_TRUTH + OUTCOMES + ["transaction_id", "event_time"])
    assert all(out[c].dtype.kind == "f" for c in feats)
    for c in ["terminal_id_fraud_rate", "device_id_distinct_card_id_1d", "kmh_from_prev_geo",
              "card_id_prior_declines_1h", "customer_id_new_device_id", "receiver_distinct_senders_7d"]:
        assert c in feats


def test_features_never_see_the_future(world):
    """Change everything after a cut-off (amounts, approvals, labels, label dates):
    features of rows before the cut-off must not change."""
    out1, feats = build_sim_features(world)
    cut = world["event_time"].quantile(0.6)
    w2 = world.copy()
    late = w2["event_time"] > cut
    w2.loc[late, "amount_usd"] *= 9
    w2.loc[late, "approved"] = 1 - w2.loc[late, "approved"]
    w2.loc[late, "is_fraud"] = 1 - w2.loc[late, "is_fraud"]
    w2.loc[late, "label_date"] = w2.loc[late, "event_time"] + pd.Timedelta(seconds=1)
    out2, _ = build_sim_features(w2)
    early = (out1["event_time"] <= cut).to_numpy()
    pd.testing.assert_frame_equal(out1.loc[early, feats].reset_index(drop=True),
                                  out2.loc[early, feats].reset_index(drop=True), check_dtype=False)


def test_label_arrival_counts_respects_label_dates():
    codes = np.array([0, 0, 0, 0])
    t = np.array([0, 10, 20, 30])
    lab = np.array([25.0, np.nan, 5.0, np.nan])     # row0 fraud reported at 25; row2 at 5 (<own t? not possible, ignored)
    y = np.array([1, 0, 1, 0])
    c = label_arrival_counts(codes, t, lab, y)
    # row1 (t=10): row2's label time 5 < 10 counts (data quirk), row0's (25) not yet
    np.testing.assert_array_equal(c, [0, 1, 1, 2])
