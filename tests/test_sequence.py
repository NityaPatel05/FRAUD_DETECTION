"""Sequence encoder: causality, incremental==batch parity, determinism, speed."""
import time

import numpy as np
import pandas as pd
import pytest

from src.sequence import SeqConfig, SeqSpec, SequenceEncoder, client_order
from src.sequence.data import gaps


def make_df(n=600, n_clients=40, seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(r.integers(0, 86400 * 30, n)),
        "TransactionAmt": np.round(r.lognormal(3.5, 1, n), 2),
        "ProductCD": r.choice(list("WCHRS"), n), "card4": r.choice(["visa", "mc", None], n),
        "card6": r.choice(["debit", "credit"], n), "P_emaildomain": r.choice(["a.com", "b.com", None], n),
        "uid1_code": r.integers(-1, n_clients, n),
    })
    return df


def build(df, epochs=2, seed=0):
    client = df["uid1_code"].to_numpy()
    order, _, starts = client_order(client, df["TransactionDT"].to_numpy(), df["TransactionID"].to_numpy())
    g = np.empty(len(df)); g[order] = gaps(df["TransactionDT"].to_numpy()[order], starts)
    spec = SeqSpec.fit(df, g, cat_cols=["ProductCD", "card4", "card6", "P_emaildomain"])
    enc = SequenceEncoder(spec, SeqConfig(epochs=epochs, seed=seed, hidden=16, emb_dim=8))
    return spec, enc


def feats(enc, spec, df):
    return enc.transform(spec.encode(df), df["uid1_code"].to_numpy(), df["TransactionID"].to_numpy())


def test_tiny_train_fast_and_deterministic():
    df = make_df()
    t0 = time.time()
    outs = []
    for _ in range(2):
        spec, enc = build(df)
        enc.fit(spec.encode(df), df["uid1_code"].to_numpy(), df["TransactionID"].to_numpy(), log=lambda *_: None)
        outs.append(feats(enc, spec, df))
    assert time.time() - t0 < 60
    pd.testing.assert_frame_equal(outs[0], outs[1])
    assert np.isfinite(outs[0].to_numpy()).all()


def test_causality_future_change_does_not_alter_past():
    df = make_df()
    spec, enc = build(df, epochs=1)
    a = feats(enc, spec, df)
    df2 = df.copy()
    cut = 400
    r = np.random.default_rng(1)
    df2.loc[cut:, "TransactionAmt"] = r.lognormal(6, 1, len(df) - cut)
    df2.loc[cut:, "ProductCD"] = "S"
    df2.loc[cut:, "TransactionDT"] = df2.loc[cut:, "TransactionDT"] + 5000
    b = feats(enc, spec, df2)
    np.testing.assert_allclose(a.iloc[:cut].to_numpy(), b.iloc[:cut].to_numpy(), atol=1e-6)
    assert not np.allclose(a.iloc[cut:].to_numpy(), b.iloc[cut:].to_numpy())


def test_own_row_excluded_from_its_features():
    """Changing only the current txn's fields must not change its embedding (surprise may change)."""
    df = make_df()
    spec, enc = build(df, epochs=1)
    a = feats(enc, spec, df)
    df2 = df.copy(); i = 300
    df2.loc[i, "TransactionAmt"] = 99999.0
    b = feats(enc, spec, df2)
    emb = [c for c in a.columns if c.startswith("seq_emb_")]
    np.testing.assert_allclose(a.loc[i, emb].to_numpy(), b.loc[i, emb].to_numpy(), atol=1e-6)


def test_incremental_equals_batch():
    df = make_df()
    spec, enc = build(df, epochs=1)
    full = feats(enc, spec, df)
    e = spec.encode(df)
    states, rows = {}, []
    order = np.lexsort((df["TransactionID"], df["TransactionDT"]))      # stream in time order
    out = np.zeros((len(df), full.shape[1]), dtype=np.float32)
    for i in order:
        c = int(df["uid1_code"][i])
        key = c if c >= 0 else ("anon", i)
        st = states.get(key) or enc.init_state()
        emb, sur, st = enc.step(st, e.row(i))
        out[i] = np.r_[emb, sur]
        states[key] = st
    np.testing.assert_allclose(out, full.to_numpy(), atol=1e-4)
    assert next(iter(states.values())).nbytes < 200
