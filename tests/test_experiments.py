"""Phase 9 sequence experiment — causality of sequences + model shape (no training)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.sequence_embedding import build_sequences, _model


def test_sequences_are_causal_and_per_client():
    df = pd.DataFrame({"uid1_code": [0, 1, 0, 0, -1, 1],
                       "TransactionDT": [100, 200, 3700, 7300, 7400, 9000],
                       "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                       "hour": [0, 0, 1, 2, 2, 2], "ProductCD": [0.0, 1, 0, 0, 2, 1]})
    X, M = build_sequences(df, k=3)
    assert X.shape == (6, 4, 5) and M.shape == (6, 4)
    # row 3 (client 0, third txn): current + 2 previous of client 0, one padded slot
    assert M[3].tolist() == [0, 1, 1, 1]
    np.testing.assert_allclose(np.expm1(X[3, 1:, 0]), [10, 30, 40], rtol=1e-5)
    # unknown client (-1) never gets someone else's history; its gap is 0
    assert M[4].tolist() == [0, 0, 0, 1] and X[4, 3, 1] == 0
    # client 1's second txn sees only client 1's first txn
    np.testing.assert_allclose(np.expm1(X[5, 2:, 0]), [20, 60], rtol=1e-5)
    assert np.isclose(np.expm1(X[5, 3, 1]), (9000 - 200) / 3600, rtol=1e-5)


def test_gru_forward_shape():
    import torch
    net = _model(5)
    out = net(torch.zeros(7, 9, 5), torch.ones(7, 9))
    assert out.shape == (7,)
