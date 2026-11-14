"""
Phase 3e — Hidden-Markov-Model sequence features (Lucas et al., 2019/2020).

Lucas et al. fitted small HMMs on short windows of each card's / terminal's recent
transactions and fed the window log-likelihoods to a random forest: +9–18% PR-AUC
over aggregate features on real Belgian card data. This is the light-weight,
evidence-backed way to capture "sequence" signal that the BiLSTM was meant to.

Design choices for leakage safety:
  * HMMs are UNSUPERVISED (labels never used) and fitted on TRAIN-period rows only.
    Lucas also trained label-specific (genuine vs fraud) HMMs; those need a fit
    period disjoint from the classifier's training rows, so they are left out here.
  * Each window is causal: the current transaction + its 2 predecessors.
  * Feature = "surprise" = -[log p(window) - log p(window without current row)],
    i.e. -log p(current | recent history). High = unusual for this client.

Fitting (Baum-Welch) and scoring (forward algorithm) are vectorised over all windows.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

from src.feature_engineering.entity_ops import make_codes
from src.feature_engineering.streaming_ops import lag_values


def _windows(codes, t, x, k=3):
    """Matrix [n, k] = (x_{-(k-1)}, …, x_{-1}, x_0) for each row (NaN when missing)."""
    cols = [lag_values(codes, t, x, j) for j in range(k - 1, 0, -1)]
    cols.append(np.where(codes >= 0, x, np.nan))
    return np.column_stack(cols)


def _log_gauss(x, means, vars_):
    # x: [n], means/vars: [K] → [n, K]
    return -0.5 * (np.log(2 * np.pi * vars_)[None, :] + (x[:, None] - means[None, :]) ** 2 / vars_[None, :])


def _forward_loglik(W, startprob, transmat, means, vars_):
    """log p(window) for each row, skipping leading NaNs (shorter history)."""
    n, k = W.shape
    log_alpha = np.full((n, len(startprob)), np.nan)
    started = np.zeros(n, dtype=bool)
    logA = np.log(transmat + 1e-300)
    for j in range(k):
        x = W[:, j]
        ok = ~np.isnan(x)
        logB = _log_gauss(np.nan_to_num(x), means, vars_)
        new_start = ok & ~started
        cont = ok & started
        if new_start.any():
            log_alpha[new_start] = np.log(startprob + 1e-300)[None, :] + logB[new_start]
        if cont.any():
            prev = log_alpha[cont]                              # [m, K]
            m = prev.max(axis=1, keepdims=True)
            trans = np.log(np.exp(prev - m) @ np.exp(logA)) + m  # log-sum-exp over previous state
            log_alpha[cont] = trans + logB[cont]
        started |= ok
    import warnings
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)          # rows with no observation yet
        mx = np.nanmax(log_alpha, axis=1)
        ll = np.log(np.nansum(np.exp(log_alpha - mx[:, None]), axis=1)) + mx
    ll[~started] = np.nan
    return ll


def _lse(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return (np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m).squeeze(axis)


def _fit_hmm(W_train, n_states, seed, max_windows=200_000, n_iter=50, tol=1e-4, min_var=1e-3):
    """Vectorised Baum-Welch for many fixed-length 1-D windows (all windows at once).

    hmmlearn loops over sequences in Python — far too slow for ~200k windows of
    length 3 — so EM is written directly in numpy here.
    """
    X = W_train[~np.isnan(W_train).any(axis=1)]
    rng = np.random.default_rng(seed)
    if len(X) > max_windows:
        X = X[rng.choice(len(X), max_windows, replace=False)]
    N, L = X.shape
    K = n_states
    flat = X.ravel()
    means = np.quantile(flat, (np.arange(K) + 0.5) / K)
    vars_ = np.full(K, max(flat.var(), min_var))
    start = np.full(K, 1.0 / K)
    trans = np.full((K, K), 0.5 / (K - 1)); np.fill_diagonal(trans, 0.5)
    prev_ll = -np.inf
    for _ in range(n_iter):
        logB = np.stack([_log_gauss(X[:, l], means, vars_) for l in range(L)], axis=1)   # [N,L,K]
        logA, logpi = np.log(trans + 1e-300), np.log(start + 1e-300)
        la = np.empty((N, L, K)); lb = np.zeros((N, L, K))
        la[:, 0] = logpi + logB[:, 0]
        for l in range(1, L):
            la[:, l] = _lse(la[:, l - 1, :, None] + logA[None], axis=1) + logB[:, l]
        for l in range(L - 2, -1, -1):
            lb[:, l] = _lse(logA[None] + (logB[:, l + 1] + lb[:, l + 1])[:, None, :], axis=2)
        ll = _lse(la[:, -1], axis=1)                                                  # [N]
        gamma = np.exp(la + lb - ll[:, None, None])                                   # [N,L,K]
        xi = np.exp(la[:, :-1, :, None] + logA[None, None] + (logB[:, 1:] + lb[:, 1:])[:, :, None, :]
                    - ll[:, None, None, None])                                         # [N,L-1,K,K]
        start = gamma[:, 0].mean(axis=0)
        trans = xi.sum(axis=(0, 1)) / np.maximum(gamma[:, :-1].sum(axis=(0, 1))[:, None], 1e-12)
        trans /= trans.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=(0, 1))
        means = (gamma * X[:, :, None]).sum(axis=(0, 1)) / np.maximum(w, 1e-12)
        vars_ = (gamma * (X[:, :, None] - means) ** 2).sum(axis=(0, 1)) / np.maximum(w, 1e-12) + min_var
        cur = ll.mean()
        if abs(cur - prev_ll) < tol:
            break
        prev_ll = cur
    return start, trans, means, vars_


HMM_PERSPECTIVES = [("uid1", "client"), ("card1_addr1", "cardaddr")]


def hmm_surprise_from_window(W: np.ndarray, params) -> tuple:
    """(surprise, window_ll) for windows W [n, 3] — shared by offline and online paths."""
    ll_full = _forward_loglik(W, *params)
    ll_prev = _forward_loglik(W[:, :-1], *params)
    surprise = -(ll_full - np.nan_to_num(ll_prev, nan=0.0))
    surprise[np.isnan(W[:, -1])] = np.nan
    return surprise, ll_full


def add_hmm_features(df: pd.DataFrame, t: np.ndarray, is_train: np.ndarray,
                     n_states: int = 5, seed: int = 42, params_out: dict | None = None) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    amt = np.log1p(df["TransactionAmt"].to_numpy(dtype=np.float64))
    for key, tag in HMM_PERSPECTIVES:
        if key not in df.columns:
            continue
        codes = make_codes(df, key)
        prev_t = lag_values(codes, t, t.astype(np.float64), 1)
        gap = np.log1p((t - prev_t) / 3600.0)                   # log hours since previous
        for obs_name, x in [("amt", amt), ("gap", gap)]:
            W = _windows(codes, t, x, k=3)
            params = _fit_hmm(W[is_train], n_states, seed)
            if params_out is not None:                             # persisted for online serving
                params_out[f"{tag}_{obs_name}"] = [np.asarray(a).tolist() for a in params]
            surprise, ll_full = hmm_surprise_from_window(W, params)  # -log p(current | history)
            out[f"hmm_{tag}_{obs_name}_surprise"] = surprise.astype(np.float32)
            out[f"hmm_{tag}_{obs_name}_window_ll"] = ll_full.astype(np.float32)
    logger.info("HMM sequence features: {} columns (unsupervised, fit on train rows).", len(out))
    return out
