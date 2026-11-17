"""
Graph statistics as tree-model features (replaces the standalone GNN classifier).

Evidence from industry and research: a GBDT fed with graph statistics matches or
beats an end-to-end GNN on tabular fraud data; GNN outputs help most as extra
features. These are the cheap, robust statistics: how many distinct clients share
an entity (device, card+address, address+email…) and how many entities a client
spreads over. Missing keys produce NaN, never a shared "unknown" hub node.

Neighbour fraud rates (the label side of the graph) live in label_history.py.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

from src.feature_engineering.entity_ops import make_codes, entity_nunique

# (entity whose neighbourhood we measure, neighbour type counted)
SHARED_ENTITY_PAIRS = [
    ("device_fp", "uid1"),       # clients per device fingerprint
    ("card1_addr1", "uid1"),     # distinct cards (by start-date) on same card BIN+address
    ("addr1", "uid1"),
    ("P_emaildomain", "addr1"),  # address spread of an email domain
    ("uid1", "device_fp"),       # devices per client
    ("uid1", "card2_dist1"),
    ("card1", "addr1"),
]


def add_graph_features(df: pd.DataFrame, t: np.ndarray, mode: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for ent, nb in SHARED_ENTITY_PAIRS:
        if ent not in df.columns or nb not in df.columns:
            continue
        out[f"{ent}_n_{nb}"] = entity_nunique(
            make_codes(df, ent), t, make_codes(df, nb), mode).astype(np.float32)
    logger.info("Graph features ({}): {} columns.", mode, len(out))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4 — network / fraud-ring features (replaces the GNN)
# Production practice: Adyen ShopperDNA & Stripe shared-entity counts, Booking.com
# count_fraud / neighbour exposure, Meta DEC neighbour aggregates, PayPal
# connected components. All causal; labels only after the delay.
# ═════════════════════════════════════════════════════════════════════════════

def add_network_v4(df: pd.DataFrame, t: np.ndarray, events, smoothing: float, prior_rate: float,
                   hub_cap: int = 20) -> Dict[str, np.ndarray]:
    """``events``: LabelEvents from the shared label policy (src.labels.policy.label_ready_times)."""
    from src.feature_engineering.entity_ops import (
        window_count_sum, first_time_pair, policy_label_stats, smoothed_rate,
    )
    from src.feature_engineering.streaming_ops import window_distinct, identity_clusters_policy

    out: Dict[str, np.ndarray] = {}
    uid = make_codes(df, "uid1")

    for ent in [c for c in ["device_fp", "card1_addr1"] if c in df.columns]:
        ec = make_codes(df, ent)
        # 4a. windowed shared-entity counts (exact sliding windows)
        for w, tag in [(86400, "1d"), (7 * 86400, "7d"), (30 * 86400, "30d")]:
            out[f"{ent}_clients_{tag}"] = window_distinct(ec, t, uid, w).astype(np.float32)
        # link novelty: is this (entity, client) pair new; pair & entity age; new-link velocity
        raw_pair = np.where((ec >= 0) & (uid >= 0), ec * (uid.max() + 1) + uid, -1)
        pair, _ = pd.factorize(raw_pair)                 # compact codes (sort-key safe)
        pair = np.where(raw_pair < 0, -1, pair).astype(np.int64)
        is_new = first_time_pair(ec, t, uid)            # NaN when the entity has no history
        out[f"{ent}_pair_is_new"] = is_new.astype(np.float32)
        # pair age = now - first time the pair was seen  (min over prior = first, rows time-sorted)
        from src.feature_engineering.entity_ops import _Sorted
        sp = _Sorted(pair, t)
        out[f"{ent}_pair_age_days"] = ((t - sp.to_orig(sp.t[sp.start])) / 86400.0).astype(np.float32)
        se = _Sorted(ec, t)
        out[f"{ent}_age_days"] = ((t - se.to_orig(se.t[se.start])) / 86400.0).astype(np.float32)
        newflag = np.nan_to_num(is_new, nan=0.0)
        out[f"{ent}_new_links_1d"] = window_count_sum(ec, t, 86400, values=newflag, s=se)["sum"].astype(np.float32)

        # 4b. fraud exposure through the shared entity, EXCLUDING the client's own history
        ent_lbl = policy_label_stats(ec, t, events)
        own_lbl = policy_label_stats(pair, t, events)
        other_n = ent_lbl["n"] - np.nan_to_num(own_lbl["n"], nan=0.0)
        other_f = ent_lbl["fraud"] - np.nan_to_num(own_lbl["fraud"], nan=0.0)
        other_fm = ent_lbl["fraud_mature"] - np.nan_to_num(own_lbl["fraud_mature"], nan=0.0)
        out[f"{ent}_other_clients_lbl_fraud"] = other_f.astype(np.float32)
        out[f"{ent}_other_clients_lbl_rate"] = smoothed_rate(other_fm, other_n, prior_rate, smoothing).astype(np.float32)

    # 4c. incremental identity clusters: clients linked through shared devices (hub-capped)
    if "device_fp" in df.columns:
        dev = make_codes(df, "device_fp")
        newp = first_time_pair(dev, t, uid)
        newp = np.where(np.isnan(newp), np.where(dev >= 0, 1.0, 0.0), newp)   # first row of a device is a new pair
        cl = identity_clusters_policy(t, uid, dev, newp, events, hub_cap)
        out["cluster_n_clients"] = cl["size"].astype(np.float32)
        out["cluster_lbl_fraud"] = cl["fraud"].astype(np.float32)
        out["cluster_lbl_rate"] = smoothed_rate(cl["fraud_mature"], cl["labelled"], prior_rate, smoothing).astype(np.float32)
        out["cluster_merges_identities"] = cl["merge"].astype(np.float32)

    # 4d. ring signal: other clients sharing BOTH the device and the card+address
    if "device_fp" in df.columns and "card1_addr1" in df.columns:
        both = make_codes(df, ["device_fp", "card1_addr1"])
        out["ring_clients_sharing_device_and_card"] = (entity_nunique(both, t, uid, "causal") - 1).astype(np.float32)
    logger.info("Network v4 features: {} columns.", len(out))
    return out
