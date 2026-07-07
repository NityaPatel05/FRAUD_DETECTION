from __future__ import annotations

import bisect
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────
# Offline batch: rolling window velocity via pandas
# ─────────────────────────────────────────────────────────

def build_velocity_features_offline(
    df: pd.DataFrame,
    group_col: str = "uid1",
    time_col: str = "TransactionDT",
    windows: tuple = (300, 900, 3600, 86400),   # 5m, 15m, 1h, 24h in seconds
    uid_history: Optional[dict] = None,
    return_history: bool = False,
):
    """
    For each transaction, compute backward-looking velocity features:
      - uid_tx_{w}:   count of transactions by same uid in last w seconds
      - uid_amt_{w}:  sum of amounts by same uid in last w seconds
      - uid_gap_last: seconds since uid's previous transaction
      - uid_avg_gap:  average gap between uid's transactions (up to this point)

    IMPORTANT: df must be sorted by TransactionDT ascending before calling.
    Computes only on rows strictly before the current timestamp (no leakage).

    This is a Python-loop implementation — for production use the Redis path.
    For training on the full ~590k row dataset this takes ~5-10 minutes.
    """
    logger.info(
        "Building offline velocity features (group={}, windows={})…",
        group_col, windows,
    )

    n = len(df)
    times  = df[time_col].values
    groups = df[group_col].values
    amts   = df["TransactionAmt"].values

    # Pre-allocate output arrays
    gap_last = np.full(n, -1.0)
    avg_gap  = np.full(n, -1.0)
    tx_counts  = {w: np.zeros(n, dtype=np.int32) for w in windows}
    amt_sums   = {w: np.zeros(n, dtype=np.float64) for w in windows}

    # Index: uid → list of (timestamp, amount) in sorted order
    if uid_history is None:
        uid_history = {}

    for i in range(n):
        g   = groups[i]
        t   = times[i]
        amt = amts[i]

        if g not in uid_history:
            uid_history[g] = []

        hist = uid_history[g]

        if hist:
            prev_t = hist[-1][0]
            gap_last[i] = t - prev_t

            if len(hist) > 1:
                # O(1) calculation: the sum of adjacent gaps is just the total time difference 
                # between the first and last transaction.
                avg_gap[i] = (hist[-1][0] - hist[0][0]) / (len(hist) - 1)

        # Window counts and sums using bisect for O(log N) cutoff lookup
        hist_times = [ht for ht, _ in hist]
        hist_amts  = [ha for _, ha in hist]
        for w in windows:
            cutoff = t - w
            # bisect_left finds first index where time >= cutoff
            lo = bisect.bisect_left(hist_times, cutoff)
            tx_counts[w][i] = len(hist_times) - lo
            amt_sums[w][i]  = sum(hist_amts[lo:])

        # Append current transaction to history
        hist.append((t, amt))

    # Write back to df
    for w in windows:
        label = f"{w//3600}h" if w % 3600 == 0 else f"{w}s"
        df[f"uid_tx_{label}"]  = tx_counts[w]
        df[f"uid_amt_{label}"] = amt_sums[w]

    df["uid_gap_last"] = gap_last
    df["uid_avg_gap"]  = avg_gap

    # Rename 24h amount column to match feature config alias
    if "uid_amt_24h" in df.columns:
        df.rename(columns={"uid_amt_24h": "uid_total_amt_24h"}, inplace=True)

    logger.info("Velocity features added: gap_last, avg_gap + {} window features.", len(windows) * 2)
    if return_history:
        return df, uid_history
    return df


# ─────────────────────────────────────────────────────────
# Online / inference: Redis sliding window
# ─────────────────────────────────────────────────────────

def get_velocity_from_redis(
    r,          # redis.Redis connection
    uid: str,
    current_dt: int,
    windows: tuple = (300, 900, 3600, 86400),
) -> dict:
    """
    Pull velocity features from Redis for a single uid at inference time.
    Latency: 50-200ms.

    Returns dict of velocity feature values.
    """
    key = f"uid:{uid}"
    features = {}

    try:
        for w in windows:
            cutoff = current_dt - w
            label  = f"{w//3600}h" if w % 3600 == 0 else f"{w}s"

            # Count transactions in window
            count = r.zcount(f"{key}:txn_times", cutoff, current_dt)
            features[f"uid_tx_{label}"] = int(count)

            # Sum of amounts in window (scores are timestamps, members are "ts:amt")
            members = r.zrangebyscore(f"{key}:txn_amts", cutoff, current_dt)
            amt_sum = sum(float(m.decode().split(":")[1]) for m in members)
            features[f"uid_amt_{label}"] = amt_sum

        # Gap since last transaction
        latest = r.zrevrange(f"{key}:txn_times", 0, 0, withscores=True)
        if latest:
            last_ts = int(latest[0][1])
            features["uid_gap_last"] = current_dt - last_ts
            # Avg gap
            all_ts = r.zrange(f"{key}:txn_times", 0, -1, withscores=True)
            if len(all_ts) >= 2:
                ts_vals = sorted([int(s) for _, s in all_ts])
                gaps = [ts_vals[i] - ts_vals[i-1] for i in range(1, len(ts_vals))]
                features["uid_avg_gap"] = np.mean(gaps)
            else:
                features["uid_avg_gap"] = -1
        else:
            features["uid_gap_last"] = -1
            features["uid_avg_gap"]  = -1

    except Exception as e:
        logger.warning("Redis velocity fetch failed for uid={}: {}", uid, e)
        # Return -1 (missing signal) for all velocity features
        for w in windows:
            label = f"{w//3600}h" if w % 3600 == 0 else f"{w}s"
            features[f"uid_tx_{label}"]  = -1
            features[f"uid_amt_{label}"] = -1
        features["uid_gap_last"] = -1
        features["uid_avg_gap"]  = -1

    return features


def update_redis_after_transaction(
    r,
    uid: str,
    transaction_amt: float,
    transaction_dt: int,
    ttl_seconds: int = 30 * 86400,  # 30 days
) -> None:
    """
    Update Redis feature store after a transaction is processed.
    Called AFTER inference — not before (would leak).

    Pipeline for atomicity.
    """
    key = f"uid:{uid}"
    pipe = r.pipeline()

    # Sorted sets: score = timestamp (for range queries)
    pipe.zadd(f"{key}:txn_times", {str(transaction_dt): transaction_dt})
    pipe.zadd(f"{key}:txn_amts",  {f"{transaction_dt}:{transaction_amt}": transaction_dt})

    # Expire old entries beyond 30 days
    cutoff = transaction_dt - ttl_seconds
    pipe.zremrangebyscore(f"{key}:txn_times", 0, cutoff)
    pipe.zremrangebyscore(f"{key}:txn_amts",  0, cutoff)

    # TTL on the whole key
    pipe.expire(f"{key}:txn_times", ttl_seconds)
    pipe.expire(f"{key}:txn_amts",  ttl_seconds)

    pipe.execute()
