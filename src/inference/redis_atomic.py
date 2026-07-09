from __future__ import annotations

import json
from typing import Any

import numpy as np
import redis as redis_lib
from loguru import logger

# ── Lua script: atomic push + trim + return full window ───────────────────────
# KEYS[1]  — the Redis LIST key (e.g. "user:uid123:seq_window")
# ARGV[1]  — serialised new feature point (JSON string)
# ARGV[2]  — maximum window length (e.g. "10")
_LUA_ATOMIC_SEQ_UPDATE = """
local key     = KEYS[1]
local point   = ARGV[1]
local max_len = tonumber(ARGV[2])

-- 1. Push the new point to the right end of the list
redis.call('RPUSH', key, point)

-- 2. Trim to keep only the last max_len entries (oldest fall off left)
redis.call('LTRIM', key, -max_len, -1)

-- 3. Return the full trimmed window in one shot (no second round-trip)
return redis.call('LRANGE', key, 0, -1)
"""


class RedisAtomicOps:
    """
    Wraps atomic Redis Lua operations for the sequence window.
    Instantiate once per service and reuse across requests.
    """

    def __init__(self, r: redis_lib.Redis):
        self._r      = r
        # Pre-register the script for efficiency (SHA cached by Redis server)
        self._script = r.register_script(_LUA_ATOMIC_SEQ_UPDATE)
        logger.debug("RedisAtomicOps: Lua script registered.")

    # ── Sequence window operations ─────────────────────────────────────────────

    def push_and_get_window(
        self,
        key:       str,
        new_point: np.ndarray,
        max_len:   int = 10,
    ) -> list[np.ndarray]:
        """
        Atomically push a new feature vector, trim the window, and return it.

        Parameters
        ----------
        key       : Redis LIST key  (e.g. "user:uid123:seq_window")
        new_point : 1-D float32 numpy array (length = n_features)
        max_len   : Maximum window length (must match SEQ_LEN)

        Returns
        -------
        window : list of numpy arrays, ordered oldest → newest (length ≤ max_len)
        """
        serialised = json.dumps(new_point.tolist())
        raw_window = self._script(keys=[key], args=[serialised, str(max_len)])
        return [np.array(json.loads(item), dtype=np.float32) for item in raw_window]

    def get_window(self, key: str, max_len: int = 10) -> list[np.ndarray]:
        """
        Read the current sequence window WITHOUT pushing a new point.
        Used for read-only inspection or for the cold-start phantom load.
        """
        raw = self._r.lrange(key, 0, -1)
        items = [np.array(json.loads(item), dtype=np.float32) for item in raw]
        return items[-max_len:]   # safety clip

    def set_window(self, key: str, vectors: list[np.ndarray], ttl: int = 30 * 86400) -> None:
        """
        Overwrite a user's sequence window with a list of feature vectors.
        Used by cold_start.py to initialise phantom steps.
        """
        pipe = self._r.pipeline()
        pipe.delete(key)
        for vec in vectors:
            pipe.rpush(key, json.dumps(vec.tolist()))
        pipe.expire(key, ttl)
        pipe.execute()

    # ── Ring detector (real-time, no GNN required) ────────────────────────────

    def fast_ring_check(
        self,
        uid:        str,
        device_id:  str,
        email_domain: str,
        window_secs: int = 300,   # 5-minute ring detection window
        device_threshold:  int = 5,
        email_threshold:   int = 10,
    ) -> dict:
        """
        Detect fraud rings by counting how many different users share the
        same device or email domain in the last `window_secs` seconds.

        Runs entirely in Redis in <1ms — no GNN refresh required.
        Fills the 1-5 minute staleness gap between GNN async updates.

        Parameters
        ----------
        uid            : Current user ID
        device_id      : Device fingerprint from the transaction
        email_domain   : Email domain of the user (e.g. "gmail.com")
        window_secs    : TTL of the shared-entity sets (default: 5 min)
        device_threshold : Trigger if ≥ N unique users share device in window
        email_threshold  : Trigger if ≥ N unique users share email domain

        Returns
        -------
        dict with ring_signal (0/1), shared_device_users, shared_email_users
        """
        device_key = f"ring:device:{device_id}:uids"
        email_key  = f"ring:email:{email_domain}:uids"

        pipe = self._r.pipeline()
        pipe.sadd(device_key, uid)
        pipe.expire(device_key, window_secs)
        pipe.sadd(email_key,  uid)
        pipe.expire(email_key,  window_secs)
        pipe.scard(device_key)
        pipe.scard(email_key)
        results = pipe.execute()

        shared_device_users = int(results[4])
        shared_email_users  = int(results[5])
        ring_signal = int(
            shared_device_users >= device_threshold or
            shared_email_users  >= email_threshold
        )

        if ring_signal:
            logger.warning(
                "Ring signal detected! uid={} device_users={} email_users={}",
                uid, shared_device_users, shared_email_users,
            )

        return {
            "ring_signal":          ring_signal,
            "shared_device_users":  shared_device_users,
            "shared_email_users":   shared_email_users,
        }
