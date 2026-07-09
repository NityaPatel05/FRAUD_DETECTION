from __future__ import annotations

import json
import pickle
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import redis as redis_lib
from loguru import logger

# ── GNN imports (guarded so the file can be imported even without PyG) ──────
try:
    from src.models.gnn.model import HeteroSAGE
    from src.models.gnn.graph_builder import build_graph_data
    _TORCH_GEO_AVAILABLE = True
except ImportError:
    _TORCH_GEO_AVAILABLE = False
    logger.warning(
        "torch-geometric not installed — GNN refresh job will be a no-op. "
        "Install with: pip install torch-geometric"
    )

# Cypher query: pull every Card→Device USED_ON edge from the last `cutoff` seconds.
# All fields map 1-to-1 onto the columns expected by build_graph_data().
_CYPHER_PULL_GRAPH = """
MATCH (c:Card)-[t:USED_ON]->(d:Device)
WHERE t.timestamp > $cutoff
OPTIONAL MATCH (c)-[:LINKED_TO]->(e:Email)
OPTIONAL MATCH (c)-[:BILLED_TO]->(a:Address)
RETURN
    c.card_id       AS card1,
    d.device_id     AS DeviceInfo,
    e.email_domain  AS P_emaildomain,
    a.addr_id       AS addr1,
    t.amt           AS TransactionAmt,
    t.hour_sin      AS hour_sin,
    t.hour_cos      AS hour_cos,
    t.time_delta    AS uid_gap_last,
    t.is_fraud      AS isFraud,
    t.timestamp     AS TransactionDT,
    t.ProductCD     AS ProductCD,
    t.ring_signal   AS ring_signal
"""

# TTL for Redis gnn:{card_id} keys — 30 minutes is enough since job runs every 1-5 min
_GNN_REDIS_TTL = 1800  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Pull graph data from Neo4j
# ─────────────────────────────────────────────────────────────────────────────

def pull_graph_from_neo4j(
    driver,
    lookback_seconds: int = 86400,
) -> pd.DataFrame:
    """
    Execute the Cypher query and return results as a pandas DataFrame.

    Parameters
    ----------
    driver          : neo4j.GraphDatabase.driver instance
    lookback_seconds: How far back to look. Default = 24 h.

    Returns
    -------
    df : DataFrame with columns matching graph_builder expectations.
         Returns an empty DataFrame on error.
    """
    import time as _time
    cutoff = int(_time.time()) - lookback_seconds

    try:
        with driver.session() as session:
            result = session.run(_CYPHER_PULL_GRAPH, cutoff=cutoff)
            records = [dict(r) for r in result]

        if not records:
            logger.warning("GNN refresh: Neo4j returned 0 records for cutoff={}.", cutoff)
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # ── Derived features needed by graph_builder ─────────────────────────
        # log-amount
        df["TransactionAmt_log"] = np.log1p(
            pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0)
        ).astype(np.float32)

        # amount z-score (global, across this batch)
        amt = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0)
        mu, sigma = amt.mean(), amt.std() + 1e-8
        df["amt_zscore"] = ((amt - mu) / sigma).astype(np.float32)

        # hour features (fill if missing from Neo4j)
        df["hour_sin"]    = pd.to_numeric(df.get("hour_sin",    0), errors="coerce").fillna(0).astype(np.float32)
        df["hour_cos"]    = pd.to_numeric(df.get("hour_cos",    1), errors="coerce").fillna(1).astype(np.float32)
        df["is_late_night"] = (
            (df["hour_sin"] < -0.5) & (df["hour_cos"] < 0)
        ).astype(np.float32)

        # uid_gap_last from Neo4j time_delta field
        df["uid_gap_last"] = pd.to_numeric(df.get("uid_gap_last", 86400), errors="coerce").fillna(86400).astype(np.float32)

        # isFraud — may be None for live transactions not yet labelled
        df["isFraud"] = pd.to_numeric(df.get("isFraud", 0), errors="coerce").fillna(0).astype(np.int8)

        # ProductCD encoding (ordinal)
        _pcd_map = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
        df["ProductCD_encoded"] = (
            df["ProductCD"].map(_pcd_map).fillna(0).astype(np.float32)
        )

        # Frequency maps for card/addr/email/device (simple count within batch)
        for col in ["card1", "addr1", "P_emaildomain", "DeviceInfo"]:
            freq = df[col].fillna("unknown").astype(str).value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].fillna("unknown").astype(str).map(freq).astype(np.float32)

        # Placeholder aggregation columns (graph_builder will zero if missing)
        for col in [
            "card1_amt_mean", "card1_amt_std", "card1_tx_count",
            "C1_uid_mean", "D1_norm_uid_mean", "M_match_count",
            "addr1_tx_count", "addr1_amt_mean",
            "email_trusted", "P_email_is_free",
            "device_os_encoded", "device_browser_encoded", "device_mobile",
        ]:
            if col not in df.columns:
                df[col] = 0.0

        logger.info(
            "GNN refresh: pulled {} transactions from Neo4j (cutoff={}s ago).",
            len(df), lookback_seconds,
        )
        return df

    except Exception as exc:
        logger.error("GNN refresh: Neo4j pull failed — {}", exc)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build PyG HeteroData
# ─────────────────────────────────────────────────────────────────────────────

def build_hetero_data(df: pd.DataFrame):
    """
    Thin wrapper around graph_builder.build_graph_data().
    Returns None if PyG is not installed or df is empty.
    """
    if not _TORCH_GEO_AVAILABLE or df.empty:
        return None
    return build_graph_data(df)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Run GraphSAGE, extract card embeddings
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_graphsage(
    model: "HeteroSAGE",
    data,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Full-graph forward pass using model.get_embeddings().
    Returns dict of {node_type: (n_nodes, embed_dim) float32 array}.
    Only the 'card' embeddings are written to Redis, but all are returned
    for optional ring-detection on device/email nodes.
    """
    model.eval()
    data = data.to(device)
    emb_dict = model.get_embeddings(data.x_dict, data.edge_index_dict)
    return {k: v.cpu().numpy() for k, v in emb_dict.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Write embeddings to Redis
# ─────────────────────────────────────────────────────────────────────────────

def write_embeddings_to_redis(
    r:         redis_lib.Redis,
    card_embs: np.ndarray,             # (n_cards, 64)
    card_map:  Dict[str, int],         # {card1_value: node_index}
    ttl:       int = _GNN_REDIS_TTL,
) -> int:
    """
    Write each card's 64-dim embedding to Redis under key gnn:{card_id}.
    Key format matches FeatureStore.get_gnn_features() and set_gnn_features().

    Returns the number of keys written.
    """
    idx_to_card = {v: k for k, v in card_map.items()}
    pipe        = r.pipeline(transaction=False)  # pipelining without MULTI/EXEC for throughput
    n_written   = 0

    for idx in range(len(card_embs)):
        card_val = idx_to_card.get(idx)
        if card_val is None:
            continue
        key  = f"gnn:{card_val}"
        data = json.dumps({
            "node_embedding": card_embs[idx].tolist(),
            # fraud_prob and ring_risk_score stay from the previous write
            # until a new full GNN training cycle overrides them.
        })
        pipe.setex(key, ttl, data)
        n_written += 1

    pipe.execute()
    logger.info("GNN refresh: wrote {} card embeddings to Redis (TTL={}s).", n_written, ttl)
    return n_written


# ─────────────────────────────────────────────────────────────────────────────
# Step 4b — Persist card_map for inference-time lookup
# ─────────────────────────────────────────────────────────────────────────────

def save_card_map(card_map: Dict[str, int], path: str) -> None:
    """Overwrite the on-disk card_map so FraudDetectionPredictor.load_all()
    picks up new card IDs on the next restart."""
    with open(path, "wb") as f:
        pickle.dump(card_map, f)
    logger.debug("GNN refresh: card_map saved to {}.", path)


# ─────────────────────────────────────────────────────────────────────────────
# Main refresh loop
# ─────────────────────────────────────────────────────────────────────────────

def gnn_refresh_once(
    model:         "HeteroSAGE",
    r:             redis_lib.Redis,
    driver,
    device:        torch.device,
    lookback_secs: int = 86400,
    card_map_path: Optional[str] = None,
) -> bool:
    """
    Execute one full GNN refresh cycle.
    Returns True on success, False on any non-fatal error.
    """
    t0 = time.time()

    # Step 1: pull
    df = pull_graph_from_neo4j(driver, lookback_seconds=lookback_secs)
    if df.empty:
        logger.warning("GNN refresh: empty graph — skipping this cycle.")
        return False

    # Step 2: build graph
    data = build_hetero_data(df)
    if data is None:
        return False

    # Step 3: infer
    try:
        embs = run_graphsage(model, data, device)
    except Exception as exc:
        logger.error("GNN refresh: GraphSAGE inference failed — {}", exc)
        return False

    card_embs = embs.get("card")
    card_map  = data.card_map

    if card_embs is None:
        logger.error("GNN refresh: no 'card' embeddings returned.")
        return False

    # Step 4: write to Redis
    write_embeddings_to_redis(r, card_embs, card_map)

    # Optional: persist updated card_map for next predictor restart
    if card_map_path:
        save_card_map(card_map, card_map_path)

    elapsed = time.time() - t0
    logger.info("GNN refresh: cycle complete in {:.2f}s.", elapsed)
    return True


def gnn_refresh_loop(
    model:         "HeteroSAGE",
    r:             redis_lib.Redis,
    driver,
    device:        Optional[torch.device] = None,
    interval_secs: int = 60,
    lookback_secs: int = 86400,
    card_map_path: Optional[str] = None,
) -> None:
    """
    Blocking loop: run gnn_refresh_once() every `interval_secs` seconds.
    Designed to run in a daemon thread or separate process.

    Parameters
    ----------
    model          : Loaded HeteroSAGE model (call model.eval() before passing).
    r              : Redis client instance.
    driver         : neo4j.GraphDatabase.driver instance.
    device         : torch.device (defaults to CUDA if available, else CPU).
    interval_secs  : Sleep between refresh cycles (default 60 s → 1 minute).
    lookback_secs  : How far back to pull from Neo4j (default 86 400 s → 24 h).
    card_map_path  : Optional path to overwrite gnn_card_map.pkl after each cycle.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "GNN refresh loop started — interval={}s, device={}, lookback={}s.",
        interval_secs, device, lookback_secs,
    )

    while True:
        try:
            gnn_refresh_once(
                model=model,
                r=r,
                driver=driver,
                device=device,
                lookback_secs=lookback_secs,
                card_map_path=card_map_path,
            )
        except Exception as exc:
            # Catch-all so the loop never dies from an unexpected error
            logger.error("GNN refresh: unhandled exception — {}. Continuing.", exc)

        logger.debug("GNN refresh: sleeping {}s until next cycle.", interval_secs)
        time.sleep(interval_secs)


def start_gnn_refresh_thread(
    model:         "HeteroSAGE",
    r:             redis_lib.Redis,
    driver,
    interval_secs: int = 60,
    lookback_secs: int = 86400,
    device:        Optional[torch.device] = None,
    card_map_path: Optional[str] = None,
) -> threading.Thread:
    """
    Convenience wrapper: starts gnn_refresh_loop() in a daemon thread.
    Call once at service startup. The thread exits automatically when
    the main process exits.

    Returns the Thread object (already started).

    Example
    -------
    >>> from src.jobs.gnn_refresh import start_gnn_refresh_thread
    >>> t = start_gnn_refresh_thread(model, redis_client, neo4j_driver, interval_secs=60)
    """
    t = threading.Thread(
        target=gnn_refresh_loop,
        kwargs=dict(
            model=model,
            r=r,
            driver=driver,
            device=device,
            interval_secs=interval_secs,
            lookback_secs=lookback_secs,
            card_map_path=card_map_path,
        ),
        name="gnn-refresh",
        daemon=True,
    )
    t.start()
    logger.info("GNN refresh daemon thread started (tid={}).", t.ident)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint: python -m src.jobs.gnn_refresh
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pickle
    import yaml
    from neo4j import GraphDatabase  # type: ignore

    parser = argparse.ArgumentParser(description="GNN embedding refresh job")
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--interval",     type=int, default=60,    help="Refresh interval (seconds)")
    parser.add_argument("--lookback",     type=int, default=86400, help="Neo4j lookback window (seconds)")
    parser.add_argument("--model-path",   default="artifacts/checkpoints/gnn_fold0.pt")
    parser.add_argument("--card-map",     default="artifacts/checkpoints/gnn_card_map.pkl")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Neo4j connection ─────────────────────────────────────────────────────
    neo4j_cfg = cfg.get("neo4j", {})
    driver = GraphDatabase.driver(
        neo4j_cfg.get("uri",      "bolt://localhost:7687"),
        auth=(
            neo4j_cfg.get("user",     "neo4j"),
            neo4j_cfg.get("password", "password"),
        ),
    )

    # ── Redis connection ─────────────────────────────────────────────────────
    r_cfg = cfg.get("redis", {})
    r = redis_lib.Redis(
        host=r_cfg.get("host", "localhost"),
        port=r_cfg.get("port", 6379),
        db=r_cfg.get("db", 0),
        decode_responses=False,
    )

    # ── Load model ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We need graph metadata to instantiate the model — load from a dummy graph
    # built from the card_map shapes. In practice, load the saved metadata instead.
    # Here we reconstruct from the latest card_map.
    with open(args.card_map, "rb") as f:
        saved_card_map = pickle.load(f)

    # Load state dict only — metadata must match the saved training metadata.
    # A real deployment would save and restore model + metadata together.
    state_dict = torch.load(args.model_path, map_location=device)

    # ── Infer in_dims from state dict ────────────────────────────────────────
    # projection weight shapes: projection.projections.{ntype}.0.weight → (128, in_dim)
    in_dims: Dict[str, int] = {}
    for k, v in state_dict.items():
        if "projection.projections." in k and k.endswith(".0.weight"):
            ntype = k.split("projection.projections.")[1].split(".")[0]
            in_dims[ntype] = v.shape[1]  # Linear: (out, in)

    if not in_dims:
        logger.error(
            "Could not infer in_dims from state dict keys. "
            "Save model metadata alongside the checkpoint."
        )
        raise SystemExit(1)

    # Build model skeleton — metadata must match training-time schema.
    # For the standalone runner, we reconstruct with known node/edge types.
    from torch_geometric.data import HeteroData  # type: ignore
    dummy_meta = (
        list(in_dims.keys()),
        [
            ("transaction", "uses_card",   "card"),
            ("transaction", "from_addr",   "addr"),
            ("transaction", "uses_email",  "email"),
            ("transaction", "uses_device", "device"),
            ("card",        "rev_uses_card",   "transaction"),
            ("addr",        "rev_from_addr",   "transaction"),
            ("email",       "rev_uses_email",  "transaction"),
            ("device",      "rev_uses_device", "transaction"),
        ],
    )
    model = HeteroSAGE(
        in_dims=in_dims,
        proj_dim=cfg.get("gnn", {}).get("proj_dim", 128),
        embed_dim=cfg.get("gnn", {}).get("embed_dim", 64),
        metadata=dummy_meta,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("GNN model loaded from {}.", args.model_path)

    # ── Run loop ─────────────────────────────────────────────────────────────
    gnn_refresh_loop(
        model=model,
        r=r,
        driver=driver,
        device=device,
        interval_secs=args.interval,
        lookback_secs=args.lookback,
        card_map_path=args.card_map,
    )
