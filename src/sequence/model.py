"""GRU sequence encoder: self-supervised next-transaction prediction, strictly causal features.

For transaction i of a client, state h_{i-1} summarises that client's transactions < i (h_{-1} = 0).
  embedding_i = W h_{i-1}                      (emb_dim floats)
  surprise_i  = NLL of txn i's attributes (amount bucket, gap bucket, categoricals) predicted from embedding_i
The current txn only enters the state AFTER its features are emitted, so nothing leaks.
Online: per client keep ClientState(h[hidden], last_t) -> O(1) `step`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sequence.data import Encoded, SeqSpec, client_order, gap_inputs, gaps, N_NUM

FEATURE_PREFIX = "seq_emb_"
SURPRISE_COLS = ["seq_surprise", "seq_nll_amt", "seq_nll_gap", "seq_nll_cat"]


@dataclass
class SeqConfig:
    hidden: int = 32
    emb_dim: int = 12
    cat_emb: int = 6
    max_len: int = 96          # training window (unrolled steps); inference streams the full history
    epochs: int = 4
    batch_tokens: int = 16384
    lr: float = 3e-3
    seed: int = 0


@dataclass
class ClientState:
    h: np.ndarray                     # [hidden] float32
    last_t: int = -1                  # seconds of the client's last txn (-1 = unseen client)

    @property
    def nbytes(self) -> int:
        return self.h.nbytes + 8


class _Net(nn.Module):
    def __init__(self, spec: SeqSpec, cfg: SeqConfig):
        super().__init__()
        self.cfg, self.spec = cfg, spec
        self.embs = nn.ModuleList([nn.Embedding(n, cfg.cat_emb) for n in spec.cat_sizes])
        self.amt_emb = nn.Embedding(spec.n_amt, cfg.cat_emb)
        d_in = cfg.cat_emb * (len(spec.cat_cols) + 1) + N_NUM + 2
        self.inp = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU())
        self.gru = nn.GRU(64, cfg.hidden, batch_first=True)
        self.proj = nn.Linear(cfg.hidden, cfg.emb_dim)
        self.heads = nn.ModuleDict({"amt": nn.Linear(cfg.emb_dim, spec.n_amt), "gap": nn.Linear(cfg.emb_dim, spec.n_gap)})
        for j, n in enumerate(spec.cat_sizes):
            self.heads[f"c{j}"] = nn.Linear(cfg.emb_dim, n)

    def encode_in(self, cats, num, amt_b, gin):
        parts = [e(cats[..., j]) for j, e in enumerate(self.embs)] + [self.amt_emb(amt_b), num, gin]
        return self.inp(torch.cat(parts, -1))

    def nll(self, emb, cats, amt_b, gap_b) -> Dict[str, torch.Tensor]:
        """Per-token NLL by head group; shapes [B, T]. Gap NLL is 0 for a client's first txn."""
        def ce(head, tgt):
            lg = self.heads[head](emb)
            return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
        g = ce("gap", gap_b)
        cat = sum(ce(f"c{j}", cats[..., j]) for j in range(len(self.spec.cat_cols)))
        return {"amt": ce("amt", amt_b), "gap": torch.where(gap_b == 0, torch.zeros_like(g), g), "cat": cat}


def _shift(h_out: torch.Tensor) -> torch.Tensor:
    """state BEFORE each position: [0, h_0 .. h_{T-2}]"""
    return torch.cat([torch.zeros_like(h_out[:, :1]), h_out[:, :-1]], 1)


class SequenceEncoder:
    def __init__(self, spec: SeqSpec, cfg: Optional[SeqConfig] = None, device: str = "cpu"):
        self.spec, self.cfg, self.device = spec, cfg or SeqConfig(), device
        torch.manual_seed(self.cfg.seed)
        self.net = _Net(spec, self.cfg).to(device)

    # ---- batching -------------------------------------------------------------------------
    def _prep(self, enc: Encoded, client: np.ndarray, tid: np.ndarray):
        order, _, starts = client_order(client, enc.t, tid)
        g = gaps(enc.t[order], starts)
        arrs = dict(cats=enc.cats[order], num=enc.num[order], amt_b=enc.amt_b[order],
                    gin=gap_inputs(g), gap_b=self.spec.gap_bucket(g))
        return order, starts, arrs

    def _batches(self, starts: np.ndarray, n: int, max_len: Optional[int], rng: Optional[np.random.Generator]):
        """Groups of (start, end) segments of similar length within a padded-token budget."""
        ends = np.r_[starts[1:], n]
        segs = []
        for s, e in zip(starts, ends):
            if max_len is None:
                segs.append((s, e))
            else:
                segs += [(a, min(a + max_len, e)) for a in range(s, e, max_len)]
        L = np.array([e - s for s, e in segs])
        idx = np.argsort(L, kind="stable")
        if rng is not None:
            idx = np.concatenate([rng.permutation(c) for c in np.array_split(idx, max(1, len(idx) // 512))])
        batch, mx = [], 0
        for i in idx:
            if batch and max(mx, L[i]) * (len(batch) + 1) > self.cfg.batch_tokens:
                yield [segs[j] for j in batch]
                batch, mx = [], 0
            batch.append(i)
            mx = max(mx, L[i])
        if batch:
            yield [segs[j] for j in batch]

    def _tensors(self, arrs, group):
        T = max(e - s for s, e in group)
        B = len(group)
        out = {}
        for k, a in arrs.items():
            buf = np.zeros((B, T) + a.shape[1:], dtype=a.dtype)
            for b, (s, e) in enumerate(group):
                buf[b, :e - s] = a[s:e]
            out[k] = torch.from_numpy(buf).to(self.device)
        mask = np.zeros((B, T), dtype=bool)
        for b, (s, e) in enumerate(group):
            mask[b, :e - s] = True
        out["mask"] = torch.from_numpy(mask).to(self.device)
        return out

    def _forward(self, tb):
        x = self.net.encode_in(tb["cats"], tb["num"], tb["amt_b"], tb["gin"])
        h_out, _ = self.net.gru(x)
        emb = self.net.proj(_shift(h_out))
        return emb, self.net.nll(emb, tb["cats"], tb["amt_b"], tb["gap_b"])

    # ---- training (label-free) ------------------------------------------------------------
    def fit(self, enc: Encoded, client: np.ndarray, tid: np.ndarray, log=print) -> List[float]:
        cfg = self.cfg
        _, starts, arrs = self._prep(enc, client, tid)
        rng = np.random.default_rng(cfg.seed)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=1e-4)
        hist = []
        for ep in range(cfg.epochs):
            self.net.train()
            tot, cnt, t0 = 0.0, 0, time.time()
            for group in self._batches(starts, len(enc), cfg.max_len, rng):
                tb = self._tensors(arrs, group)
                _, nl = self._forward(tb)
                m = tb["mask"]
                loss = ((nl["amt"] + nl["gap"] + nl["cat"]) * m).sum() / m.sum()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * int(m.sum())
                cnt += int(m.sum())
            hist.append(tot / cnt)
            log(f"seq epoch {ep + 1}/{cfg.epochs} loss {hist[-1]:.4f} ({time.time() - t0:.1f}s)")
        return hist

    # ---- batch features (streaming-equivalent) --------------------------------------------
    @torch.no_grad()
    def transform(self, enc: Encoded, client: np.ndarray, tid: np.ndarray) -> pd.DataFrame:
        """Causal features for every row (original row order). History = all rows given (time order)."""
        self.net.eval()
        order, starts, arrs = self._prep(enc, client, tid)
        n, E = len(enc), self.cfg.emb_dim
        res = np.zeros((n, E + 4), dtype=np.float32)
        for group in self._batches(starts, n, None, None):
            tb = self._tensors(arrs, group)
            emb, nl = self._forward(tb)
            tot = nl["amt"] + nl["gap"] + nl["cat"]
            feat = torch.cat([emb, tot[..., None], nl["amt"][..., None], nl["gap"][..., None],
                              nl["cat"][..., None]], -1).cpu().numpy()
            for b, (s, e) in enumerate(group):
                res[s:e] = feat[b, :e - s]
        out = np.zeros_like(res)
        out[order] = res
        return pd.DataFrame(out, columns=[f"{FEATURE_PREFIX}{i}" for i in range(E)] + SURPRISE_COLS)

    # ---- online ---------------------------------------------------------------------------
    def init_state(self) -> ClientState:
        return ClientState(np.zeros(self.cfg.hidden, dtype=np.float32), -1)

    @torch.no_grad()
    def step(self, state: ClientState, txn: Encoded) -> Tuple[np.ndarray, np.ndarray, ClientState]:
        """One transaction (Encoded of length 1) of a client. O(1) in history length.

        -> (embedding [emb_dim], surprise [total, amt, gap, cat], new_state). Features use the state
        BEFORE this txn; the txn is folded into the state afterwards."""
        self.net.eval()
        t = int(txn.t[0])
        g = np.array([-1.0 if state.last_t < 0 else float(t - state.last_t)])
        dev = self.device
        c = torch.from_numpy(txn.cats).to(dev)[:, None]
        nu = torch.from_numpy(txn.num).to(dev)[:, None]
        a = torch.from_numpy(txn.amt_b).to(dev)[:, None]
        gi = torch.from_numpy(gap_inputs(g)).to(dev)[:, None]
        gb = torch.from_numpy(self.spec.gap_bucket(g)).to(dev)[:, None]
        h0 = torch.from_numpy(state.h).to(dev)[None, None]            # [layers=1, B=1, H]
        emb = self.net.proj(h0[0])                                    # [1, E] from the state before the txn
        nl = self.net.nll(emb[:, None], c, a, gb)
        sur = torch.stack([nl["amt"] + nl["gap"] + nl["cat"], nl["amt"], nl["gap"], nl["cat"]], -1)[0, 0]
        _, hn = self.net.gru(self.net.encode_in(c, nu, a, gi), h0)
        return (emb[0].cpu().numpy(), sur.cpu().numpy(),
                ClientState(hn[0, 0].cpu().numpy().astype(np.float32), t))

    # ---- persistence ----------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"state": self.net.state_dict(), "cfg": asdict(self.cfg), "spec": self.spec}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SequenceEncoder":
        ck = torch.load(path, map_location=device, weights_only=False)
        enc = cls(ck["spec"], SeqConfig(**ck["cfg"]), device)
        enc.net.load_state_dict(ck["state"])
        return enc
