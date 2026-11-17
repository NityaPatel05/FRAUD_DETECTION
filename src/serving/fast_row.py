"""
Single-transaction fast path for ``OnlineFeatureEngine.row_frame`` (pure python/numpy, no DataFrame).

The batch row-level preprocessing (``rename id-``, dtype coercion, ``build_base_features``, ``build_uid_keys``,
``_downcast``, ``add_derived_signals``) is re-expressed here for ONE transaction dict. It must return exactly
what ``row_frame(pd.DataFrame([txn])).iloc[0].to_dict()`` returns (same keys, values, float32 rounding, missing
markers): ``tests/test_serving_fast_row.py`` checks that on hundreds of varied rows. Anything unusual (exotic
types, duplicate ``id-``/``id_`` keys, a bundle schema this code does not know) raises ``FastRowUnsupported``
and the engine silently uses the slow pandas path for that request.

Numeric care (these are what make the keys byte-identical to offline):
  * values are rounded to the raw column dtype (float32 columns hold float32-rounded values),
  * strings of float32 columns use ``str(np.float32(x))`` (shortest float32 repr, as pandas does),
  * all float64 columns except TransactionAmt are downcast to float32 AFTER the uid/combo keys were built,
  * numpy scalar/array ufuncs (log1p, round) are used where the batch code uses them.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

import numpy as np

from src.feature_engineering.base_features import D_NORMALISE, M_TF_COLS
from src.feature_engineering.derived_signals import (
    COUNTRY_TLD, FREE_MAIL, ISP_MAIL, PRIVACY_MAIL,
)

NAN = float("nan")
_ABSENT = object()
_SUPPORTED_DTYPES = {"object", "int64", "float32", "float64"}
_REQUIRED = {"TransactionID", "TransactionDT", "TransactionAmt", "card1", "addr1", "D1", "P_emaildomain"}

_RE_BRAND = re.compile(r"[ /\-_]")
_RE_FAM_STRIP = re.compile(r"^(mobile|generic/android|samsung/sm-[a-z0-9]+)\s*")
_RE_FAM = re.compile(r"^([a-z ]+?)(?:\s*[\d/]|$)")
_RE_VER = re.compile(r"(\d+(?:\.\d+)?)")
_RE_OS = re.compile(r"^([a-z ]+?)(?:\s*\d|$)")
_RE_WH = re.compile(r"(\d+)x(\d+)")


class FastRowUnsupported(Exception):
    """Input/schema outside what the fast path reproduces exactly -> use the pandas path."""


def _miss(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def _a(x) -> np.ndarray:
    return np.array([x], dtype=np.float64)


class FastRowBuilder:
    def __init__(self, meta: Dict):
        self.cols = list(meta["raw_columns"])
        self.dtypes = dict(meta["raw_dtypes"])
        self.dropped_v = set(meta.get("dropped_v_cols", []))
        self.use_derived = bool(meta["fe_cfg"].get("use_derived_signals", True))
        colset = set(self.cols)
        self.colset = colset
        self.ok = (_REQUIRED <= colset and all(self.dtypes.get(c) in _SUPPORTED_DTYPES for c in self.cols)
                   and (("dist1" not in colset) or "card2" in colset))
        self.keep = [c for c in self.cols if c not in self.dropped_v and c not in ("day", "hour")]
        self.m_map = {c: ({"T": 1.0, "F": 0.0} if c != "M4" else {"M0": 0.0, "M1": 1.0, "M2": 2.0})
                      for c in M_TF_COLS + ["M4"] if c in colset and self.dtypes[c] == "object"}
        self.d_norm = [c for c in D_NORMALISE if c in colset]
        self.has = colset.__contains__

    # ── coercion to the raw dtype (what ``astype(raw_dtypes)`` does, for the types we accept) ───────
    def _coerce(self, c: str, v):
        dt = self.dtypes[c]
        if dt == "object":
            if v is _ABSENT:
                return NAN
            if v is None or isinstance(v, str) or (isinstance(v, float) and v != v):
                return v
            raise FastRowUnsupported(c)
        if v is _ABSENT or v is None:
            return NAN
        if isinstance(v, (bool, np.bool_)):
            raise FastRowUnsupported(c)
        if isinstance(v, (int, np.integer)):
            f, is_int = float(v), True
        elif isinstance(v, (float, np.floating)):
            f, is_int = float(v), False
        else:
            raise FastRowUnsupported(c)
        if dt == "int64":
            if is_int:
                return int(v)
            if f != f:
                return NAN
            if f in (float("inf"), float("-inf")) or not f.is_integer():
                raise FastRowUnsupported(c)
            return int(f)
        if dt == "float32":
            return float(np.float32(f))
        return f                                                      # float64

    @staticmethod
    def _s(dt: str, x) -> Optional[str]:
        """``astype("string")`` of a numeric raw value (None = <NA>)."""
        if x is None or (isinstance(x, float) and x != x):
            return None
        if dt == "float32":
            return str(np.float32(x))
        if dt == "float64":
            return str(float(x))
        return str(int(x))

    @staticmethod
    def _int_str(x) -> str:
        return str(int(np.round(np.float64(x))))

    # ── the row ──────────────────────────────────────────────────────────────────────────────────────
    def build(self, txn: Dict) -> Dict:
        if not self.ok:
            raise FastRowUnsupported("schema")
        src: Dict = {}
        for k, v in txn.items():
            if isinstance(k, str) and k.startswith("id-"):
                k = k.replace("id-", "id_")
            if k in src:
                raise FastRowUnsupported("duplicate columns")         # id-19 + id_19: slow path bfill-merges
            src[k] = v
        dts, has = self.dtypes, self.has
        raw: Dict = {c: self._coerce(c, src.get(c, _ABSENT)) for c in self.keep}   # post-astype values

        row: Dict = {}
        t = raw["TransactionDT"]
        if not isinstance(t, int):
            raise FastRowUnsupported("TransactionDT")
        day = t // 86400
        hour = (t // 3600) % 24
        amt = float(raw["TransactionAmt"])

        # build_base_features -------------------------------------------------------------------------
        for c, m in self.m_map.items():                               # encode_m (object columns only)
            v = raw[c]
            raw[c] = NAN if _miss(v) else m.get(v, NAN)
        floats = {}                                                   # float64 intermediates (pre-downcast)
        for c in self.d_norm:
            floats[c] = float(np.float32(day - raw[c])) if not _miss(raw[c]) else NAN
        d1 = raw["D1"]
        d1n = NAN if _miss(d1) else float(np.round(np.float64(day - d1)))
        a = _a(amt)
        log_amt = float(np.float32(np.log1p(a)[0]))
        cents = float(np.float32(np.round(a - np.floor(a), 3)[0]))

        P = raw["P_emaildomain"] if has("P_emaildomain") else None
        R = raw.get("R_emaildomain") if has("R_emaildomain") else None
        derived_str: Dict = {}
        for side, val in (("P", P), ("R", R)):
            if has(f"{side}_emaildomain"):
                derived_str[f"{side}_email_prefix"] = None if _miss(val) else val.split(".")[0]
        p_eq_r = NAN
        if has("P_emaildomain") and has("R_emaildomain") and not _miss(P) and not _miss(R):
            p_eq_r = float(P == R)
        if has("id_30"):
            v = raw["id_30"]
            derived_str["device_os"] = None if _miss(v) else v.split(" ")[0]
        if has("id_31"):
            v = raw["id_31"]
            derived_str["browser"] = None if _miss(v) else v.split(" ")[0]
        if has("DeviceInfo"):
            v = raw["DeviceInfo"]
            derived_str["device_brand"] = None if _miss(v) else _RE_BRAND.split(v)[0].lower()

        # combo keys (strings built BEFORE the float64 -> float32 downcast, from raw-dtype values)
        s_card1 = self._s(dts["card1"], raw["card1"])
        s_addr1 = self._s(dts["addr1"], raw["addr1"])
        s_p = None if _miss(P) else P
        combos: Dict = {}
        combos["card1_addr1"] = None if s_card1 is None or s_addr1 is None else f"{s_card1}_{s_addr1}"
        combos["card1_addr1_P_emaildomain"] = (None if s_card1 is None or s_addr1 is None or s_p is None
                                               else f"{s_card1}_{s_addr1}_{s_p}")
        if has("dist1"):
            s_c2, s_d = self._s(dts["card2"], raw["card2"]), self._s(dts["dist1"], raw["dist1"])
            combos["card2_dist1"] = None if s_c2 is None or s_d is None else f"{s_c2}_{s_d}"
        else:
            combos["card2_dist1"] = None
        di = raw["DeviceInfo"] if has("DeviceInfo") else None
        if has("DeviceInfo") and not _miss(di):
            fp = di
            for c in ("id_19", "id_20", "id_31", "id_33"):
                if has(c):
                    v = raw[c]
                    s = (None if _miss(v) else v) if dts[c] == "object" else self._s(dts[c], v)
                    fp = fp + "_" + ("na" if s is None else s)
            combos["device_fp"] = fp
        else:
            combos["device_fp"] = None

        # build_uid_keys (float64 values, before the downcast)
        parts_ok = not _miss(raw["card1"]) and not _miss(raw["addr1"]) and not _miss(d1n)
        uid1 = uid2 = None
        if parts_ok:
            uid1 = f"{self._int_str(raw['card1'])}_{self._int_str(raw['addr1'])}_{self._int_str(d1n)}"
            if not _miss(P):
                uid2 = f"{uid1}_{P}"

        # _downcast: float64 -> float32 for everything but TransactionAmt (raw cols, mapped M, D1n, P_eq_R)
        for c in self.keep:
            v = raw[c]
            if dts[c] == "float64" and c != "TransactionAmt":
                v = float(np.float32(v))
            elif dts[c] == "int64" and isinstance(v, float):
                v = NAN
            row[c] = v
        if "TransactionAmt" in raw and dts["TransactionAmt"] != "float64":
            row["TransactionAmt"] = raw["TransactionAmt"]
        for c in self.m_map:                                          # mapped M columns became float64 -> float32
            row[c] = float(np.float32(raw[c]))
        row.update({"day": day, "hour": hour, "weekday": day % 7})
        row.update(floats)
        row["D1n"] = float(np.float32(d1n))
        row["TransactionAmt_log"], row["cents"] = log_amt, cents
        row.update(derived_str)
        if has("P_emaildomain") and has("R_emaildomain"):
            row["P_eq_R_email"] = float(np.float32(p_eq_r))
        row.update(combos)
        row["uid1"], row["uid2"] = uid1, uid2

        if self.use_derived:
            self._derived(row, amt)
        return row

    # ── add_derived_signals on the downcast row ──────────────────────────────────────────────────────
    def _derived(self, row: Dict, amt: float) -> None:
        has = self.has
        a = _a(amt)
        frac100 = a * 100
        whole = np.abs(a - np.round(a)) < 1e-9
        row["amt_gt2_decimals"] = float(np.float32(bool((np.abs(frac100 - np.round(frac100)) > 1e-6)[0])))
        row["amt_is_whole"] = float(np.float32(bool(whole[0])))
        row["amt_is_round10"] = float(np.float32(bool((whole & (np.round(a) % 10 == 0))[0])))
        if has("addr2"):
            v = row["addr2"]
            row["addr2_foreign"] = NAN if _miss(v) else float(v != 87)
        if has("card3"):
            v = row["card3"]
            row["card3_not150"] = NAN if _miss(v) else float(v != 150)
        tld_c: Dict = {}
        for side in ("P", "R"):
            col = f"{side}_emaildomain"
            if has(col):
                d = row[col]
                if _miss(d):
                    row[f"{side}_email_class"] = row[f"{side}_email_tld_country"] = None
                    tld_c[side] = None
                    continue
                d = d.lower()
                cls = "corporate_other"
                for name, keys in (("free", FREE_MAIL), ("isp", ISP_MAIL), ("privacy", PRIVACY_MAIL)):
                    if any(k in d for k in keys):
                        cls = name
                if d == "anonymous.com":
                    cls = "placeholder"
                tld = COUNTRY_TLD.get(d.rsplit(".", 1)[-1])
                row[f"{side}_email_class"], row[f"{side}_email_tld_country"] = cls, tld
                tld_c[side] = tld
        if has("P_emaildomain") and has("addr2"):
            a2 = row["addr2"]
            row["email_tld_vs_us_addr"] = NAN if _miss(a2) else float(tld_c.get("P") is not None and a2 == 87)
        if has("id_23"):
            s = row["id_23"]
            row["is_proxy"] = NAN if _miss(s) else float("PROXY" in s)
            row["proxy_type"] = None if _miss(s) else s.replace("IP_PROXY:", "")
        found = [c for c in ("id_12", "id_15", "id_16", "id_28", "id_29") if has(c)]
        if found:
            vals = [row[c] for c in found]
            row["id_found_count"] = float(sum(v == "Found" for v in vals if not _miss(v)))
            row["id_new_count"] = float(sum(v == "New" for v in vals if not _miss(v)))
            row["id_notfound_count"] = float(sum(v == "NotFound" for v in vals if not _miss(v)))
        if has("id_31"):
            b = row["id_31"]
            fam = ver = None
            if not _miss(b):
                b = b.lower()
                m = _RE_FAM.search(_RE_FAM_STRIP.sub("", b))
                fam = m.group(1).strip() if m else None
                m = _RE_VER.search(b)
                ver = float(m.group(1)) if m else None
            row["browser_family"] = fam
            row["browser_version"] = NAN if ver is None else float(np.float32(ver))
            row["browser_version_lag"] = NAN if ver is None else 0.0     # running max of a single row = itself
        if has("id_30"):
            s = row["id_30"]
            fam = None
            if not _miss(s):
                m = _RE_OS.search(s.lower())
                fam = m.group(1).strip() if m else None
            row["os_family"] = fam
        if has("id_33"):
            s = row["id_33"]
            m = None if _miss(s) else _RE_WH.search(s)
            row["screen_area"] = NAN if m is None else float(np.float32(float(m.group(1)) * float(m.group(2))))
