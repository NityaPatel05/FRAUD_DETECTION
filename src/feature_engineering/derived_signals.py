"""
Phase 2 — honest, real-world-style signals derived from EXISTING columns, plus
column removal and the "raw" (real-world gateway) schema.

Nothing here is fabricated or label-conditioned: every feature is a deterministic
re-expression of fields the transaction already carries, mirroring a signal a real
payment system has (cross-border flag, proxy flag, new-device flag, email type…).
Evidence for each comes from the read-only data study (fraud rates in comments are
train-period rates).
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

# Columns that are NOT raw fields a payment gateway receives: Vesta's own
# pre-computed counts / timedeltas / engineered features. Removed in schema=raw so
# that every model input is something our own system can compute from raw events.
def vendor_engineered_cols(columns) -> List[str]:
    out = []
    for c in columns:
        if re.fullmatch(r"V\d+", c) or re.fullmatch(r"C\d+", c):
            out.append(c)
        elif re.fullmatch(r"D\d+", c) and c != "D1":          # D1 = card age, kept
            out.append(c)
        elif re.fullmatch(r"id_(0[1-9]|1[01])", c):            # numeric vendor scores
            out.append(c)
    return out


# Data-study removals: near-constant, >99% missing, duplicates.
DEFAULT_DROP = [
    "V107", "V305", "V108", "V110", "V111", "V112", "V113", "V114",
    "V117", "V118", "V119", "V120", "V121", "V122", "C3",
    "id_07", "id_08", "id_21", "id_22", "id_24", "id_25", "id_26", "id_27",
    "D9",    # == fractional part of D8 (time of day; `hour` covers it)
    "D2",    # == D1 in 92.8% of rows
    "C10",   # Spearman 0.971 with C8
]

FREE_MAIL = ("gmail", "yahoo", "ymail", "rocketmail", "hotmail", "outlook", "live", "msn",
             "aol", "icloud", "me.com", "mac.com", "mail.com", "gmx", "yandex", "web.de")
ISP_MAIL = ("comcast", "att.net", "verizon", "sbcglobal", "bellsouth", "cox.net", "charter",
            "earthlink", "centurylink", "frontier", "optonline", "windstream", "juno", "twc",
            "roadrunner", "rr.com", "suddenlink", "embarqmail", "q.com", "ptd.net", "netzero",
            "prodigy", "cableone", "servicios-ta")
PRIVACY_MAIL = ("protonmail",)
COUNTRY_TLD = {"mx": "MX", "de": "DE", "jp": "JP", "fr": "FR", "es": "ES", "uk": "UK",
               "it": "IT", "ca": "CA", "br": "BR"}


def email_class(domain: pd.Series) -> pd.Series:
    d = domain.astype("string").str.lower()
    out = pd.Series(pd.NA, index=domain.index, dtype="string")
    out[d.notna()] = "corporate_other"
    for name, keys in [("free", FREE_MAIL), ("isp", ISP_MAIL), ("privacy", PRIVACY_MAIL)]:
        m = d.notna() & d.apply(lambda x: isinstance(x, str) and any(k in x for k in keys))
        out[m] = name
    out[d == "anonymous.com"] = "placeholder"
    return out


def email_tld_country(domain: pd.Series) -> pd.Series:
    tld = domain.astype("string").str.lower().str.rsplit(".", n=1).str[-1]
    return tld.map(COUNTRY_TLD).astype("string")


def add_derived_signals(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return new columns (strings are label-encoded later by the pipeline)."""
    out: Dict[str, object] = {}
    amt = df["TransactionAmt"].astype(np.float64)

    # Foreign-currency / cross-border proxies (real-world: card country != IP country)
    frac100 = amt * 100
    out["amt_gt2_decimals"] = (np.abs(frac100 - np.round(frac100)) > 1e-6).astype(np.float32)  # 11.7% vs 2.5%
    out["amt_is_whole"] = (np.abs(amt - np.round(amt)) < 1e-9).astype(np.float32)
    out["amt_is_round10"] = ((np.abs(amt - np.round(amt)) < 1e-9) & (np.round(amt) % 10 == 0)).astype(np.float32)
    if "addr2" in df.columns:
        out["addr2_foreign"] = np.where(df["addr2"].isna(), np.nan, (df["addr2"] != 87).astype(float))  # 10.2% vs 2.4%
    if "card3" in df.columns:
        out["card3_not150"] = np.where(df["card3"].isna(), np.nan, (df["card3"] != 150).astype(float))   # 11.3% vs 2.5%

    # Email: type of provider and country TLD vs billing country
    for side in ["P", "R"]:
        col = f"{side}_emaildomain"
        if col in df.columns:
            out[f"{side}_email_class"] = email_class(df[col])
            out[f"{side}_email_tld_country"] = email_tld_country(df[col])
    if "P_emaildomain" in df.columns and "addr2" in df.columns:
        foreign_tld = email_tld_country(df["P_emaildomain"]).notna()
        out["email_tld_vs_us_addr"] = np.where(df["addr2"].isna(), np.nan,
                                               (foreign_tld & (df["addr2"] == 87)).astype(float))

    # Anonymising proxy (only IP-type signal in the data; 13.7% fraud)
    if "id_23" in df.columns:
        s = df["id_23"].astype("string")
        out["is_proxy"] = np.where(s.isna(), np.nan, s.str.contains("PROXY", na=False).astype(float))
        out["proxy_type"] = s.str.replace("IP_PROXY:", "", regex=False)

    # Verification / seen-before flags (undocumented match flags — NOT AVS/CVV/3DS)
    found_cols = [c for c in ["id_12", "id_15", "id_16", "id_28", "id_29"] if c in df.columns]
    if found_cols:
        sub = df[found_cols].astype("string")
        out["id_found_count"] = (sub == "Found").sum(axis=1).astype(np.float32)
        out["id_new_count"] = (sub == "New").sum(axis=1).astype(np.float32)
        out["id_notfound_count"] = (sub == "NotFound").sum(axis=1).astype(np.float32)

    # Stable device derivations (raw strings go stale: 60% of test browsers unseen)
    if "id_31" in df.columns:
        b = df["id_31"].astype("string").str.lower()
        fam = (b.str.replace(r"^(mobile|generic/android|samsung/sm-[a-z0-9]+)\s*", "", regex=True)
                .str.extract(r"^([a-z ]+?)(?:\s*[\d/]|$)")[0].str.strip())
        ver = b.str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
        out["browser_family"] = fam
        out["browser_version"] = ver.astype(np.float32)
        # how far behind the newest version of that browser seen SO FAR (causal)
        newest = pd.Series(ver.to_numpy()).groupby(fam.fillna("NA").to_numpy()).cummax().to_numpy()
        out["browser_version_lag"] = (newest - ver.to_numpy()).astype(np.float32)
    if "id_30" in df.columns:
        out["os_family"] = df["id_30"].astype("string").str.lower().str.extract(r"^([a-z ]+?)(?:\s*\d|$)")[0].str.strip()
    if "id_33" in df.columns:
        wh = df["id_33"].astype("string").str.extract(r"(\d+)x(\d+)").astype(float)
        out["screen_area"] = (wh[0] * wh[1]).astype(np.float32)
    logger.info("Derived real-world-style signals: {} columns.", len(out))
    return out
