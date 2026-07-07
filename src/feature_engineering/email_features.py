from __future__ import annotations

import pandas as pd
from loguru import logger


# Common free email providers
FREE_EMAIL_DOMAINS = frozenset([
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "aol.com", "icloud.com", "live.com", "msn.com",
    "protonmail.com", "mail.com",
])

# Trusted (low-fraud) domains — example whitelist, expand as needed
TRUSTED_DOMAINS = frozenset([
    "gmail.com", "yahoo.com", "hotmail.com",
    "comcast.net", "sbcglobal.net", "att.net",
])


def build_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build email-based features from P_emaildomain and R_emaildomain.
    """
    if "P_emaildomain" not in df.columns:
        df["P_emaildomain"] = "unknown"
    if "R_emaildomain" not in df.columns:
        df["R_emaildomain"] = "unknown"

    p = df["P_emaildomain"].fillna("unknown")
    r = df["R_emaildomain"].fillna("unknown")

    # ── email_match: buyer and seller use same domain ─────────────────────────
    df["email_match"] = (p == r).astype(int)

    # ── free-provider flags ───────────────────────────────────────────────────
    df["P_email_is_free"] = p.isin(FREE_EMAIL_DOMAINS).astype(int)
    df["R_email_is_free"] = r.isin(FREE_EMAIL_DOMAINS).astype(int)

    # ── trusted domain flags ──────────────────────────────────────────────────
    df["email_trusted"] = p.isin(TRUSTED_DOMAINS).astype(int)

    # ── TLD extraction (ACM 2025 method) ─────────────────────────────────────
    # P_emaildomain = "gmail.com"  → P_email_tld = "com"
    # P_emaildomain = "yahoo.co.uk" → P_email_tld = "uk"
    df["P_email_tld"] = p.str.split(".").str[-1]
    df["R_email_tld"] = r.str.split(".").str[-1]

    logger.info(
        "Email features added: email_match={:.2%}, P_email_is_free={:.2%}, "
        "email_trusted={:.2%}.",
        df["email_match"].mean(),
        df["P_email_is_free"].mean(),
        df["email_trusted"].mean(),
    )
    return df
