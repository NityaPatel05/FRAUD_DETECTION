from __future__ import annotations

import re

import pandas as pd
from loguru import logger


# Regex patterns for OS family extraction from DeviceInfo / id_30
_OS_PATTERN = re.compile(
    r"(Windows|iOS|Android|Mac|Linux)",
    flags=re.IGNORECASE,
)

# Regex patterns for browser family extraction from id_31
_BROWSER_PATTERN = re.compile(
    r"(chrome|safari|firefox|edge|ie)",
    flags=re.IGNORECASE,
)


def build_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build device OS, browser, and mobile flag features.

    ACM 2025 approach: extract OS family from DeviceInfo, browser family
    from id_31.  Mobile flag from DeviceType.
    """
    if "DeviceInfo" not in df.columns:
        df["DeviceInfo"] = "unknown"
    if "id_31" not in df.columns:
        df["id_31"] = "unknown"
    if "DeviceType" not in df.columns:
        df["DeviceType"] = "unknown"

    # ── OS family from DeviceInfo (regex, ACM 2025) ───────────────────────────
    device_info = df["DeviceInfo"].fillna("unknown").astype(str)
    df["device_os"] = device_info.str.extract(_OS_PATTERN, expand=False).fillna("Other")

    # ── Browser family from id_31 (drop version number) ──────────────────────
    id31 = df["id_31"].fillna("unknown").astype(str)
    df["device_browser"] = id31.str.extract(_BROWSER_PATTERN, expand=False).fillna("Other")

    # ── Mobile flag ───────────────────────────────────────────────────────────
    df["device_mobile"] = (df["DeviceType"] == "mobile").astype(int)

    # ── Label-encode OS and browser (low cardinality after regex) ────────────
    os_map     = {v: i for i, v in enumerate(df["device_os"].unique())}
    browser_map = {v: i for i, v in enumerate(df["device_browser"].unique())}

    df["device_os_encoded"]      = df["device_os"].map(os_map).fillna(-1).astype(int)
    df["device_browser_encoded"] = df["device_browser"].map(browser_map).fillna(-1).astype(int)

    logger.info(
        "Device features added: device_os ({} unique), "
        "device_browser ({} unique), device_mobile ({:.1%}).",
        df["device_os"].nunique(),
        df["device_browser"].nunique(),
        df["device_mobile"].mean(),
    )
    return df


def get_device_encoders(df: pd.DataFrame) -> dict:
    """
    Return OS and browser label maps fitted on the current df.
    Save these to apply consistently on val/test.
    """
    os_map      = {v: i for i, v in enumerate(sorted(df["device_os"].dropna().unique()))}
    browser_map = {v: i for i, v in enumerate(sorted(df["device_browser"].dropna().unique()))}
    return {"os_map": os_map, "browser_map": browser_map}


def apply_device_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply pre-fitted device encoders (for val/test)."""
    os_map      = encoders["os_map"]
    browser_map = encoders["browser_map"]
    df["device_os_encoded"]      = df["device_os"].map(os_map).fillna(-1).astype(int)
    df["device_browser_encoded"] = df["device_browser"].map(browser_map).fillna(-1).astype(int)
    return df
