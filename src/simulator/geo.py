"""
Static reference tables: countries, cities, MCCs, email domains, device types, and a
synthetic IP-block -> country map.

The IP-to-country map is invented (first octets assigned in order); it is
self-consistent inside the simulator but is NOT real geolocation data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# code, currency, usd_per_unit, currency_decimals, customer_w, online_merchant_w, fraud_home_w, avs_supported, sca_3ds_mandate
_COUNTRIES = [
    ("US", "USD", 1.00,    2, 0.955, 0.92, 0.10, True,  False),
    ("CA", "CAD", 0.73,    2, 0.012, 0.015, 0.02, True,  False),
    ("GB", "GBP", 1.27,    2, 0.008, 0.02, 0.04, True,  True),
    ("MX", "MXN", 0.058,   2, 0.008, 0.005, 0.06, False, False),
    ("DE", "EUR", 1.08,    2, 0.004, 0.015, 0.03, False, True),
    ("FR", "EUR", 1.08,    2, 0.003, 0.01, 0.03, False, True),
    ("IN", "INR", 0.012,   2, 0.006, 0.005, 0.07, False, False),
    ("BR", "BRL", 0.20,    2, 0.006, 0.005, 0.12, False, False),
    ("NG", "NGN", 0.00065, 2, 0.003, 0.0, 0.12, False, False),
    ("RO", "RON", 0.22,    2, 0.003, 0.0, 0.12, False, True),
    ("RU", "RUB", 0.011,   2, 0.004, 0.0, 0.12, False, False),
    ("CN", "CNY", 0.14,    2, 0.004, 0.015, 0.08, False, False),
    ("PH", "PHP", 0.018,   2, 0.004, 0.0, 0.06, False, False),
    ("JP", "JPY", 0.0067,  0, 0.0015, 0.01, 0.01, False, False),
    ("AU", "AUD", 0.66,    2, 0.002, 0.005, 0.02, False, False),
]

# country, city, lat, lon, utc_offset_hours, population weight (millions)
_CITIES = [
    ("US", "New York", 40.71, -74.01, -5, 8.3), ("US", "Los Angeles", 34.05, -118.24, -8, 3.9),
    ("US", "Chicago", 41.88, -87.63, -6, 2.7), ("US", "Houston", 29.76, -95.37, -6, 2.3),
    ("US", "Phoenix", 33.45, -112.07, -7, 1.6), ("US", "Philadelphia", 39.95, -75.17, -5, 1.6),
    ("US", "San Antonio", 29.42, -98.49, -6, 1.5), ("US", "San Diego", 32.72, -117.16, -8, 1.4),
    ("US", "Dallas", 32.78, -96.80, -6, 1.3), ("US", "Miami", 25.76, -80.19, -5, 0.9),
    ("US", "Seattle", 47.61, -122.33, -8, 0.75), ("US", "Denver", 39.74, -104.99, -7, 0.7),
    ("US", "Atlanta", 33.75, -84.39, -5, 0.6), ("US", "Boston", 42.36, -71.06, -5, 0.7),
    ("CA", "Toronto", 43.65, -79.38, -5, 2.8), ("CA", "Montreal", 45.50, -73.57, -5, 1.8),
    ("CA", "Vancouver", 49.28, -123.12, -8, 0.7),
    ("GB", "London", 51.51, -0.13, 0, 9.0), ("GB", "Manchester", 53.48, -2.24, 0, 0.6),
    ("MX", "Mexico City", 19.43, -99.13, -6, 9.0), ("MX", "Guadalajara", 20.66, -103.35, -6, 1.5),
    ("DE", "Berlin", 52.52, 13.40, 1, 3.6), ("DE", "Munich", 48.14, 11.58, 1, 1.5),
    ("FR", "Paris", 48.86, 2.35, 1, 2.1), ("FR", "Lyon", 45.76, 4.84, 1, 0.5),
    ("IN", "Mumbai", 19.08, 72.88, 5.5, 12.0), ("IN", "Delhi", 28.70, 77.10, 5.5, 11.0),
    ("BR", "Sao Paulo", -23.55, -46.63, -3, 12.0), ("BR", "Rio de Janeiro", -22.91, -43.17, -3, 6.7),
    ("NG", "Lagos", 6.52, 3.38, 1, 15.0), ("RO", "Bucharest", 44.43, 26.10, 2, 1.8),
    ("RU", "Moscow", 55.76, 37.62, 3, 12.0), ("CN", "Shanghai", 31.23, 121.47, 8, 24.0),
    ("PH", "Manila", 14.60, 120.98, 8, 1.8), ("JP", "Tokyo", 35.68, 139.69, 9, 14.0),
    ("AU", "Sydney", -33.87, 151.21, 10, 5.3),
]

# mcc, name, log-amount mu, log-amount sigma, card-not-present share, tx-share weight, physical goods, high resale risk
_MCCS = [
    ("5411", "grocery",           3.5, 0.75, 0.08, 0.200, True,  False),
    ("5812", "restaurants",       3.2, 0.60, 0.08, 0.140, False, False),
    ("5541", "fuel",              3.6, 0.45, 0.00, 0.090, False, False),
    ("5311", "department_store",  4.0, 0.90, 0.35, 0.060, True,  False),
    ("5651", "clothing",          4.1, 0.75, 0.50, 0.050, True,  False),
    ("5732", "electronics",       5.0, 1.00, 0.55, 0.035, True,  True),
    ("5944", "jewelry",           5.4, 1.00, 0.30, 0.008, True,  True),
    ("5947", "gift_cards",        4.0, 0.70, 0.60, 0.015, False, True),
    ("5999", "misc_retail",       3.8, 0.90, 0.50, 0.070, True,  False),
    ("5912", "pharmacy",          3.3, 0.70, 0.10, 0.050, False, False),
    ("4121", "rideshare",         2.9, 0.50, 0.95, 0.060, False, False),
    ("4814", "telecom",           4.0, 0.40, 0.95, 0.025, False, False),
    ("4899", "streaming",         2.6, 0.40, 1.00, 0.040, False, False),
    ("5815", "digital_goods",     2.3, 0.90, 1.00, 0.050, False, True),
    ("5816", "games",             2.8, 1.00, 1.00, 0.025, False, True),
    ("4511", "airlines",          5.9, 0.70, 0.90, 0.012, False, True),
    ("7011", "hotels",            5.3, 0.70, 0.60, 0.015, False, False),
    ("6051", "crypto_quasi_cash", 5.0, 1.00, 1.00, 0.004, False, True),
    ("6011", "atm_cash",          4.6, 0.60, 0.00, 0.030, False, False),
    ("8398", "donations",         3.2, 1.00, 1.00, 0.006, False, False),
]
P2P_MCC = "4829"  # money transfer platform (not part of the shopping MCC mix)

# Legit customers use EVERY domain a fraudster might (privacy providers, throwaway mail, locale
# providers): a domain must never be a deterministic fraud flag (audit finding 1).
_EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "aol.com",
                  "comcast.net", "att.net", "msn.com", "live.com", "protonmail.com", "mail.ru",
                  "yandex.ru", "guerrillamail.com", "tutanota.com"]
LEGIT_EMAIL = (_EMAIL_DOMAINS,
               [0.430, 0.130, 0.085, 0.075, 0.075, 0.035, 0.035, 0.025, 0.025, 0.018,
                0.030, 0.012, 0.008, 0.007, 0.010])
# locale-specific legit use (RU / ex-USSR customers really do use mail.ru and yandex)
LOCALE_EMAIL = {
    "RU": [0.20, 0.03, 0.01, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.40, 0.28, 0.01, 0.01],
    "RO": [0.60, 0.10, 0.06, 0.06, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.05, 0.04, 0.01, 0.02],
    "CN": [0.15, 0.02, 0.02, 0.10, 0.05, 0.0, 0.0, 0.0, 0.0, 0.10, 0.02, 0.02, 0.02, 0.0, 0.0],
}
FRAUD_EMAIL = (_EMAIL_DOMAINS,
               [0.434, 0.115, 0.075, 0.072, 0.065, 0.030, 0.028, 0.020, 0.020, 0.014,
                0.050, 0.025, 0.017, 0.020, 0.015])

DEVICE_OS = (["iOS", "Android", "Windows", "macOS", "Linux"], [0.37, 0.29, 0.19, 0.10, 0.05])
FRAUD_DEVICE_OS = (["iOS", "Android", "Windows", "macOS", "Linux"], [0.31, 0.33, 0.22, 0.09, 0.05])
BROWSER_BY_OS = {
    "iOS": (["Safari", "Chrome", "App"], [0.55, 0.15, 0.30]),
    "Android": (["Chrome", "Samsung Internet", "App"], [0.55, 0.15, 0.30]),
    "Windows": (["Chrome", "Edge", "Firefox"], [0.65, 0.25, 0.10]),
    "macOS": (["Safari", "Chrome", "Firefox"], [0.50, 0.45, 0.05]),
    # developers, scrapers/aggregators and monitoring bots are legit Linux/headless traffic too
    "Linux": (["Chrome", "Firefox", "Headless Chrome"], [0.45, 0.33, 0.22]),
}

# Hourly profile of activity in LOCAL time (shape similar to IEEE-CIS: quiet 2-7am).
GLOBAL_HOURLY = np.array([2.2, 1.4, 0.9, 0.6, 0.5, 0.6, 1.0, 1.8, 2.8, 3.8, 4.6, 5.2,
                          5.6, 5.8, 5.8, 5.9, 6.0, 6.0, 5.9, 5.7, 5.3, 4.6, 3.8, 3.0])
GLOBAL_HOURLY = GLOBAL_HOURLY / GLOBAL_HOURLY.sum()


def email_probs(country_codes: np.ndarray) -> np.ndarray:
    """(n, n_domains) legit email-domain probabilities per customer country."""
    base = np.asarray(LEGIT_EMAIL[1], float)
    base = base / base.sum()
    out = np.tile(base, (len(country_codes), 1))
    for code, p in LOCALE_EMAIL.items():
        m = np.asarray(country_codes) == code
        if m.any():
            p = np.asarray(p, float)
            out[m] = 0.85 * p / p.sum() + 0.15 * base
    return out


def countries_df() -> pd.DataFrame:
    df = pd.DataFrame(_COUNTRIES, columns=["code", "currency", "usd_per_unit", "decimals",
                                           "customer_w", "online_merchant_w", "fraud_home_w",
                                           "avs_supported", "sca"])
    for c in ["customer_w", "online_merchant_w", "fraud_home_w"]:
        df[c] = df[c] / df[c].sum()
    return df


def cities_df(countries: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(_CITIES, columns=["country", "city", "lat", "lon", "tz", "pop"])
    code_to_idx = {c: i for i, c in enumerate(countries["code"])}
    df["country_idx"] = df["country"].map(code_to_idx).astype(np.int64)
    # weight of a city among its country's customers
    df["w_in_country"] = df["pop"] / df.groupby("country")["pop"].transform("sum")
    return df


def mccs_df() -> pd.DataFrame:
    df = pd.DataFrame(_MCCS, columns=["mcc", "name", "mu", "sigma", "cnp_share", "weight",
                                      "physical", "high_risk"])
    df["weight"] = df["weight"] / df["weight"].sum()
    return df


def ip_blocks(n_countries: int, hosting_countries: list[int]) -> tuple[dict, dict]:
    """First-octet blocks per country (residential/mobile) and per hosting country."""
    usable = [o for o in range(1, 224) if o not in (10, 100, 127, 169, 172, 192, 198, 203)]
    it = iter(usable)
    res = {0: np.array([next(it) for _ in range(40)])}             # US gets a big block
    for c in range(1, n_countries):
        res[c] = np.array([next(it) for _ in range(3)])
    host = {c: np.array([next(it) for _ in range(2)]) for c in hosting_countries}
    return res, host


def ip_to_str(ip_int: np.ndarray) -> np.ndarray:
    ip_int = np.asarray(ip_int, dtype=np.int64)
    parts = [((ip_int >> s) & 255).astype(str) for s in (24, 16, 8, 0)]
    out = parts[0]
    for p in parts[1:]:
        out = np.char.add(np.char.add(out, "."), p)
    return out.astype(object)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
