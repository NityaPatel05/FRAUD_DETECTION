"""
Payment checks and authorisation outcome — driven by MECHANISMS, never by the label.

``cred_type`` says what the payer actually holds (owner, card on file, full stolen
details, a dump without CVC, a guessed PAN, a counterfeit stripe). AVS / CVC / 3DS
results are drawn from that, and the response code follows deterministic-ish issuer
rules (CVC mismatch -> N7, failed 3DS -> 05, invalid PAN -> 14, ...) plus an issuer risk
score that reacts to *observable* things (foreign IP, hosting IP, unusual amount)
for legit and fraud rows alike. Codes follow ISO 8583 field 39 conventions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .entities import World

# AVS: Y full match, A address only, Z zip only, N no match, U unavailable
AVS = {
    "owner":            (["Y", "A", "Z", "N", "U"], [0.86, 0.04, 0.04, 0.02, 0.04]),
    "card_on_file":     (["Y", "A", "Z", "N", "U"], [0.95, 0.02, 0.02, 0.00, 0.01]),
    "stolen_full":      (["Y", "A", "Z", "N", "U"], [0.76, 0.07, 0.07, 0.05, 0.05]),
    "stolen_partial":   (["Y", "A", "Z", "N", "U"], [0.20, 0.12, 0.22, 0.30, 0.16]),
    "enumerated_valid": (["Y", "A", "Z", "N", "U"], [0.02, 0.03, 0.05, 0.60, 0.30]),
    "invalid_pan":      (["Y", "A", "Z", "N", "U"], [0.00, 0.00, 0.00, 0.50, 0.50]),
}
# CVC: M match, N no match, P not processed / not provided
CVC = {
    "owner":            (["M", "N", "P"], [0.95, 0.01, 0.04]),
    "card_on_file":     (["M", "N", "P"], [0.10, 0.00, 0.90]),
    "stolen_full":      (["M", "N", "P"], [0.90, 0.03, 0.07]),
    "stolen_partial":   (["M", "N", "P"], [0.25, 0.35, 0.40]),
    "enumerated_valid": (["M", "N", "P"], [0.10, 0.70, 0.20]),
    "invalid_pan":      (["M", "N", "P"], [0.00, 0.50, 0.50]),
}
# 3DS (when the merchant runs it): Y frictionless ok, C challenge passed, A attempted,
# N authentication failed, R rejected
THREE_DS = {
    "owner":            (["Y", "C", "A", "N", "R"], [0.86, 0.09, 0.02, 0.02, 0.01]),
    "stolen_full":      (["Y", "C", "A", "N", "R"], [0.62, 0.20, 0.04, 0.10, 0.04]),
    "card_on_file":     (["Y", "C", "A", "N", "R"], [0.66, 0.18, 0.04, 0.09, 0.03]),   # ATO on a stored card, ecom flow
    "stolen_partial":   (["Y", "C", "A", "N", "R"], [0.30, 0.25, 0.02, 0.35, 0.08]),
    "enumerated_valid": (["Y", "C", "A", "N", "R"], [0.10, 0.00, 0.00, 0.60, 0.30]),
    "invalid_pan":      (["Y", "C", "A", "N", "R"], [0.00, 0.00, 0.00, 0.40, 0.60]),
}


def _sample_by(rng, keys: np.ndarray, table: dict, mask: np.ndarray, out: np.ndarray) -> None:
    for k in np.unique(keys[mask]):
        if k not in table:
            continue
        pos = np.nonzero(mask & (keys == k))[0]
        codes, p = table[k]
        out[pos] = rng.choice(codes, len(pos), p=p)


def apply_auth(core: pd.DataFrame, world: World, rng: np.random.Generator):
    n = len(core)
    K, M, I, C = world.cards, world.merchants, world.ips, world.customers
    countries, cities, mccs = world.countries, world.cities, world.mccs
    cred = core["cred_type"].to_numpy()
    channel = core["channel"].to_numpy()
    pos = core["pos_entry_mode"].to_numpy()
    card = core["card"].to_numpy()
    merch = core["merchant"].to_numpy()
    ip = core["ip"].to_numpy()
    cust = core["customer"].to_numpy()
    value = core["value_usd"].to_numpy()

    card_country = K["country"][card]
    m_country = cities["country_idx"].to_numpy()[M["city"][merch]]
    sca = countries["sca"].to_numpy()
    cnp = channel == "CNP"

    avs = np.full(n, None, dtype=object)
    cvc = np.full(n, None, dtype=object)
    tds = np.full(n, None, dtype=object)
    _sample_by(rng, cred, AVS, cnp, avs)
    avs[cnp & ~countries["avs_supported"].to_numpy()[card_country]] = "U"
    _sample_by(rng, cred, CVC, cnp, cvc)
    uses = cnp & (M["uses_3ds"][merch] | (sca[card_country] & sca[m_country])) & (pos != "card_on_file")
    _sample_by(rng, cred, THREE_DS, uses, tds)
    tds[cnp & (pos == "card_on_file")] = "exempt_mit"
    tds[cnp & ~uses & (pos != "card_on_file")] = "not_used"

    # observable risk inputs
    ip_ok = ip >= 0
    ip_country = np.where(ip_ok, I["country"][ip.clip(0)], -1)
    foreign_ip = ip_ok & (ip_country != card_country)
    hosting = ip_ok & (I["ip_type"][ip.clip(0)] == "hosting")
    mcc = M["mcc_idx"][merch]
    mu = np.where(mcc >= 0, mccs["mu"].to_numpy()[mcc.clip(0)], 3.9)
    sg = np.where(mcc >= 0, mccs["sigma"].to_numpy()[mcc.clip(0)], 0.9)
    mu = mu + np.where(cust >= 0, np.nan_to_num(C["spend_offset"][cust.clip(0)]), 0.0)
    z = (np.log(np.maximum(value, 0.01)) - mu) / sg
    prepaid = K["card_type"][card] == "prepaid"

    code = np.full(n, "00", dtype=object)

    def decline(mask, c):
        sel = (code == "00") & mask
        code[sel] = c if np.ndim(c) == 0 else np.asarray(c)[sel]

    u = lambda: rng.random(n)  # noqa: E731
    decline(cred == "invalid_pan", np.where(u() < 0.75, "14", "54"))
    decline((cvc == "N") & (u() < 0.9), "N7")
    decline(np.isin(tds, ["N", "R"]) & (u() < 0.95), "05")
    decline((avs == "N") & (u() < 0.15), "05")
    p59 = 0.002 + 0.03 * foreign_ip + 0.03 * hosting + 0.02 * (z > 2.5) + 0.01 * np.isin(pos, ["swipe", "keyed"])
    # realistic legit declines that look like "attack" codes: mistyped PANs (14) and expired /
    # closed cards (54) happen to ordinary customers at a rate comparable to fraud attempts
    real_pan = cred != "invalid_pan"
    decline(real_pan & cnp & (u() < 0.0018), "14")
    decline(real_pan & (u() < 0.0025), "54")
    decline(u() < p59, "59")
    p51 = 0.006 + 0.12 / (1 + np.exp(-(z - 3))) + 0.02 * prepaid
    decline(u() < p51, "51")
    decline((channel == "CP") & (mcc == mccs.index[mccs["name"] == "atm_cash"][0]) & (z > 2) & (u() < 0.3), "61")
    decline(u() < 0.004, "05")
    return avs, cvc, tds, code
