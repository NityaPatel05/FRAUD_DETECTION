"""Renders data_card.md for a simulated dataset from its metadata + validation report."""
from __future__ import annotations

import pandas as pd

COLUMN_DOCS = {
    "transaction_id": "Row id, increasing with event_time.",
    "event_time": "UTC timestamp of the authorisation request.",
    "local_hour": "Hour of day where the payment happened (merchant city for card-present, IP location online, else home).",
    "customer_id": "Cardholder / account holder (sender for P2P). Null for attempts with invented card numbers.",
    "account_age_days": "Days since the account was opened (KYC). Null when the customer is unknown.",
    "email_domain": "Email given at checkout / on the account (online channels only). Changes under account takeover.",
    "billing_country": "Cardholder billing country (card issuing country when the cardholder is unknown).",
    "address_id": "Hashed billing address. Households share one; some mule-ring accounts share one.",
    "card_id": "Card (PAN token). Card-testing attacks also create never-before-seen invalid PANs.",
    "card_bin": "First six digits of the card number (issuer range).",
    "card_country": "Issuing country.",
    "card_type": "debit / credit / prepaid.",
    "card_network": "visa / mastercard / amex / discover.",
    "merchant_id": "Merchant (P2P rows: the transfer platform).",
    "mcc": "ISO 18245 merchant category code (4829 = P2P money transfer).",
    "merchant_country": "Merchant country.",
    "terminal_id": "Physical terminal for card-present, the merchant's virtual terminal online.",
    "channel": "CP (card present), CNP (card not present / e-commerce), P2P (account-to-account transfer).",
    "pos_entry_mode": "chip / contactless / swipe / keyed (CP); ecom / card_on_file / moto (CNP); p2p.",
    "amount": "Amount in the transaction currency.",
    "currency": "ISO 4217 currency of the merchant.",
    "amount_usd": "Amount converted to USD (not rounded to cents for foreign currencies, like IEEE TransactionAmt).",
    "ip": "Client IPv4 (online channels). The IP-to-country map is synthetic, not real geolocation.",
    "ip_country": "Country of the IP.",
    "ip_lat": "IP geolocation latitude.",
    "ip_lon": "IP geolocation longitude.",
    "ip_type": "residential / mobile (shared carrier NAT) / hosting (datacenter, VPN, proxy).",
    "device_id": "Persistent device fingerprint (online channels; missing for some card-on-file / phone orders).",
    "device_os": "Device operating system.",
    "browser": "Browser or app.",
    "shipping_country": "Shipping country for physical goods bought online.",
    "counterparty_customer_id": "P2P receiver.",
    "avs_result": "Address verification: Y full, A address only, Z zip only, N no match, U unavailable (CNP only).",
    "cvc_check": "M match, N no match, P not provided / not processed (CNP only).",
    "three_ds_result": "Y frictionless, C challenge passed, A attempted, N failed, R rejected, not_used, exempt_mit.",
    "auth_response_code": "ISO 8583 field 39: 00 approved, 05 do not honour, 14 invalid card, 51 insufficient funds, "
                          "54 expired, 59 suspected fraud, 61 over limit, N7 CVV2 failure. OUTCOME column.",
    "approved": "1 if auth_response_code == 00. OUTCOME column.",
    "is_fraud": "1 if the row was produced by a fraud process (or is a disputed purchase). GROUND TRUTH.",
    "label_date": "When the fraud label becomes known (chargeback, victim report, attack review). GROUND TRUTH.",
    "chargeback_date": "Chargeback date for approved card fraud. GROUND TRUTH.",
    "fraud_scenario": "Which process produced the row ('legit' otherwise). GROUND TRUTH, evaluation only.",
    "incident_id": "Attack incident (see ground_truth/incidents.parquet). GROUND TRUTH.",
    "ring_id": "Mule ring (see ground_truth/rings.parquet). GROUND TRUTH.",
}

SCENARIO_DOCS = {
    "card_testing": "Bursts of tiny online charges on one merchant from a stolen card list or a BIN enumeration "
                    "(invalid PANs -> 14/54). Cards that pass go into a pool that is cashed out later.",
    "stolen_card_cnp": "A compromised card used online by a gang: shared gang devices, hosting IPs or residential "
                       "proxies, resale-friendly merchants, reshipping addresses, AVS/CVC from what was stolen.",
    "account_takeover": "Login from a new device and IP, email changed, stored card used for high-value goods, "
                        "money pushed by P2P to mule accounts.",
    "terminal_cashout": "Handbook-style compromised terminal: a crew runs counterfeit / stolen cards through one "
                        "terminal for 1-4 weeks (terminal risk with label delay).",
    "skimming": "Cards used at a skimmed terminal are cloned and used card-present in another city 2-30 days "
                "later (common point of purchase, impossible travel).",
    "mule_ring": "AMLworld-style laundering. Scam victims send P2P money to entry mules, funds hop through the "
                 "ring (chain / scatter-gather / gather-scatter / cycle / fan-out), exits cash out via crypto, "
                 "gift cards or ATMs. Rings share devices, IPs and sometimes addresses; half the mules are new accounts.",
    "bust_out": "NOVEL (test period only, never used for feature design): long-standing accounts suddenly max out "
                "on high-value goods from their usual device and IP.",
    "friendly_fraud": "Legit approved purchases later disputed by the cardholder. Label noise by design: nothing in "
                      "the row distinguishes it.",
}


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def render(meta: dict, report: dict | None) -> str:
    cfg = meta["config"]
    sp = meta["splits"]
    L = []
    L.append(f"# Data card: synthetic payments world ({meta['generator']})\n")
    L.append("**What it is.** A simulated payments dataset made by an agent-based generator "
             "(`src/simulator`). Customers, merchants, terminals, devices and IPs are simulated entities; "
             "fraud rows are produced by attacker agents, not stamped onto rows after generation.\n")
    L.append("**What it is for.** Demonstrating real-world fraud components that IEEE-CIS cannot support "
             "because the fields don't exist there: IP / device velocity, impossible travel, merchant and "
             "terminal risk with label delay, mule-ring detection against ground-truth ring ids, and feeding "
             "the live Kafka -> Redis -> FastAPI path a realistic payload.\n")
    L.append("**What it is NOT.** Not a measure of real-world accuracy: results here show engineering, "
             "not performance on real fraud. Not for augmenting or balancing IEEE-CIS, and never joined to it. "
             "No real customer data was used. IEEE-CIS was read only for aggregate statistics.\n")

    L.append("## Reproducibility\n")
    L.append(f"- seed `{cfg['seed']}`, {cfg['n_days']} days from {cfg['start_date']}, "
             f"{cfg['n_customers']:,} customers, {cfg['n_merchants']:,} merchants (+ local shops per city)")
    L.append(f"- rows: {meta['n_rows']:,}; fraud rate {meta['fraud_rate']:.2%} (target {cfg['target_fraud_rate']:.2%})")
    L.append("- regenerate: `python scripts/run_simulator.py --out <dir> --seed <seed>` (same seed + config = same data)\n")

    L.append("## Time splits (use these; never shuffle)\n")
    L.append(_md_table(pd.DataFrame([{"split": k, "from (UTC)": v[0], "to (UTC, excl.)": v[1]} for k, v in sp.items()])))
    L.append("\nThe novel scenario (`bust_out`) occurs only in the test split. Labels arrive late: use "
             "`label_date` (not `is_fraud`) when building label-history features, and at a training cutoff "
             "treat rows whose `label_date` is after the cutoff as unlabelled.\n")

    L.append("## Columns\n")
    L.append("Never use as features: " + ", ".join(f"`{c}`" for c in meta["ground_truth_columns"]) + ".  ")
    L.append("Outcome columns (" + ", ".join(f"`{c}`" for c in meta["outcome_columns"]) + ") are the result of the "
             "authorisation decision that a fraud model sits in front of. Use them only from earlier rows "
             "(e.g. a card's decline count in the last hour), never for the current row.\n")
    L.append(_md_table(pd.DataFrame([{"column": k, "meaning": v} for k, v in COLUMN_DOCS.items()])))

    L.append("\n## Fraud scenarios\n")
    rows = []
    stats = {r["fraud_scenario"]: r for r in (report or {}).get("summary", {}).get("scenarios", [])}
    for s, doc in SCENARIO_DOCS.items():
        st = stats.get(s, {})
        rows.append({"scenario": s, "rows": st.get("rows", ""), "incidents": st.get("incidents", ""),
                     "approved": st.get("approved", ""), "description": doc})
    L.append(_md_table(pd.DataFrame(rows)))

    L.append("\n## Other files\n")
    L.append("- `entities/` customers, cards, merchants, terminals, devices, ips (public attributes only)")
    L.append("- `ground_truth/` incidents, rings, ring_members, gang_infrastructure, customers (is_mule): "
             "evaluation only")
    L.append("- `metadata.json` config, splits, column groups; `validation.json` / `validation.md` the checks below\n")

    if report:
        s = report["summary"]
        L.append("## Validation summary\n")
        L.append(f"- fraud rate train / val / test: {s['train_fraud_rate']:.2%} / {s['val_fraud_rate']:.2%} / "
                 f"{s['test_fraud_rate']:.2%}")
        L.append(f"- channel mix: {s['channel_share']}; device coverage {s['device_coverage']:.1%}")
        L.append(f"- decline rate legit {s['decline_rate_legit']:.1%}, fraud {s['decline_rate_fraud']:.1%}")
        L.append("- structure checks: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in report["structure"].items()))
        a = s["amounts"]
        ref = report.get("ieee_reference")
        comp = [{"stat": "fraud rate", "synthetic": f"{a['fraud_rate']:.2%}",
                 "IEEE-CIS train": f"{ref['fraud_rate']:.2%}" if ref else "n/a"}]
        for lab in ("legit", "fraud"):
            for q in ("q25", "q50", "q75", "q90", "mean", "share_gt2_decimals"):
                v = a[lab][q]
                r_ = ref[lab][q] if ref else None
                fmt = (lambda x: f"{x:.1%}") if q == "share_gt2_decimals" else (lambda x: f"{x:,.2f}")
                comp.append({"stat": f"{lab} amount {q}", "synthetic": fmt(v), "IEEE-CIS train": fmt(r_) if r_ is not None else "n/a"})
        L.append("\n**Fidelity vs IEEE-CIS aggregates** (IEEE is online-only; this world also has card-present and "
                 "P2P, so amounts are comparable in shape, not identical):\n")
        L.append(_md_table(pd.DataFrame(comp)))
        L.append("\n**Single-column leakage scan** (AUC of one column on its own, fitted on train, scored on "
                 "val+test; flag at >= 0.85). Top 10:\n")
        L.append(_md_table(pd.DataFrame(report["leakage"]).head(10)))
        L.append("\n**Scenario signals** (each scenario vs legit rows of the same channels, using the causal "
                 "primitive it was designed around; friendly fraud is expected near 0.5):\n")
        L.append(_md_table(pd.DataFrame(report["signals"])))

    L.append("\n## Known limitations\n")
    L.append("- Behaviour is parametric (Poisson sessions, lognormal amounts, scripted attackers). Real fraud "
             "adapts to the defences in place; here it does not.")
    L.append("- Geography is 36 cities in 15 countries; the IP-to-country map is invented.")
    L.append("- Authorisation is a simple rule set (CVC/3DS/AVS/issuer risk), not a real issuer.")
    L.append("- No seasonality beyond the weekly and daily cycles; no holiday peaks.")
    L.append("- Amount shape is calibrated loosely to IEEE-CIS; this world has card-present and P2P traffic "
             "IEEE does not have.")
    return "\n".join(L) + "\n"
