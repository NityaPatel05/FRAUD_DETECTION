"""Renders data_card.md for a simulated dataset from its metadata + validation report."""
from __future__ import annotations

import pandas as pd

from .events import EVENT_COLUMNS, EVENT_TRUTH_COLUMNS

COLUMN_DOCS = {
    "transaction_id": "Row id, increasing with event_time (the id of a refund's original purchase is in original_transaction_id).",
    "event_time": "UTC timestamp of the authorisation request.",
    "local_hour": "Hour of day where the payment happened (merchant city for card-present, IP location online, else home).",
    "customer_id": "Cardholder / account holder (sender for P2P). Null for guest checkouts. Invented card numbers belong to phantom ids.",
    "account_age_days": "Days since the account was opened (KYC). Null for guest checkouts and unknown cardholders.",
    "email_domain": "Email given at checkout / on the account (online channels only). Changes under account takeover and when legit customers change address.",
    "billing_country": "Cardholder billing country (card issuing country when the cardholder is unknown).",
    "address_id": "Hashed billing address. Households share one; some mule-ring and synthetic-identity accounts share one.",
    "card_id": "Card (PAN token). Reissued cards get a NEW card_id (see prev_card_id).",
    "prev_card_id": "The card this one replaced after a fraud block (null for first-issue cards).",
    "card_bin": "First six digits of the card number (issuer range).",
    "card_country": "Issuing country.",
    "card_type": "debit / credit / prepaid.",
    "card_network": "visa / mastercard / amex / discover.",
    "merchant_id": "Merchant (P2P rows: the transfer platform).",
    "mcc": "ISO 18245 merchant category code (4829 = P2P money transfer).",
    "merchant_country": "Merchant country.",
    "merchant_age_days": "Days since the merchant was onboarded (young merchants are ordinary: ~15% launch during the horizon).",
    "terminal_id": "Physical terminal for card-present, the merchant's virtual terminal online.",
    "channel": "CP (card present), CNP (card not present / e-commerce), P2P (account-to-account transfer).",
    "pos_entry_mode": "chip / contactless / swipe / keyed (CP); ecom / card_on_file / moto (CNP); p2p.",
    "txn_type": "purchase / refund. Refunds bypass the decision policy (always approved).",
    "recurring": "1 for a scheduled merchant-initiated charge (subscription / bill): fixed amount, monthly day-of-month.",
    "original_transaction_id": "For refunds of simulated purchases: transaction_id of the purchase (null otherwise).",
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
    "avs_result": "Address verification: Y full, A address only, Z zip only, N no match, U unavailable (CNP only). Input to the decision.",
    "cvc_check": "M match, N no match, P not provided / not processed (CNP only). Input to the decision.",
    "three_ds_result": "Y frictionless, C challenge passed, A attempted, N failed, R rejected, not_used, exempt_mit. "
                       "Reflects a step-up if the policy asked for one (then C or N).",
    "decision": "OUTCOME. Policy action: approve / step_up_3ds / review / decline (a would-be decline let through by exploration shows approve).",
    "risk_score": "OUTCOME. The policy's score in [0, 1] (null for refunds).",
    "propensity": "OUTCOME. P(logged action | features) for inverse-propensity weighting; < 1 only for exploration / stochastic policies.",
    "declined_by": "OUTCOME. none / issuer (base auth rules) / policy / sca (failed step-up) / review (analyst).",
    "liability_shift": "OUTCOME. 1 if an approved CNP row was authenticated by 3DS (liability moves to the issuer).",
    "auth_response_code": "OUTCOME. ISO 8583 field 39: 00 approved, 05 do not honour, 14 invalid card, 51 insufficient funds, "
                          "54 expired, 59 suspected fraud, 61 over limit, 1A step-up failed, N7 CVV2 failure.",
    "approved": "OUTCOME. 1 if the row was finally approved.",
    "reported_label": "LABEL. What the label pipeline says (1 fraud, 0 legit; null = no outcome). Corrupted by design; known only at label_date.",
    "label_status": "LABEL. no_dispute / unobserved / unreported / confirmed / chargeback / reported / reversed / review_confirmed / review_cleared.",
    "label_date": "LABEL. When reported_label becomes known (null for unobserved rows). May lie after the end of the data: censor at your cutoff.",
    "chargeback_date": "LABEL. Chargeback arrival for chargeback / reversed rows.",
    "dispute_reason": "LABEL. Reason code of a reported dispute (fraud_cnp, fraud_account_takeover, fraud_counterfeit, scam_push_payment, first_party_default, ...).",
    "is_fraud": "GROUND TRUTH. Alias of is_fraud_true (kept for backward compatibility; it is NOT the label).",
    "is_fraud_true": "GROUND TRUTH. 1 if the row was produced by a fraud process. Friendly-fraud rows are 0.",
    "fraud_scenario": "GROUND TRUTH. Which process produced the row ('legit' otherwise; friendly_fraud / synthetic_identity_build are not fraud rows).",
    "incident_id": "GROUND TRUTH. Attack incident (see ground_truth/incidents.parquet).",
    "ring_id": "GROUND TRUTH. Mule ring (see ground_truth/rings.parquet).",
    "gang_id": "GROUND TRUTH. Attacker group.",
}

SCENARIO_DOCS = {
    "card_testing": "Bursts of tiny online charges on one merchant from a stolen card list or a BIN enumeration "
                    "(invalid PANs -> 14/54 declines, owned by phantom customers). Passing cards enter a pool cashed out later.",
    "stolen_card_cnp": "A compromised card used online by a gang: gang devices, hosting IPs / residential proxies, "
                       "resale-friendly merchants, reshipping. Tactics (ip kind, amount band, merchant group, card age) adapt to approvals.",
    "account_takeover": "Credential stuffing -> login from a new device -> credential change -> dwell -> purchases with the stored "
                        "card and P2P pushes to mules (see events.parquet).",
    "terminal_cashout": "A crew runs counterfeit / stolen cards through one terminal for 1-4 weeks.",
    "skimming": "Cards used at a skimmed terminal are cloned and used card-present in another city 2-30 days later.",
    "mule_ring": "AMLworld-style laundering. Scam victims push P2P money to entry mules, funds hop through the ring, "
                 "exits cash out via crypto, gift cards or ATMs.",
    "synthetic_identity": "Fake identities (shared address / infrastructure) build 1.5-8 months of thin good behaviour "
                          "(rows labelled legit: fraud_scenario synthetic_identity_build) then run a spree.",
    "merchant_collusion": "A new (or turned long-standing) merchant ramps volume with stolen and complicit cards, "
                          "refund abuse, chargeback spike; the merchant then disappears.",
    "transaction_laundering": "A normal-looking front merchant processes stolen cards for an illicit merchant (round price "
                              "points, night hours, high chargeback ratio).",
    "bust_out": "NOVEL (test period only, never in training): long-standing accounts suddenly max out on high-value goods "
                "from their usual device and IP. Different from synthetic_identity: real old accounts, no shared infrastructure.",
    "friendly_fraud": "Legit approved purchases later disputed by the cardholder (is_fraud_true = 0, reported_label = 1). "
                      "Label noise by design: nothing in the row distinguishes it.",
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
    L.append(f"# Data card: synthetic payments world ({meta['generator']}, preset `{cfg.get('preset', 'realistic')}`)\n")
    L.append("**What it is.** A simulated payments system made by an agent-based, closed-loop generator "
             "(`src/simulator`): simulated entities, attacker agents that adapt to the decisions of a policy, "
             "delayed and selective labels, card reissue and an account-event stream.\n")
    L.append("**What it is for.** Demonstrating real-system capabilities IEEE-CIS cannot support (no merchant / IP / "
             "device ids, no label dates): velocity and entity features, label-maturity-aware training, exploration "
             "and propensities, action thresholds, adaptive adversaries, merchant-side fraud, account events.\n")
    L.append("**What it is NOT.** Not a measure of real-world accuracy: results show engineering, not performance on "
             "real fraud. Never joined to IEEE-CIS or used to augment it. No real customer data was used.\n")
    L.append("## Reproducibility\n")
    L.append(f"- seed `{cfg['seed']}`, {cfg['n_days']} days from {cfg['start_date']}, "
             f"{cfg['n_customers']:,} customers, {cfg['n_merchants']:,} merchants (+ local shops per city)")
    L.append(f"- rows: {meta['n_rows']:,}; account events: {meta.get('n_events', 0):,}; fraud rate (is_fraud_true) "
             f"{meta['fraud_rate']:.3%} (target {cfg['target_fraud_rate']:.3%}); decision policy `{meta.get('policy')}`")
    L.append(f"- loop stats: {meta.get('loop_stats')}")
    L.append("- regenerate: `python scripts/run_simulator.py --out <dir> --preset <name> --seed <seed>`\n")
    L.append("## Time splits (use these; never shuffle)\n")
    L.append(_md_table(pd.DataFrame([{"split": k, "from (UTC)": v[0], "to (UTC, excl.)": v[1]} for k, v in sp.items()])))
    L.append(f"\nThe novel scenario (`bust_out`) occurs only in the test split. Labels arrive late (chargebacks up to "
             f"{meta.get('label_window_days', 120)} days): at a training cutoff keep only rows with `label_date <= cutoff` "
             "and use `reported_label`, never `is_fraud_true`.\n")
    L.append("## Columns\n")
    L.append("Never features: ground truth " + ", ".join(f"`{c}`" for c in meta["ground_truth_columns"]) + ".  ")
    L.append("Labels (usable only from rows whose `label_date` is in the past): " + ", ".join(f"`{c}`" for c in meta["label_columns"]) + ".  ")
    L.append("Outcome columns (" + ", ".join(f"`{c}`" for c in meta["outcome_columns"]) + ") are the result of the decision "
             "a model sits in front of: use them only from earlier rows (e.g. decline velocity), never for the current row.\n")
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
    L.append("- `events.parquet` account events (" + ", ".join(EVENT_COLUMNS) + "; truth: " + ", ".join(EVENT_TRUTH_COLUMNS) + ")")
    L.append("- `entities/` customers, cards (prev_card_id, issued/blocked dates), merchants (open_date), terminals, devices, ips")
    L.append("- `ground_truth/` incidents, rings, ring_members, gang_infrastructure, gang_tactics, customers, merchants: evaluation only")
    L.append("- `metadata.json` config, splits, column groups; `validation.json` the checks below\n")

    if report:
        s = report["summary"]
        L.append("## Calibration table\n")
        cal = pd.DataFrame(report["calibration"])
        L.append(f"Pass rate: {report['calibration_pass_rate']:.0%} ({int(cal['pass'].sum())}/{len(cal)}).\n")
        L.append(_md_table(cal.assign(**{"pass": cal["pass"].map({True: "PASS", False: "FAIL"})})))
        L.append("\n## Validation summary\n")
        L.append(f"- fraud rate train / val / test: {s['train_fraud_rate']:.3%} / {s['val_fraud_rate']:.3%} / {s['test_fraud_rate']:.3%}")
        L.append(f"- approved fraud value / approved card value: {s['fraud_value_rate']:.4%}; CNP share of card-fraud value {s['cnp_share_of_fraud_value']:.1%}")
        L.append(f"- channel mix: {s['channel_share']}; decisions: {s['decision_share']}")
        L.append(f"- decline rate legit {s['decline_rate_legit']:.2%}, fraud {s['decline_rate_fraud']:.1%}")
        L.append("- structure checks: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in report["structure"].items()))
        lab = report["labels"]
        L.append(f"- label statuses: {lab['status_counts']}")
        L.append(f"- eventually reported as fraud: {lab['share_fraud_eventually_reported_1']:.1%} of true fraud rows; "
                 f"unreported {lab['share_fraud_unreported']:.1%}; friendly share of disputes {lab['friendly_share_of_disputes']:.1%}")
        pur = report["purity"]
        L.append(f"\n**Category purity** (no value with >= 50 rows and smoothed fraud rate > 0.9): "
                 f"{'PASS' if pur['pass'] else 'FAIL'}" + ("" if pur["pass"] else f" {pur['violations'][:10]}"))
        ce = report.get("row_only_ceiling")
        if ce:
            L.append(f"\n**Row-only ceiling** (HistGBDT on current-row fields; AUC {ce['overall_auc']:.3f}, AUC-PR "
                     f"{ce['overall_auc_pr']:.3f}; per-scenario AUC should stay < 0.9 except card_testing): "
                     f"{'PASS' if ce['pass'] else 'FAIL'}\n")
            L.append(_md_table(pd.DataFrame([{"scenario": k, **v} for k, v in ce["per_scenario"].items()])))
        a = s["amounts"]
        ref = report.get("ieee_reference")
        if ref:
            comp = [{"stat": "fraud rate", "synthetic": f"{a['fraud_rate']:.2%}", "IEEE-CIS train": f"{ref['fraud_rate']:.2%}"}]
            for lab_ in ("legit", "fraud"):
                for q in ("q25", "q50", "q75", "q90", "mean"):
                    comp.append({"stat": f"{lab_} amount {q}", "synthetic": f"{a[lab_][q]:,.2f}", "IEEE-CIS train": f"{ref[lab_][q]:,.2f}"})
            L.append("\n**Fidelity vs IEEE-CIS aggregates** (shape comparison only):\n")
            L.append(_md_table(pd.DataFrame(comp)))
        L.append("\n**Single-column leakage scan** (AUC of one column on its own, fitted on train, scored on val+test; flag >= 0.85). Top 10:\n")
        L.append(_md_table(pd.DataFrame(report["leakage"]).head(10)))
        L.append("\n**Scenario signals** (scenario vs legit rows of the same channels using the causal primitive it was designed around):\n")
        L.append(_md_table(pd.DataFrame(report["signals"])))
    L.append("\n## Known limitations\n")
    L.append("- Behaviour is parametric (Poisson sessions, lognormal amounts); attackers adapt only along four tactic dimensions.")
    L.append("- Geography is 36 cities in 15 countries; the IP-to-country map is invented.")
    L.append("- Base authorisation is a rule set, not a real issuer; reissue, account-updater and review models are simple.")
    L.append("- Calibration targets are order-of-magnitude (see the table's sources); several are assumptions, not measurements.")
    L.append("- Labels after the last day are not simulated: rows near the end have `label_date` beyond the data; censor them.")
    return "\n".join(L) + "\n"
