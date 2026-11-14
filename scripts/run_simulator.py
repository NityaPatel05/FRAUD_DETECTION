"""
Generate the synthetic payments world (Track C) and its data card.

    venv/Scripts/python.exe scripts/run_simulator.py --out data/synthetic/v1
    venv/Scripts/python.exe scripts/run_simulator.py --small --out data/synthetic/smoke
    venv/Scripts/python.exe scripts/run_simulator.py --config configs/config.yaml   # uses a `simulator:` section if present

Writes <out>/transactions.parquet, entities/, ground_truth/, metadata.json,
validation.json and data_card.md. Nothing here trains a model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import SimConfig, simulate  # noqa: E402
from src.simulator.datacard import render  # noqa: E402
from src.simulator.validate import ieee_reference, validate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic/v1")
    ap.add_argument("--config", help="YAML file; its `simulator:` section overrides defaults")
    ap.add_argument("--small", action="store_true", help="quick world (~40k rows)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n-customers", type=int)
    ap.add_argument("--n-days", type=int)
    ap.add_argument("--ieee-csv", default="data/raw/train_transaction.csv",
                    help="IEEE-CIS train_transaction.csv for aggregate fidelity stats ('' to skip)")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    overrides = {}
    if args.config:
        import yaml
        overrides.update((yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}).get("simulator") or {})
    for k in ("seed", "n_customers", "n_days"):
        v = getattr(args, k)
        if v is not None:
            overrides[k] = v
    cfg = SimConfig.small(**overrides) if args.small else SimConfig.from_dict(overrides)

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    t0 = time.time()
    res = simulate(cfg, verbose=True)
    print(f"[sim] generated {len(res.transactions):,} rows in {time.time() - t0:.0f}s, "
          f"fraud rate {res.metadata['fraud_rate']:.2%}")
    res.save(out)
    # keep generated data out of git (data/raw etc. are DVC-tracked; this folder is not)
    gi = out.parent / ".gitignore"
    if out.parent.name == "synthetic" and not gi.exists():
        gi.write_text("*\n!.gitignore\n")

    report = None
    if not args.no_validate:
        ref = None
        ieee = ROOT / args.ieee_csv if args.ieee_csv else None
        if ieee and ieee.exists():
            cache = out.parent / "ieee_reference.json"
            if cache.exists():
                ref = json.loads(cache.read_text())
            else:
                print("[sim] reading IEEE-CIS aggregates (amount, isFraud, time only)...")
                ref = ieee_reference(ieee)
                cache.write_text(json.dumps(ref, indent=2))
        t1 = time.time()
        report = validate(res, ref)
        (out / "validation.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"[sim] validation done in {time.time() - t1:.0f}s; structure: {report['structure']}")
        flagged = [r["column"] for r in report["leakage"] if r["flag"]]
        print(f"[sim] leakage flags (single-column AUC >= 0.85): {flagged or 'none'}")
    (out / "data_card.md").write_text(render(res.metadata, report), encoding="utf-8")
    print(f"[sim] wrote {out}")


if __name__ == "__main__":
    main()
