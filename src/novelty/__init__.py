"""
Phase 6 — catching NEW fraud patterns (replaces the VAE anomaly score).

Production practice (Uber RADAR, Stripe card-testing defence, Feedzai SAMM, Monzo):
new attacks show up as bursts in some slice of traffic long before chargebacks
arrive. So instead of a per-transaction anomaly score, we:
  1. detect anomalous segments over time          (attack_detection.py)
  2. mine the feature=value patterns behind them  (rule_mining.py)
  3. backtest and run them as expiring rules      (rules_engine.py)
  4. monitor drift without waiting for labels     (drift.py)
  5. surface unusual clusters for analysts        (discovery.py)
"""
