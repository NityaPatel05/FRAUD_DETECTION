
## 2026-11-17 16:29 - Phase D sequence embeddings (causal_raw, pretrain=first_cutoff, 3 seeds)

- seq_all: AP 0.3344 vs base 0.3436 delta -0.0092 +/- 0.0069 (SE, 5 folds x 3 seeds) -> drop (gain below threshold)
- surprise_only: AP 0.3290 vs base 0.3436 delta -0.0146 +/- 0.0082 (SE, 5 folds x 3 seeds) -> drop (gain below threshold)
- emb_only: AP 0.3269 vs base 0.3436 delta -0.0167 +/- 0.0088 (SE, 5 folds x 3 seeds) -> drop (gain below threshold)

## 2026-11-17 16:31 - Phase D sequence embeddings (causal_raw, pretrain=first_cutoff, 3 seeds)

- seq_all: AP 0.3339 vs base 0.3436 delta -0.0097 +/- 0.0100 (SE, 5 folds x 3 seeds) -> drop (gain below threshold)
- surprise_only: AP 0.3269 vs base 0.3436 delta -0.0167 +/- 0.0109 (SE, 5 folds x 3 seeds) -> drop (gain below threshold)
- emb_only: AP 0.3257 vs base 0.3436 delta -0.0179 +/- 0.0088 (SE, 5 folds x 3 seeds) -> HURTS (removing it helps)
