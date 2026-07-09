# src/jobs/__init__.py
# Background job package for the fraud detection pipeline.
#
# Job 1: gnn_refresh    — periodic GNN embedding refresh (Neo4j → GraphSAGE → Redis)
# Job 2: post_inference — per-transaction fire-and-forget tasks (Redis, Kafka, ring check)
