"""Self-supervised per-client transaction-sequence encoder (Phase D). See docs/SEQUENCE_EMBEDDINGS.md."""
from src.sequence.data import SeqSpec, Encoded, client_order, DEFAULT_CAT_COLS
from src.sequence.model import SequenceEncoder, SeqConfig, ClientState, FEATURE_PREFIX, SURPRISE_COLS

__all__ = ["SeqSpec", "Encoded", "client_order", "DEFAULT_CAT_COLS", "SequenceEncoder", "SeqConfig",
           "ClientState", "FEATURE_PREFIX", "SURPRISE_COLS"]
