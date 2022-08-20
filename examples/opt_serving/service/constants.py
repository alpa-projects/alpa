"""Hyper params for serving Meta's OPT model."""

MAX_SEQ_LEN = 512
BATCH_SIZE = 2048  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 3072
TIMEOUT_MS = 2000
DEFAULT_PORT = 10001
LOGDIR = "weblogs"
MAX_BS = 8
