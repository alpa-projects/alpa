"""Hyper params for serving Meta's OPT model."""

MAX_SEQ_LEN = 512
MAX_BATCH_TOKENS = 3072
TIMEOUT_MS = 2000
LOGDIR = "weblogs"
MAX_BS = 8

# Generation-related params
NUM_BEAMS = 1
NUM_RETURN_SEQ = 1
