"""Hyper params for serving Meta's OPT model."""

MAX_SEQ_LEN = 512
MAX_BATCH_TOKENS = 3072
BATCHING_TIMEOUT_MS = 2000
LOGDIR = "weblogs"
MAX_BS = 8

# Generation-related params
NUM_BEAMS = 1
NUM_RETURN_SEQ = 1

# Logprobs endpoint cache-related params
LOGPROBS_BATCHING_TIMEOUT_MS = 0
LOGPROBS_PAST_CACHE_TIMEOUT = 60 * 10 # 10 minutes
LOGPROBS_PAST_CACHE_SIZE_LIMIT = 8 # max sets of past_key_values to keep at a time in cache
