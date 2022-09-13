"""Hyper params for serving Meta's OPT model."""

MAX_SEQ_LEN = 1024
MAX_BATCH_TOKENS = 3072
BATCHING_TIMEOUT_MS = 2000
LOGDIR = "weblogs"
MAX_BS = 4

# Generation-related params
NUM_BEAMS = 1
NUM_RETURN_SEQ = 1

# Logprobs endpoint cache-related params
LOGPROBS_PAST_CACHE_TIMEOUT = 60 * 10 # 10 minutes
LOGPROBS_PAST_CACHE_SIZE_LIMIT = 4 # max sets of past_key_values to keep at a time in cache
