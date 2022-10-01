"""Hyper params for serving Meta's OPT model."""

# Alpa serve url
ALPA_SERVE_PORT = 20001
ALPA_SERVE_URL = f"window.location.protocol + '//' + window.location.hostname + ':{ALPA_SERVE_PORT}/completions'"
#ALPA_SERVE_URL = f'"completions"'

# Generation params
NUM_BEAMS = 1
NUM_RETURN_SEQ = 1

# Authentication params
USE_RECAPTCHA = False
KEYS_FILENAME = "/home/ubuntu/efs/alpa/examples/llm_serving/keys_file.json"

# Logging params
LOGDIR = "weblogs"
