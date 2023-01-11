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

# Priority queue params
# Note: Here we serve 1 background request every 50 interactive requests
INTERACTIVE_REQUESTS_PRIORITY = 50
BACKGROUND_REQUESTS_PRIORITY = 1
DEFAULT_BACKGROUND_REQUESTS_PRIORITY = 1

KEYS_PRIORITY_FILENAME = "/home/ubuntu/efs/alpa/examples/llm_serving/keys_priority_file.json"
