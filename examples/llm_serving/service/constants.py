"""Hyper params for serving Meta's OPT model."""
from enum import Enum

# Alpa serve url
ALPA_SERVE_PORT = 20001
ALPA_SERVE_URL = f"window.location.protocol + '//' + window.location.hostname + ':{ALPA_SERVE_PORT}/completions'"
#ALPA_SERVE_URL = f'"completions"'

# Generation params
NUM_BEAMS = 1
NUM_RETURN_SEQ = 1

# Authentication params
USE_RECAPTCHA = False
USE_API_KEYS = False
ALLOW_NON_KEY_ACCESS = True
KEYS_FILENAME = "/home/ubuntu/efs/alpa/examples/llm_serving/keys_file.json"

# Scheduler params
class AuthGroups(Enum):
    RECAPTCHA_USER = 1
    API_KEY_USER = 2
    NON_KEY_USER = 3

AUTH_GROUP_WEIGHTS = {
    AuthGroups.RECAPTCHA_USER: 300,
    AuthGroups.API_KEY_USER: 10,
    AuthGroups.NON_KEY_USER: 1
}
AUTH_GROUP_SCHEDULER_SCALE = 300
API_KEY_SCHEDULER_SCALE = 100
API_KEY_DEFAULT_WEIGHT = 10

# Logging params
LOGDIR = "weblogs"
