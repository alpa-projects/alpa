from . import api
from . import monkey_patch
from . import util

# Shortcut
from .api import parallelize
from .data_parallel import data_parallel, annotate_gradient
from .util import compute_bytes
