"""
A global context to pass arguments from python to XLA c++ passes.
"""

from jax.lib import xla_extension as _xla

class XlaPassContext(object):
    current = None

    def __init__(self, value_dict):
        self.value_dict = value_dict
      
    def __enter__(self):
        assert XlaPassContext.current is None, "Do not support recurrent context"
        XlaPassContext.current = self
        _xla.set_pass_context(self.value_dict)

    def __exit__(self ,type, value, traceback):
        XlaPassContext.current = None
        _xla.clear_pass_context()

