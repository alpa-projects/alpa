"""Utility for testing"""

class TestingEnv:
    def __init__(self):
        self.last_compiled_executable = None

testing_env = TestingEnv()


def last_compiled_executable():
    return testing_env.last_compiled_executable


def set_last_compiled_executable(executable):
    testing_env.last_compiled_executable = executable

