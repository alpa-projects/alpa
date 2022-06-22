import sys
from setuptools import Extension, find_packages, setup


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]

extensions = [
    NumpyExtension(
        "dataset.token_block_utils_fast",
        sources=["dataset/token_block_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "dataset.data_utils_fast",
        sources=["dataset/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

cmdclass = {}

try:
    # torch is not available when generating docs
    from torch.utils import cpp_extension

    cmdclass["build_ext"] = cpp_extension.BuildExtension

except ImportError:
    pass

setup(ext_modules=extensions, cmdclass=cmdclass)
