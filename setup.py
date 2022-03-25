import os
import sys

import setuptools
from setuptools import setup, find_packages


SUPPORTED_PYTHONS = [(3, 6), (3, 7), (3, 8)]
SUPPORTED_BAZEL = (3, 2, 0)

ROOT_DIR = os.path.dirname(__file__)


# TODO(Hao): implement get_cuda_version
def get_cuda_version():
    return "111"


install_require_list = [
    "numpy",
    "cmake",
    "tqdm",
    "scipy",
    "numba",
    "pybind11",
    "ray[default]",
    "flax==0.3.6",
    f"cupy-cuda{get_cuda_version()}",
    "pulp",
    #"jax-alpa"
]

dev_require_list = [
    "prospector",
    "yapf"
]


# TODO(Hao): figure out how to build jax-alpa
def pip_run(build_ext):
    # check python version
    if tuple(sys.version_info[:2]) not in SUPPORTED_PYTHONS:
        msg = ("Detected Python version {}, which is not supported. "
               "Only Python {} are supported.").format(
            ".".join(map(str, sys.version_info[:2])),
            ", ".join(".".join(map(str, v)) for v in SUPPORTED_PYTHONS))
        raise RuntimeError(msg)


if __name__ == "__main__":
    import setuptools.command.build_ext
    class build_ext(setuptools.command.build_ext.build_ext):
        def run(self):
            return pip_run(self)

    with open(os.path.join(ROOT_DIR, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    setup(
        name="alpa",
        version="0.0.0", # TODO(Hao): change to os.environ.get('VERSION')
        author="Alpa team",
        author_email="",
        descrption="Alpa automatically parallelizes large tensor computation graphs and "
                   "runs them on a distributed cluster.",
        long_description=long_description,
        url="https://github.com/alpa-projects/alpa",
        classifiers=[
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Artificial Intelligence'
        ],
        keywords=("alpa distributed parallel machine-learning model-parallelism"
                  "gpt-3 deep-learning language-model python"),
        packages=find_packages(exclude=["playground"]),
        cmdclass={"build_ext": build_ext},
        install_requires=install_require_list,
        extra_require={
            'dev': dev_require_list,
        },
        python_requires='>=3.6',
    )