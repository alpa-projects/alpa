import os
import subprocess
import sys

from setuptools import setup, find_packages


SUPPORTED_PYTHONS = [(3, 7), (3, 8), (3, 9)]

ROOT_DIR = os.path.dirname(__file__)


# TODO(Hao): implement get_cuda_version
def get_cuda_version():
    return "116"


install_require_list = [
    "numpy",
    "cmake",
    "tqdm",
    "scipy",
    "numba",
    "pybind11",
    "ray[default]",
    "flax==0.4.1",
    f"cupy-cuda{get_cuda_version()}",
    "pulp",
    #"jax-alpa"
]

dev_require_list = [
    "prospector",
    "yapf"
]


def build(build_ext):
    """Build the custom pipeline marker API."""
    # Check cuda version
    build_command = []
    if "CUDACXX" in os.environ and os.path.exists(os.environ["CUDACXX"]):
        cudacxx_path = os.environ["CUDACXX"]
    else:
        # infer CUDACXX
        cuda_version = get_cuda_version()
        cuda_version = cuda_version[:-1] + "." + cuda_version[:-1]
        print(f"CUDA version: {cuda_version}")
        cudacxx_path = f"/usr/local/cuda-{cuda_version}/bin/nvcc"
        if not os.path.exists(cudacxx_path):
            raise ValueError("Cannot find CUDACXX compiler.")

    build_command += [f"CUDACXX={cudacxx_path} "]
    print(build_command)
    # Enter the folder and build
    build_command += [f"cd {ROOT_DIR}/alpa/pipeline_parallel/xla_custom_call_marker; "]
    build_command += [f"CUDACXX={cudacxx_path} ./build.sh"]
    print(build_command)
    if subprocess.call(build_command, shell=True) != 0:
        exit(-1)


if __name__ == "__main__":
    import setuptools
    import setuptools.command.build_ext

    class build_ext(setuptools.command.build_ext.build_ext):
        def run(self):
            return build(self)

    class BinaryDistribution(setuptools.Distribution):
        def has_ext_modules(self):
            return True

    with open(os.path.join(ROOT_DIR, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    setup(
        name="alpa",
        version="0.0.0", # TODO(Hao): change to os.environ.get('VERSION')
        author="Alpa team",
        author_email="",
        description="Alpa automatically parallelizes large tensor computation graphs and "
                   "runs them on a distributed cluster.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/alpa-projects/alpa",
        classifiers=[
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Artificial Intelligence'
        ],
        keywords=("alpa distributed parallel machine-learning model-parallelism"
                  "gpt-3 deep-learning language-model python"),
        packages=find_packages(exclude=["playground"]),
        python_requires='>=3.7',
        cmdclass={"build_ext": build_ext},
        distclass=BinaryDistribution,
        install_requires=install_require_list,
        extra_requires={
            'dev': dev_require_list,
        },
    )
