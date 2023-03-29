import glob
import os
import re
import shutil
import subprocess
import sys

from setuptools import setup, find_packages

IS_WINDOWS = sys.platform == "win32"
ROOT_DIR = os.path.dirname(__file__)
HAS_CUDA = os.system("nvidia-smi > /dev/null 2>&1") == 0


def get_cuda_version(cuda_home):
    """Locate the CUDA version."""
    version_file = os.path.join(cuda_home, "version.txt")
    try:
        if os.path.isfile(version_file):
            with open(version_file, "r") as f_version:
                version_str = f_version.readline().replace("\n", "").replace(
                    "\r", "")
                return version_str.split(" ")[2][:4]
        else:
            version_str = subprocess.check_output(
                [os.path.join(cuda_home, "bin", "nvcc"), "--version"])
            version_str = str(version_str).replace("\n", "").replace("\r", "")
            idx = version_str.find("release")
            return version_str[idx + len("release "):idx + len("release ") + 4]
    except RuntimeError:
        raise RuntimeError("Cannot read cuda version file")


def locate_cuda():
    """Locate the CUDA environment on the system."""
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        try:
            which = "where" if IS_WINDOWS else "which"
            nvcc = subprocess.check_output([which,
                                            "nvcc"]).decode().rstrip("\r\n")
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except subprocess.CalledProcessError:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*")
                if len(cuda_homes) == 0:
                    cuda_home = ""
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = "/usr/local/cuda"
            if not os.path.exists(cuda_home):
                cuda_home = None
    version = get_cuda_version(cuda_home)
    cudaconfig = {
        "home":
            cuda_home,
        "include":
            os.path.join(cuda_home, "include"),
        "lib64":
            os.path.join(cuda_home,
                         os.path.join("lib", "x64") if IS_WINDOWS else "lib64"),
    }
    if not all([os.path.exists(v) for v in cudaconfig.values()]):
        raise EnvironmentError(
            "The CUDA  path could not be located in $PATH, $CUDA_HOME or $CUDA_PATH. "
            "Either add it to your path, or set $CUDA_HOME or $CUDA_PATH.")

    return cudaconfig, version


def get_cuda_version_str(no_dot=False):
    """Return the cuda version in the format of [x.x]."""
    ver = locate_cuda()[1]
    if no_dot:
        ver = ver.replace(".", "")
    return ver


install_require_list = [
    "tqdm",
    "ray",
    "jax==0.3.22",
    "chex==0.1.5",
    "flax==0.6.2",
    "pulp>=2.6.0",
    "numpy>=1.20",
    "numba",
]

dev_require_list = ["yapf==0.32.0", "pylint==2.14.0", "cmake", "pybind11"]

if HAS_CUDA:
    dev_require_list += [
        f"cupy-cuda{get_cuda_version_str(no_dot=True)}",
    ]

doc_require_list = [
    "sphinx", "sphinx-rtd-theme", "sphinx-gallery", "matplotlib"
]


def get_alpa_version():
    with open(os.path.join(ROOT_DIR, "alpa", "version.py")) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":
    import setuptools
    from setuptools.command.install import install

    class BinaryDistribution(setuptools.Distribution):

        def has_ext_modules(self):
            return False

    class InstallPlatlib(install):

        def finalize_options(self):
            install.finalize_options(self)
            if self.distribution.has_ext_modules():
                self.install_lib = self.install_platlib

    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()

    setup(
        name="alpa",
        version=get_alpa_version(),
        author="Alpa Developers",
        author_email="",
        description=
        "Alpa automatically parallelizes large tensor computation graphs and "
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
        packages=find_packages(
            exclude=["benchmark", "examples", "playground", "tests"]),
        python_requires='>=3.7',
        cmdclass={"install": InstallPlatlib},
        install_requires=install_require_list,
        extras_require={
            'dev': dev_require_list,
            'doc': doc_require_list + dev_require_list,
        },
    )
