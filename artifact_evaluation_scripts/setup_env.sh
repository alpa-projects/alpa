ALPA_FOLDER_PATH="$1"

# Building the jaxlib
cd $ALPA_FOLDER_PATH
cd build_jaxlib
python build/build.py --enable_cuda --bazel_options=--override_repository=org_tensorflow=$(pwd)/../third_party/tensorflow-alpa
# Installing the built jaxlib
cd dist
pip install -e .
# Installing jax
cd $ALPA_FOLDER_PATH
cd third_party/jax
pip install -e .
