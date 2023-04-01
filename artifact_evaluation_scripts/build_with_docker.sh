ALPA_FOLDER_PATH="$1"
cd $ALPA_FOLDER_PATH

bash docker build -t mlsys23-ae -f ../docker/build_jaxlib.Dockerfile ../docker/

mkdir -p dist
docker run --gpus all --tmpfs /build:exec \
  --rm -v $(pwd)/dist:/dist build-jaxlib-image \
  3.8 cuda 11.1 main master

# Install the built jaxlib by:
# cd dist
# pip install jaxlib-0.3.22+cuda111.cudnn805-cp38-cp38-manylinux2014_x86_64
