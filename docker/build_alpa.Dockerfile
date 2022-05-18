FROM quay.io/pypa/manylinux2014_x86_64

WORKDIR /
SHELL ["/bin/bash", "-c"]
RUN yum update
RUN yum-config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
RUN yum --enablerepo=epel -y install cuda-11-6
#WORKDIR /
#RUN CUDA_VERSION=11.1
#RUN CUBLAS=libcublas-11-1
#RUN CUBLAS_DEV=libcublas-dev-11-1
#RUN CUDNN_VERSION=8.0.5.39
#RUN LIBCUDNN=libcudnn8
#RUN yum update
#RUN yum install libcublas-11-1 libcublas-dev-11-1
#    cuda-nvml-dev-$CUDA_VERSION \
#    cuda-command-line-tools-$CUDA_VERSION \
#    cuda-libraries-dev-$CUDA_VERSION \
#    cuda-minimal-build-$CUDA_VERSION \
#    $LIBCUDNN=$CUDNN_VERSION-1+cuda$CUDA_VERSION \
#    $LIBCUDNN-dev=$CUDNN_VERSION-1+cuda$CUDA_VERSION
#
#RUN rm -f /usr/local/cuda
#RUN ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda


COPY scripts/build_alpa.sh /build_alpa.sh
RUN chmod +x /build_alpa.sh

WORKDIR /build
ENV TEST_TMPDIR /build
ENTRYPOINT ["/build_alpa.sh"]
