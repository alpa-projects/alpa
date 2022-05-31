FROM quay.io/pypa/manylinux2014_x86_64

WORKDIR /
SHELL ["/bin/bash", "-c"]
RUN yum-config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
RUN yum --enablerepo=epel -y install cuda-11-1

COPY scripts/build_alpa.sh /build_alpa.sh
RUN chmod +x /build_alpa.sh

WORKDIR /build
ENV TEST_TMPDIR /build
ENTRYPOINT ["/build_alpa.sh"]
