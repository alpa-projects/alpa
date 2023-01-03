# base docker image
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

# InfiniBand (IB) dependencies adopoted from CoreWeave's github
# https://github.com/coreweave/nccl-tests
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && \
    apt-get -qq install -y --allow-change-held-packages --no-install-recommends \
    build-essential libtool autoconf automake autotools-dev unzip \
    ca-certificates \
    wget curl openssh-server vim environment-modules \
    iputils-ping net-tools \
    libnuma1 libsubunit0 libpci-dev \
    libpmix-dev \
    datacenter-gpu-manager

# Mellanox OFED (latest)
RUN wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add -
RUN cd /etc/apt/sources.list.d/ && wget https://linux.mellanox.com/public/repo/mlnx_ofed/latest/ubuntu18.04/mellanox_mlnx_ofed.list
RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
    ibverbs-utils libibverbs-dev libibumad3 libibumad-dev librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils \
    && rm -rf /var/lib/apt/lists/*

# HPC-X (2.12)
ENV HPCX_VERSION=2.12
RUN cd /tmp && \
    wget -q -O - http://blobstore.s3.ord1.coreweave.com/drivers/hpcx-v${HPCX_VERSION}-gcc-MLNX_OFED_LINUX-5-ubuntu20.04-cuda11-gdrcopy2-nccl${HPCX_VERSION}-x86_64.tbz | tar xjf - && \
    mv hpcx-v${HPCX_VERSION}-gcc-MLNX_OFED_LINUX-5-ubuntu20.04-cuda11-gdrcopy2-nccl${HPCX_VERSION}-x86_64 /opt/hpcx

# GDRCopy userspace components (2.3)
RUN cd /tmp && \
    wget -q https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2011.4/x86/Ubuntu20.04/gdrcopy-tests_2.3-1_amd64.cuda11_4.Ubuntu20_04.deb && \
    wget -q https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2011.4/x86/Ubuntu20.04/libgdrapi_2.3-1_amd64.Ubuntu20_04.deb && \
    dpkg -i *.deb && \
    rm *.deb

# Begin auto-generated paths
ENV HPCX_DIR=/opt/hpcx
ENV HPCX_UCX_DIR=/opt/hpcx/ucx
ENV HPCX_UCC_DIR=/opt/hpcx/ucc
ENV HPCX_SHARP_DIR=/opt/hpcx/sharp
ENV HPCX_NCCL_RDMA_SHARP_PLUGIN_DIR=/opt/hpcx/nccl_rdma_sharp_plugin
ENV HPCX_HCOLL_DIR=/opt/hpcx/hcoll
ENV HPCX_MPI_DIR=/opt/hpcx/ompi
ENV HPCX_OSHMEM_DIR=/opt/hpcx/ompi
ENV HPCX_MPI_TESTS_DIR=/opt/hpcx/ompi/tests
ENV HPCX_OSU_DIR=/opt/hpcx/ompi/tests/osu-micro-benchmarks-5.8
ENV HPCX_OSU_CUDA_DIR=/opt/hpcx/ompi/tests/osu-micro-benchmarks-5.8-cuda
ENV HPCX_IPM_DIR=/opt/hpcx/ompi/tests/ipm-2.0.6
ENV HPCX_CLUSTERKIT_DIR=/opt/hpcx/clusterkit
ENV OMPI_HOME=/opt/hpcx/ompi
ENV MPI_HOME=/opt/hpcx/ompi
ENV OSHMEM_HOME=/opt/hpcx/ompi
ENV OPAL_PREFIX=/opt/hpcx/ompi
ENV PATH=/opt/hpcx/clusterkit/bin:/opt/hpcx/hcoll/bin:/opt/hpcx/ucc/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ompi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/hpcx/nccl_rdma_sharp_plugin/lib:/opt/hpcx/ucc/lib/ucc:/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib/ucx:/opt/hpcx/ucx/lib:/opt/hpcx/sharp/lib:/opt/hpcx/hcoll/lib:/opt/hpcx/ompi/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/hpcx/nccl_rdma_sharp_plugin/lib:/opt/hpcx/ompi/lib:/opt/hpcx/sharp/lib:/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:/opt/hpcx/hcoll/lib:/opt/hpcx/ompi/lib:/usr/local/cuda/lib64/stubs
ENV CPATH=/opt/hpcx/ompi/include:/opt/hpcx/ucc/include:/opt/hpcx/ucx/include:/opt/hpcx/sharp/include:/opt/hpcx/hcoll/include:
ENV PKG_CONFIG_PATH=/opt/hpcx/hcoll/lib/pkgconfig:/opt/hpcx/sharp/lib/pkgconfig:/opt/hpcx/ucx/lib/pkgconfig:/opt/hpcx/ompi/lib/pkgconfig:
# End of auto-generated paths

# install common tool & conda
RUN apt update && \
    apt install -y wget git vim screen && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    mkdir -p /opt/conda/envs/alpa && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Some of my own dev config
RUN wget --quiet https://raw.githubusercontent.com/zhisbug/RC/master/.screenrc -P /root/ && \
    wget --quiet https://raw.githubusercontent.com/zhisbug/RC/master/.vimrc -P /root/

# install conda alpa env
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create --name alpa python=3.8 -y && \
    conda activate alpa && \
    apt install coinor-cbc -y && \
    pip3 install --upgrade pip && \
    pip3 install cupy-cuda113 && \
    pip3 install alpa && \
    pip3 install jaxlib==0.3.22+cuda113.cudnn820 -f https://alpa-projects.github.io/wheels.html

# install additional deps for opt finetuning
RUN conda activate alpa && \
    pip3 install datasets && \
    pip3 install transformers && \
    pip3 install tensorflow-gpu

# Execute in Alpa conda env
ENV PATH /opt/conda/envs/alpa/bin:$PATH