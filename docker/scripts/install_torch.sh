#!/bin/bash
set -xe

install_torch_deps() {
    # NOTE: functorch is pinned to the last commit that works with PyTorch 1.12
    pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==1.12 torchdistx && \
    ([ -d "functorch" ] || git clone https://github.com/pytorch/functorch) && \
    pushd functorch && git checkout 76976db8412b60d322c680a5822116ba6f2f762a && python setup.py install && popd
}

install_torch_deps
