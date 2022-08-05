#!/bin/bash
set -xe

install_torch_deps() {
    # NOTE: functorch is pinned to the last commit that works with PyTorch 1.12
    pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==1.12 torchdistx && \
    ([ -d "functorch" ] || git clone https://github.com/pytorch/functorch) && \
    pushd functorch && git checkout 091d9999b16bf0015b735971580be2d9ad704144 && python setup.py install && popd
}

install_torch_deps
