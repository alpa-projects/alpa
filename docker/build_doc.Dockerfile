FROM gcr.io/tensorflow-testing/nosla-cuda11.1-cudnn8-ubuntu18.04-manylinux2010-multipython

WORKDIR /
SHELL ["/bin/bash", "-c"]
RUN rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list
RUN apt-get update
RUN apt-get install -y coinor-cbc glpk-utils python3-virtualenv

RUN virtualenv --python=python3.8 python3.8-env
RUN source python3.8-env/bin/activate && pip install --upgrade pip \
    && pip install numpy==1.20 setuptools wheel six auditwheel \
    sphinx sphinx-rtd-theme sphinx-gallery matplotlib
COPY scripts/build_doc.sh /build_doc.sh
RUN chmod +x build_doc.sh
ENTRYPOINT ["/build_doc.sh"]
