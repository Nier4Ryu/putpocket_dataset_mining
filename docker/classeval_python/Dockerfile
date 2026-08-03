FROM ubuntu:22.04

ARG PYTHON_VERSION=3.13.1
ARG PYTHON_BUILD_JOBS=16
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    coreutils \
    curl \
    git \
    jq \
    libbz2-dev \
    libffi-dev \
    libjpeg-dev \
    liblzma-dev \
    libopenblas-dev \
    libpng-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    patch \
    ripgrep \
    tk-dev \
    tree \
    uuid-dev \
    wget \
    xz-utils \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/python-build
RUN curl -fsSLO "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
  && tar -xzf "Python-${PYTHON_VERSION}.tgz" \
  && cd "Python-${PYTHON_VERSION}" \
  && ./configure --prefix=/opt/python/3.13 --with-ensurepip=install \
  && make -j "${PYTHON_BUILD_JOBS}" \
  && make install \
  && ln -sf /opt/python/3.13/bin/python3.13 /usr/local/bin/python3 \
  && ln -sf /opt/python/3.13/bin/python3.13 /usr/local/bin/python \
  && ln -sf /opt/python/3.13/bin/pip3.13 /usr/local/bin/pip \
  && cd / \
  && rm -rf /tmp/python-build

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
  && python -m pip install --no-cache-dir \
    pytest \
    beautifulsoup4 \
    gensim \
    netifaces \
    nltk \
    numpy \
    openpyxl \
    pandas \
    pillow \
    pypdf2 \
    python-docx \
    reportlab \
    scipy

ENV PATH="/opt/python/3.13/bin:${PATH}"
ENV HOME=/tmp/putpocket-home
WORKDIR /workspace
