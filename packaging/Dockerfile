FROM quay.io/pypa/manylinux2014_x86_64
LABEL authors="Omar Awile, Pramod Kumbhar, Alexandru Savulescu"

# install basic packages
RUN yum -y install \
    git \
    wget \
    make \
    vim \
    curl \
    unzip \
    autoconf \
    automake \
    make \
    openssh-server \
    libtool \
    && yum -y clean all \
    && rm -rf /var/cache

WORKDIR /root

RUN curl -L -o flex-2.6.4.tar.gz https://github.com/westes/flex/files/981163/flex-2.6.4.tar.gz \
    && tar -xvzf flex-2.6.4.tar.gz \
    && cd flex-2.6.4 \
    && ./configure --prefix=/nmodlwheel/flex \
    && make -j 3 install \
    && cd .. \
    && rm -rf flex-2.6.4.tar.gz flex-2.6.4

RUN curl -L -o bison-3.7.3.tar.gz https://ftp.gnu.org/gnu/bison/bison-3.7.3.tar.gz \
    && tar -xvzf bison-3.7.3.tar.gz \
    && cd bison-3.7.3 \
    && ./configure --prefix=/nmodlwheel/bison \
    && make -j 3 install \
    && cd .. \
    && rm -rf bison-3.7.3.tar.gz bison-3.7.3

ENV PATH /nmodlwheel/flex/bin:/nmodlwheel/bison/bin:$PATH

# Copy Dockerfile for reference
COPY Dockerfile .

# build wheels from there
WORKDIR /root
