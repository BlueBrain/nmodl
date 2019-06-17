FROM alpine:latest AS builder

WORKDIR /nmodl/src

RUN apk add --update build-base gcc g++ make cmake flex flex-dev bison git python3-dev

RUN pip3 install --trusted-host pypi.python.org jinja2 pyyaml pytest sympy


RUN git clone --recursive https://github.com/BlueBrain/nmodl.git && \
    cd nmodl && \
    git checkout 0.2

WORKDIR /nmodl/src/nmodl

RUN pip3 install .


FROM alpine:latest


RUN apk add --no-cache --update python3 libgfortran libstdc++ openblas && \
    apk add --no-cache --virtual build-dependencies \
            build-base linux-headers openblas-dev freetype-dev \
            pkgconfig gfortran python3-dev && \
    pip3 install --no-cache-dir --trusted-host pypi.python.org \
                 jinja2 pyyaml pytest sympy numpy matplotlib jupyter && \
    apk del build-dependencies && \
    rm -rf /var/cache/apk/*
    
WORKDIR /usr/lib/python3.6/site-packages

COPY --from=builder /usr/lib/python3.6/site-packages/nmodl .

EXPOSE 8888
WORKDIR /nmodl

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]


