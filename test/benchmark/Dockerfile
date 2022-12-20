FROM ubuntu:22.04

RUN su -

RUN apt update

RUN apt install -y wget gpg

RUN apt install -y build-essential

# Intel compilers

# download the key to system keyring
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# add signed entry to apt sources and configure the APT client to use Intel repository:
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

RUN apt update

RUN apt install -y intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic

RUN . /opt/intel/oneapi/setvars.sh && \
gcc --version && g++ --version && \
icc --version && icpc --version

RUN apt install -y libsleef3

RUN apt install -y lsb-release software-properties-common gnupg

RUN wget https://apt.llvm.org/llvm.sh && \
chmod +x llvm.sh && \
./llvm.sh 13

RUN which clang++-13

RUN wget https://developer.download.nvidia.com/hpc-sdk/22.11/nvhpc_2022_2211_Linux_x86_64_cuda_11.8.tar.gz && \
tar xpzf nvhpc_2022_2211_Linux_x86_64_cuda_11.8.tar.gz && \
nvhpc_2022_2211_Linux_x86_64_cuda_11.8/install

RUN rm -rf nvhpc_2022_2211_Linux_x86_64_cuda_11.8.tar.gz

RUN apt install -y gfortran

RUN /opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin/makelocalrc /opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin -v -x /opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin

# ENV PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin:$PATH"

# RUN which nvc++

ENV svml_lib=/opt/intel/oneapi/compiler/2023.0.0/linux/compiler/lib/intel64_lin/libsvml.so
ENV intel_exe=/opt/intel/oneapi/compiler/2023.0.0/linux/bin/intel64/icpc
ENV sleef_lib=/lib/x86_64-linux-gnu/libsleefgnuabi.so.3.5
ENV clang_exe=/usr/bin/clang++-13
ENV llc_exe=/usr/bin/llc-13
ENV gcc_exe=/usr/bin/g++
ENV nvhpc_exe=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin/nvc++
ENV libdevice_lib=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/nvvm/libdevice/libdevice.10.bc

RUN apt install -y git

RUN git clone --recursive -b magkanar/python_benchmark https://github.com/BlueBrain/nmodl.git

RUN apt install -y cmake flex bison

RUN apt install -y python3-pip

RUN pip install Jinja2 PyYAML sympy pytest

RUN apt install -y libomp-13-dev

RUN echo ". /opt/intel/oneapi/setvars.sh" >> $HOME/.bashrc

# RUN cd nmodl && \
# mkdir build && \
# cd build && \
# cmake .. \
#   -DCMAKE_C_COMPILER=$(which clang-13) \
#   -DCMAKE_CXX_COMPILER=$(which clang++-13) \
#   -DNMODL_ENABLE_LLVM=ON \
#   -DNMODL_ENABLE_LLVM_GPU=ON \
#   -DNMODL_ENABLE_LLVM_CUDA=ON \
#   -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/bin/nvcc \
#   -DCMAKE_INSTALL_PREFIX=./install && \
# cmake --build . --target install --parallel 10

RUN cd nmodl && \
mkdir build && \
cd build && \
cmake .. \
  -DCMAKE_C_COMPILER=$(which clang-13) \
  -DCMAKE_CXX_COMPILER=$(which clang++-13) \
  -DNMODL_ENABLE_LLVM=ON \
  -DCMAKE_INSTALL_PREFIX=./install && \
cmake --build . --target install
