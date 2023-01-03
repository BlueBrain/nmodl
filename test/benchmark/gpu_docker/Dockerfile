FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN su -

RUN apt update

RUN apt install -y wget gpg vim

RUN apt install -y build-essential

# Install Intel Compiler Classic
# Download the key to system keyring
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# Add signed entry to apt sources and configure the APT client to use Intel repository:
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

# Update apt
RUN apt update

# Install only Intel Compiler Classic and DPC++ compilers
RUN apt install -y intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic-2022.2.1

# Install sleef library 3.5.1
RUN apt install -y libsleef3

# Install LLVM 13
# First install dependencies
RUN apt install -y lsb-release software-properties-common gnupg

# Install LLVM 13
RUN wget https://apt.llvm.org/llvm.sh && \
chmod +x llvm.sh && \
./llvm.sh 13

# Install libomp for LLVM 13
RUN apt install -y libomp-13-dev

# Install NVHPC 22.3
RUN wget https://developer.download.nvidia.com/hpc-sdk/22.3/nvhpc_2022_223_Linux_x86_64_cuda_11.6.tar.gz && \
tar xpzf nvhpc_2022_223_Linux_x86_64_cuda_11.6.tar.gz && \
nvhpc_2022_223_Linux_x86_64_cuda_11.6/install && \
rm -rf nvhpc_2022_223_Linux_x86_64_cuda_11.6.tar.gz nvhpc_2022_223_Linux_x86_64_cuda_11.6

# Needed to run makelocalrc
RUN apt install -y gfortran

# Run makelocalrc to set up nvc++, nvc
RUN /opt/nvidia/hpc_sdk/Linux_x86_64/22.3/compilers/bin/makelocalrc /opt/nvidia/hpc_sdk/Linux_x86_64/22.3/compilers/bin -v -x /opt/nvidia/hpc_sdk/Linux_x86_64/22.3/compilers/bin

# Set up necessary environment variables to pass to the benchmark script
ENV svml_lib=/opt/intel/oneapi/compiler/2022.2.1/linux/compiler/lib/intel64_lin/libsvml.so
ENV intel_exe=/opt/intel/oneapi/compiler/2022.2.1/linux/bin/intel64/icpc
ENV sleef_lib=/lib/x86_64-linux-gnu/libsleefgnuabi.so.3.5
ENV clang_exe=/usr/bin/clang++-13
ENV llc_exe=/usr/bin/llc-13
ENV gcc_exe=/usr/bin/g++
ENV nvhpc_exe=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/compilers/bin/nvc++
ENV libdevice_lib=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/11.6/nvvm/libdevice/libdevice.10.bc

# Install prerequisites necessary to build NMODL
RUN apt install -y git cmake flex bison python3-pip
RUN pip install Jinja2 PyYAML sympy pytest

# Clone NMODL branch for benchmarking LLVM
RUN git clone --recursive --branch mod2ir-cc23 --depth 1 https://github.com/BlueBrain/nmodl.git

# Setup Intel compiler specific variables to the environment
RUN echo ". /opt/intel/oneapi/setvars.sh" >> $HOME/.bashrc

# Install NMODL with GPU enabled (needs separate Docker file with NVIDIA runtime enabled)
RUN cd nmodl && \
mkdir build && \
cd build && \
cmake .. \
  -DCMAKE_CXX_COMPILER=$(which clang++-13) \
  -DNMODL_ENABLE_LLVM=ON \
  -DNMODL_ENABLE_LLVM_GPU=ON \
  -DNMODL_ENABLE_LLVM_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/11.6/bin/nvcc \
  -DCMAKE_INSTALL_PREFIX=./install && \
cmake --build . --target install --parallel

# Install python packages needed for plotting results
RUN pip install seaborn
