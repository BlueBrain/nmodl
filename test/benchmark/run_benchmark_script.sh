#!/usr/bin/env bash
#SBATCH --account=proj16
#SBATCH --partition=prod
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH -n 40
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --constraint=gpu_32g

#
# Driver for nmodl-llvm benchmarking
#

set -e
#set -x

module purge
unset MODULEPATH
export MODULEPATH=/gpfs/bbp.cscs.ch/ssd/apps/bsd/modules/_meta
module load unstable gcc cuda


#intel paths
intel_library_dir=$(module show intel-oneapi-compilers 2>&1 | grep " LD_LIBRARY_PATH " | grep "intel64_lin" | awk -F' ' '{print $3}' | head -n 1)
svml_lib=$intel_library_dir/libsvml.so
intel_exe=$(module show intel 2>&1 | grep " PATH " | awk -F' ' '{print $3}' | head -n 1)/icpc

#sleef library
sleef_lib=/gpfs/bbp.cscs.ch/apps/hpc/llvm-install/0621/sleef-3.5.1/lib64/libsleefgnuabi.so

#llvm path
llvm_path=$(module show llvm/13.0.0 2>&1 | grep " PATH " | awk -F' ' '{print $3}' | head -n 1)
clang_exe=${llvm_path}/clang++
llc_exe=${llvm_path}/llc

#gcc path
gcc_exe=$(module show gcc 2>&1 | grep " PATH " | awk -F' ' '{print $3}' | head -n 1)/g++

#libdevice path
libdevice_lib=${CUDA_HOME}/nvvm/libdevice/libdevice.10.bc

#add ld library path
export LD_LIBRARY_PATH=`dirname $svml_lib`:`dirname $sleef_lib`:${llvm_path}/lib:$LD_LIBRARY_PATH

# nmodl binary
nmodl_src_dir=$(pwd)/../../
nmodl_exe=${nmodl_src_dir}/build_benchmark_gpu/bin/nmodl

# external kernel
kernels_path=${nmodl_src_dir}/test/benchmark/kernels
modfile_directory=${nmodl_src_dir}/test/benchmark/kernels
ext_lib="libextkernel.so"

srun -n 1 python run_benchmark_script.py 