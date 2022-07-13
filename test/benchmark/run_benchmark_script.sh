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
set -x

module purge
unset MODULEPATH
export MODULEPATH=/gpfs/bbp.cscs.ch/ssd/apps/bsd/modules/_meta:/gpfs/bbp.cscs.ch/data/scratch/proj16/magkanar/spack_modules/linux-rhel7-skylake
module load unstable gcc/11.2.0 cuda/11.6.1 python-dev

#intel paths
intel_library_dir=$(module show intel-oneapi-compilers/2021.4.0 2>&1 | grep " LD_LIBRARY_PATH " | grep "intel64_lin" | awk -F' ' '{print $3}' | head -n 1)
svml_lib=$intel_library_dir/libsvml.so
intel_exe=$(module show intel-oneapi-compilers/2021.4.0 2>&1 | grep " PATH " | awk -F' ' '{print $3}' | head -n 1)/intel64/icc

#sleef library
sleef_lib=/gpfs/bbp.cscs.ch/apps/hpc/llvm-install/0621/sleef-3.5.1/lib64/libsleefgnuabi.so

#llvm path
llvm_path=$(module show llvm/13.0.0 2>&1 | grep " PATH " | awk -F' ' '{print $3}' | head -n 1)
clang_exe=${llvm_path}/clang++
llc_exe=${llvm_path}/llc

#gcc path
gcc_exe=$(module show gcc/11.2.0 2>&1 | grep " PATH " | awk -F' ' '{print $3}' | head -n 1)/g++

#nvhpc path
nvhpc_exe=$(module show nvhpc/22.3 2>&1 | grep " PATH " | awk -F' ' '{print $3}' | head -n 1)/nvc++

#libdevice path
libdevice_lib=${CUDA_HOME}/nvvm/libdevice/libdevice.10.bc

#add ld library path
export LD_LIBRARY_PATH=`dirname $svml_lib`:`dirname $sleef_lib`:${llvm_path}/lib:$LD_LIBRARY_PATH

# nmodl binary
nmodl_src_dir=$(pwd)/../../
nmodl_exe=${nmodl_src_dir}/build_benchmark_gpu_math1/install/bin/nmodl

# external kernel
kernels_path=${nmodl_src_dir}/test/benchmark/kernels
modfile_directory=${nmodl_src_dir}/test/benchmark/kernels
ext_lib="libextkernel.so"

export PYTHONPATH=/gpfs/bbp.cscs.ch/data/scratch/proj16/magkanar/nmodl_llvm_benchmark/nmodl/build_benchmark_gpu_math1/install/lib:$PYTHONPATH

execute_benchmark() {
    python benchmark_script.py \
        --modfiles "./kernels/hh.mod" "./kernels/expsyn.mod" \
        --architectures "default" "skylake-avx512" \
        --compilers "intel" "gcc" "clang" \
        --external \
        --nmodl_jit \
        --output "./hh_expsyn_mavx512f" \
        --instances 100000000 \
        --experiments 5 \
        --svml_lib $svml_lib \
        --intel_exe $intel_exe \
        --sleef_lib $sleef_lib \
        --clang_exe $clang_exe \
        --llc_exe $llc_exe \
        --gcc_exe $gcc_exe \
        --nvhpc_exe $nvhpc_exe \
        --libdevice_lib $libdevice_lib \
        --nmodl_exe $nmodl_exe /
}

roofline_gpu() {
    mod_name=$1
    ncu --set full -f -o "${mod_name}_full" python benchmark_script.py \
        --modfiles "./kernels/${mod_name}.mod" \
        --architectures "nvptx64" \
        --compilers "nvhpc" \
        --external \
        --nmodl_jit \
        --output "./${mod_name}_nvhpc_ncu" \
        --instances 100000000 \
        --experiments 1 \
        --svml_lib $svml_lib \
        --intel_exe $intel_exe \
        --sleef_lib $sleef_lib \
        --clang_exe $clang_exe \
        --llc_exe $llc_exe \
        --gcc_exe $gcc_exe \
        --nvhpc_exe $nvhpc_exe \
        --libdevice_lib $libdevice_lib \
        --nmodl_exe $nmodl_exe /
}

roofline_cpu() {
    mod_name=$1
    module load intel-oneapi-advisor/2021.4.0
    advisor --collect roofline --project-dir "${mod_name}_advisor" python benchmark_script.py \
        --modfiles "./kernels/${mod_name}.mod" \
        --architectures "default" "skylake-avx512" \
        --compilers "intel" \
        --external \
        --nmodl_jit \
        --output "./${mod_name}_icc_default_skylake_advisor" \
        --instances 100000000 \
        --experiments 1 \
        --svml_lib $svml_lib \
        --intel_exe $intel_exe \
        --sleef_lib $sleef_lib \
        --clang_exe $clang_exe \
        --llc_exe $llc_exe \
        --gcc_exe $gcc_exe \
        --nvhpc_exe $nvhpc_exe \
        --libdevice_lib $libdevice_lib \
        --nmodl_exe $nmodl_exe /
    module unload intel-oneapi-advisor/2021.4.0
}

execute_benchmark
# roofline_gpu hh
# roofline_cpu hh
