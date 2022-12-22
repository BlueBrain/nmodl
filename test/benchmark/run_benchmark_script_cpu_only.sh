#!/usr/bin/env bash
#
# Driver for MOD2IR (CPU-only) benchmarking
#

set -e
set -x

# nmodl binary
nmodl_src_dir=$(pwd)/../../
nmodl_exe=${nmodl_src_dir}/build/install/bin/nmodl

# external kernel
kernels_path=${nmodl_src_dir}/test/benchmark/kernels
modfile_directory=${nmodl_src_dir}/test/benchmark/kernels
ext_lib="libextkernel.so"

export PYTHONPATH=${nmodl_src_dir}/build/install/lib:$PYTHONPATH

execute_benchmark_cpu() {
    python3 benchmark_script.py \
        --modfiles "./kernels/hh.mod" "./kernels/expsyn.mod" \
        --architectures "skylake-avx512"  \
        --compilers "intel" "gcc" "nvhpc" "clang" \
        --external \
        --nmodl_jit \
        --output "./hh_expsyn_cpu" \
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

execute_benchmark_cpu
