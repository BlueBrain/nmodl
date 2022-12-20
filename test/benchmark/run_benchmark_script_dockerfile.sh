#!/usr/bin/env bash
#
# Driver for nmodl-llvm benchmarking
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

execute_benchmark() {
    python3 benchmark_script.py \
        --modfiles "./kernels/hh.mod" "./kernels/expsyn.mod" \
        --architectures "nehalem"  \
        --compilers "intel" "gcc" "nvhpc" "clang" \
        --external \
        --nmodl_jit \
        --output "./hh_expsyn_final_cpu" \
        --instances 10000000 \
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
    ncu --set full -f -o "${mod_name}_full_200mil" python benchmark_script.py \
        --modfiles "./kernels/${mod_name}.mod" \
        --architectures "nvptx64" \
        --compilers "nvhpc" \
        --external \
        --nmodl_jit \
        --output "./${mod_name}_nvhpc_ncu_200mil" \
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
    advisor --collect roofline --project-dir "${mod_name}_advisor_clang_avx512" python benchmark_script.py \
        --modfiles "./kernels/${mod_name}.mod" \
        --architectures "skylake-avx512" \
        --compilers "clang" \
        --external \
        --output "./${mod_name}_clang_avx512_skylake_advisor" \
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
# roofline_gpu expsyn
# roofline_cpu hh
# roofline_cpu expsyn
