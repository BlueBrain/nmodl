#!/bin/bash
# set -x
#
# Driver for nmodl-llvm benchmarking
#


# default params
inst_size=100000000
num_exp=10
vec_width=8

# version
version="0.0.1"
version_date="20-5-2021"
version_string="nmodl-llvm-time $version ($version_date)"

# show usage and handle arguments
function showusage {
    echo "usage: nmodl-llvm-time [options].
-n NUMBER, --instance-size NUMBER
-e NUMBER, --num-exeperiment NUMBER
-v NUMBER, --vec-width NUMBER
-d, --dry-run
-h, --help       Display this usage information.
-V, --version    Show version and exit.
Driver for benchmarking.
"
}


while [[ "$1" != "" ]]; do
    case $1 in
        "")
            shift
            ;;
        -n|--instance-size)
            inst_size=$2
            shift
            shift
            ;;
        -e|--num-exeperiment)
            num_exp=$2
            shift
            shift
            ;;
        -v|--vec-width)
            vec_width=$2
            shift
            shift
            ;;
        -d|--dry-run)
            echo "debug mode"
            debug=echo
            shift
            ;;
        -V|--version)
            echo "$version_string"
            exit 0
            ;;
        -h|-\?|--help)
            showusage
            exit 0
            ;;
        *)
            showusage
            exit 1
            ;;
    esac

done

# vec libs
vec_lib_path="/gpfs/bbp.cscs.ch/ssd/apps/hpc/jenkins/deploy/compilers/2021-01-06/linux-rhel7-x86_64/gcc-4.8.5/intel-20.0.2-ilowey/lib/intel64_lin"
vec_lib="libsvml.so"

# nmodl
nmodl_exe="/gpfs/bbp.cscs.ch/home/gcastigl/project16/nmodl-llvm/build/install/bin/nmodl"

# external kernel
nmodl_src_path="/gpfs/bbp.cscs.ch/home/gcastigl/project16/nmodl-llvm"
kernels_path=${nmodl_src_path}/"test/benchmark/kernels"
ext_lib="libextkernel.so"

# compilers
icpc_exe=icpc
declare -a icpc_flags=(
    # "-O2"
    "-O2 -march=skylake-avx512 -mtune=skylake-avx512 -fimf-use-svml"
    "-O2 -qopt-zmm-usage=high -xCORE-AVX512 -fimf-use-svml"
    )

clang_bin_path="/gpfs/bbp.cscs.ch/data/project/proj16/software/llvm/install/0521/bin"
clang_exe=${clang_bin_path}/clang++
declare -a clang_flags=(
    # "-O3"
    "-O3 -march=skylake-avx512 -fveclib=SVML"
    "-O3 -march=skylake-avx512 -ffast-math -fveclib=SVML"
    "-O3 -mavx512f -ffast-math -fveclib=SVML"
    )

# loop over options
for kernel_target in compute-bound memory-bound; do # add here hh
    echo "kernel: "${kernel_target}
    
    for compiler in icpc clang; do
        echo "|  compiler: "${compiler}

        compiler_exe=${compiler}_exe
        compiler_flags=${compiler}_flags[@]
        
        for flags in "${!compiler_flags}"; do
            echo "|  |  flags: "${flags}

            spec=${compiler}_${flags//[[:blank:]]/}
            rel_ext_path=${kernel_target}_${spec}

            ${debug} mkdir ${rel_ext_path}
            ${debug} cd ${rel_ext_path}
            ext_path=$(pwd)
            ${debug} ${!compiler_exe} ${flags} ${kernels_path}/${kernel_target}.cpp \
            -shared -fpic -o ${ext_lib}
            ${debug} eval "llvm-objdump ${ext_lib} -d > ${ext_lib::-1}"
            ${debug} cd ..

            nmodl_args="${kernels_path}/${kernel_target}.mod llvm --ir --vector-width ${vec_width} --veclib SVML benchmark \
            --opt-level-ir 3 --opt-level-codegen 3 --run --instance-size ${inst_size} \
            --repeat ${num_exp} \
            --libs  ${vec_lib_path}/${vec_lib} \
            --backend default"

            # run experiment
            ${debug} eval "LD_LIBRARY_PATH=${ext_path}:${vec_lib_path} ${nmodl_exe} ${nmodl_args} &> ${kernel_target}_${spec}.log"
        done
    done
done