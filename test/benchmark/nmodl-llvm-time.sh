#!/bin/bash
# set -x
#
# Driver for nmodl-llvm benchmarking
#

# sh nmodl-llvm-time.sh -vec-sweep -mod-dir /gpfs/bbp.cscs.ch/data/scratch/proj16/magkanar/nmodl/bbp_mod -n 100000000
# default params
inst_size=100000000
num_exp=10
vec_width=8
external_kernel_exec=false
modfile_directory=$(pwd)
vec_width_sweep=false
output_dir=$(pwd)

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
        -ext|--external-kernel)
            external_kernel_exec=true
            shift
            ;;
        -vec-sweep|--vec-width-sweep)
            vec_width_sweep=true
            shift
            ;;
        -mod-dir|--modfile-directory)
            modfile_directory=$2
            shift
            shift
            ;;
        -o|--output-directory)
            output_dir=$2
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
nmodl_exe="/gpfs/bbp.cscs.ch/data/scratch/proj16/magkanar/nmodl/build_llvm/install/bin/nmodl"

# external kernel
nmodl_src_path="/gpfs/bbp.cscs.ch/home/gcastigl/project16/nmodl-llvm"
kernels_path=${nmodl_src_path}/"test/benchmark/kernels"
ext_lib="libextkernel.so"
if ${external_kernel_exec}; then
    modfile_directory=${kernels_path}
fi

mkdir -p ${output_dir}

# compilers
icpc_exe=icpc
declare -a icpc_flags=(
    "-O2 -march=skylake-avx512 -mtune=skylake-avx512 -prec-div -fimf-use-svml"
    "-O2 -qopt-zmm-usage=high -xCORE-AVX512 -prec-div -fimf-use-svml"
    "-O2 -mavx512f -prec-div -fimf-use-svml"
    "-O2 -mavx2 -prec-div -fimf-use-svml"
    "-O2 -msse2 -prec-div -fimf-use-svml"
    )

llvm_path="/gpfs/bbp.cscs.ch/apps/hpc/llvm-install/0621"
llvm_lib=${llvm_path}/lib
clang_exe=${llvm_path}/bin/clang++
declare -a clang_flags=(
    "-O3 -mavx512f -ffast-math -fopenmp -fveclib=SVML"
    "-O3 -mavx2 -ffast-math -fopenmp -fveclib=SVML"
    "-O3 -msse2 -ffast-math -fopenmp -fveclib=SVML"
    "-O3 -mavx512f -ffast-math -fveclib=SVML"
    "-O3 -mavx512f -fveclib=SVML"
    "-O3 -march=skylake-avx512 -ffast-math -fopenmp -fveclib=SVML"
    )

gcc_bin_path="/gpfs/bbp.cscs.ch/ssd/apps/hpc/jenkins/deploy/compilers/2021-01-06/linux-rhel7-x86_64/gcc-4.8.5/gcc-9.3.0-45gzrp/bin"
gcc_exe=${gcc_bin_path}/g++
declare -a gcc_flags=(
    "-O3 -mavx512f -ffast-math -ftree-vectorize -mveclibabi=svml"
    "-O3 -mavx2 -ffast-math -ftree-vectorize -mveclibabi=svml"
    "-O3 -msse2 -ffast-math -ftree-vectorize -mveclibabi=svml"
    )

# loop over options
# for kernel_target in compute-bound memory-bound hh; do
#for kernel_target in compute-bound memory-bound; do
#for kernel_target in Ca_HVA2 can2 cat DetAMPANMDA DetGABAAB SKv3_1; do
for kernel_target in ; do
# for kernel_target in hh; do
    echo "kernel: "${kernel_target}
    
    # loop over other compilers
    # for compiler in icpc clang gcc; do
    for compiler in clang; do
        echo "|  compiler: "${compiler}

        compiler_exe=${compiler}_exe
        compiler_flags=${compiler}_flags[@]
        
         if $external_kernel_exec; then
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
            done
        fi

        if $vec_width_sweep; then
            for power_of_two in $(seq 1 3); do
                vec_width=$((2**${power_of_two}))
                echo "|  | Running JIT for vec width ${vec_width}"
                nmodl_args="${modfile_directory}/${kernel_target}.mod passes --inline \
                llvm --ir --fmf nnan contract afn --vector-width ${vec_width} --veclib SVML \
                benchmark \
                --opt-level-ir 3 --opt-level-codegen 3 --run --instance-size ${inst_size} \
                --repeat ${num_exp} \
                --libs  ${vec_lib_path}/${vec_lib} \
                --backend default"

                # run experiment
                if $external_kernel_exec; then
                    ${debug} eval "LD_LIBRARY_PATH=${ext_path}:${vec_lib_path}:${llvm_lib} ${nmodl_exe} ${nmodl_args} &> ${output_dir}/${kernel_target}_${spec}_v${vec_width}.log"
                else
                    ${debug} eval "LD_LIBRARY_PATH=${vec_lib_path}:${llvm_lib} ${nmodl_exe} ${nmodl_args} &> ${output_dir}/${kernel_target}_${spec}_v${vec_width}.log"
                fi
            done
        else
            nmodl_args="${modfile_directory}/${kernel_target}.mod passes --inline \
            llvm --ir --fmf nnan contract afn --vector-width ${vec_width} --veclib SVML benchmark \
            --opt-level-ir 3 --opt-level-codegen 3 --run --instance-size ${inst_size} \
            --repeat ${num_exp} \
            --libs  ${vec_lib_path}/${vec_lib} \
            --backend default"

            # run experiment
            if $external_kernel_exec; then
                ${debug} eval "LD_LIBRARY_PATH=${ext_path}:${vec_lib_path}:${llvm_lib} ${nmodl_exe} ${nmodl_args} &> ${output_dir}/${kernel_target}_${spec}_v${vec_width}.log"
            else
                ${debug} eval "LD_LIBRARY_PATH=${vec_lib_path}:${llvm_lib} ${nmodl_exe} ${nmodl_args} &> ${output_dir}/${kernel_target}_${spec}_v${vec_width}.log"
            fi
        fi
    done

done