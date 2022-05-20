#!/bin/bash

#
# Driver for nmodl-llvm benchmarking
#

set -e
#set -x

# sample run
# sh nmodl-llvm-time.sh -n 100000000 -o llvm_benchmark_all_big_100mil -ext -e 5

module purge
unset MODULEPATH
export MODULEPATH=/gpfs/bbp.cscs.ch/ssd/apps/bsd/modules/_meta
module load unstable gcc cuda

# default params
inst_size=100000000
num_exp=5
vec_width=8
external_kernel_exec=false
modfile_directory=$(pwd)
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
-ext, --external-kernel    Runs external kernels.
-o, --output-directory     Sets the output directory.
-d, --dry-run              Debug run.
-h, --help                 Display this usage information.
-V, --version              Show version and exit.
-ncu, --nsight-compute     Run CUDA ncu for GPU kernels
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
        -ncu|--nsight-compute)
            echo "Nsight Compute mode for gpu"
            ncu_exec=$(which ncu)
            shift
            ;;
        -nsys|--nsys-profile)
            echo "Nsys profiling mode for gpu"
            nsys_exec=$(which nsys)
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

# compiler flags
declare -a intel_flags_skylake_avx512=(
    "-O2 -march=skylake-avx512 -mtune=skylake -prec-div -fimf-use-svml"
    "-O2 -march=skylake-avx512 -mtune=skylake -prec-div -fimf-use-svml -fopenmp"
    )

declare -a intel_flags_broadwell=(
    "-O2 -march=broadwell -mtune=broadwell -prec-div -fimf-use-svml"
    )

declare -a intel_flags_nehalem=(
    "-O2 -msse2 -prec-div -fimf-use-svml"
    )

declare -a clang_flags_skylake_avx512=(
    "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math"
    "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -fopenmp"
    "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -fveclib=SVML"
    "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -fopenmp -fveclib=SVML"
    )

declare -a clang_flags_broadwell=(
    "-O3 -march=broadwell -mtune=broadwell -ffast-math -fopenmp"
    "-O3 -march=broadwell -mtune=broadwell -ffast-math -fopenmp -fveclib=SVML"
    )

declare -a clang_flags_nehalem=(
    "-O3 -march=nehalem -mtune=nehalem -ffast-math -fopenmp"
    "-O3 -march=nehalem -mtune=nehalem -ffast-math -fopenmp -fveclib=SVML"
    )

declare -a gcc_flags_skylake_avx512=(
    "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -ftree-vectorize"
    "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -ftree-vectorize -fopenmp"
    "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -ftree-vectorize -mveclibabi=svml"
    "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -ftree-vectorize -mveclibabi=svml -fopenmp"
    )

declare -a gcc_flags_broadwell=(
    "-O3 -march=broadwell -mtune=broadwell -ffast-math -ftree-vectorize -fopenmp"
    "-O3 -march=broadwell -mtune=broadwell -ffast-math -ftree-vectorize -mveclibabi=svml -fopenmp"
    )

declare -a gcc_flags_nehalem=(
    "-O3 -march=nehalem -mtune=nehalem -ffast-math -ftree-vectorize -fopenmp"
    "-O3 -march=nehalem -mtune=nehalem -ffast-math -ftree-vectorize -mveclibabi=svml -fopenmp"
    )

declare -a benchmark_description
declare -a benchmark_time
declare -a benchmark_variance

# Kernels, architectures and compilers loop

KERNEL_TARGETS="compute-bound memory-bound hh"

ARCHITECTURES="skylake_avx512 broadwell nehalem default nvptx64"

COMPILERS="intel clang gcc"

mkdir -p ${output_dir}

# loop over options
for kernel_target in ${KERNEL_TARGETS}; do
    echo "Kernel: $kernel_target"

    # hh mechanism size 5 times the compute-bound and memory-bound
    if [ "$kernel_target" == "hh" ]; then
        kernel_inst_size=$(($inst_size/5))
    else
        kernel_inst_size=$inst_size
    fi
    for architecture in ${ARCHITECTURES}; do
        if [ "$architecture" = "skylake_avx512" ] ; then
            vec_width=8
        elif [ "$architecture" = "broadwell" ] ; then
            vec_width=4
        elif [ "$architecture" = "nehalem" ]; then
            vec_width=2
        else
            vec_width=1
        fi
        echo "|  Architecture: $architecture"
        if [ "$architecture" = "skylake_avx512" ] ; then
            nmodl_architecture="skylake-avx512"
        else
            nmodl_architecture=$architecture
        fi

        if $external_kernel_exec; then
            for compiler in ${COMPILERS}; do
                echo "|  |  Compiler: $compiler"

				compiler_exe=${compiler}_exe
	        	compiler_flags=${compiler}_flags_${architecture}[@]
	        	for flags in "${!compiler_flags}"; do
	        		echo "|  |  |  flags: "${flags}

	                spec=${compiler}_${flags//[[:blank:]]/}
	                rel_ext_path_cpp=${kernel_target}_${spec}_cpp
                    rel_ext_path_cpp=${rel_ext_path_cpp//=/_}
                    rel_ext_path_cpp=${rel_ext_path_cpp//-/_}

	                ${debug} mkdir -p ${rel_ext_path_cpp}
	                ${debug} cd ${rel_ext_path_cpp}
	                ext_path=$(pwd)
                    # replace pragmas with the corresponding ones for openmp or certain compiler
                    if [[ "$kernel_target" == "hh" ]]; then
                        if [[ "$flags" == *"openmp"* ]] || [[ "$compiler" == "intel" ]]; then
                            ${debug} sed -i 's/#pragma.*/#pragma omp simd/g' ${kernels_path}/${kernel_target}.cpp
                        elif [[ "$compiler" == "clang" ]]; then
                            ${debug} sed -i 's/#pragma.*/#pragma clang vectorize(enable)/g' ${kernels_path}/${kernel_target}.cpp
                        elif [[ "$compiler" == "gcc" ]]; then
                            ${debug} sed -i 's/#pragma.*/#pragma GCC ivdep/g' ${kernels_path}/${kernel_target}.cpp
                        fi
                    fi
                    # Create shared library that is linked with SVML shared library and add its RPATH to the generated shared library to be found later by NMODL
	                ${debug} ${!compiler_exe} ${flags} ${kernels_path}/${kernel_target}.cpp -shared -fpic -o ${ext_lib} -Wl,-rpath $intel_library_dir -L$intel_library_dir -lsvml
	                ${debug} eval "objdump ${ext_lib} -d > ${ext_lib::-1}"
	                ${debug} cd ..

                    # add --fmf nnan contract afn here to generate .ll file similar to the fast-math options from external compilers
                    nmodl_common_args="${kernels_path}/${kernel_target}.mod benchmark --run --instance-size ${kernel_inst_size} --repeat ${num_exp} --opt-level-codegen 3 --libs ${svml_lib} ${sleef_lib} --external"
                    nmodl_llvm_args="llvm --ir --opt-level-ir 3 --fmf nnan contract afn cpu --name ${nmodl_architecture} --vector-width ${vec_width} --math-library SVML"

                    benchmark_ext_desc=${kernel_target}_${compiler}_${nmodl_architecture}_v${vec_width}_${flags//[[:blank:]]/}
                    benchmark_description+=("${benchmark_ext_desc}")
                    # runs only external kernel
                    ${debug} eval "LD_PRELOAD=${ext_path}/${ext_lib} ${nmodl_exe} ${nmodl_common_args} ${nmodl_llvm_args} 2>&1 | tee ${output_dir}/${benchmark_ext_desc}.log"
                    benchmark_time+=($(grep "Average compute time" ${output_dir}/${benchmark_ext_desc}.log | awk '{print $NF}'))
                    benchmark_variance+=($(grep "Compute time variance" ${output_dir}/${benchmark_ext_desc}.log | awk '{print $NF}'))
                done

                if [ "$compiler" == "clang" ]; then
                    module load llvm/13.0.0
                    for math_lib in SVML SLEEF;
                    do
                      nmodl_llvm_args="llvm --ir --opt-level-ir 3 --fmf nnan contract afn cpu --vector-width ${vec_width} --math-library ${math_lib}"
                      rel_ext_path_llvm=${kernel_target}_nmodl_${spec}_llvm_${math_lib}
                      rel_ext_path_llvm=${rel_ext_path_llvm//=/_}
                      rel_ext_path_llvm=${rel_ext_path_llvm//-/_}
                      ${debug} mkdir -p ${rel_ext_path_llvm}
                      ${debug} eval "LD_PRELOAD=${ext_path}/${ext_lib} ${nmodl_exe} ${kernels_path}/${kernel_target}.mod ${nmodl_common_args} ${nmodl_llvm_args}"
                      # Generate external library from LLVM IR of JIT
                      ${debug} sed 's/nrn_state_hh/_Z16nrn_state_hh_extPv/g' v${vec_width}_${kernel_target}_opt.ll > ${rel_ext_path_llvm}/v${vec_width}_${kernel_target}_opt_ext.ll
                      ${debug} ${!compiler_exe} ${flags} -shared -fpic ${rel_ext_path_llvm}/v${vec_width}_${kernel_target}_opt_ext.ll -o ${rel_ext_path_llvm}/${ext_lib} &>/dev/null # overwrites previous ${ext_lib}
                      benchmark_ext_jit_desc=${kernel_target}_nmodl_${nmodl_architecture}_v${vec_width}_${flags//[[:blank:]]/}_${math_lib}
                      benchmark_description+=("${benchmark_ext_jit_desc}")
                      # run external library generated by the LLVM IR code of JIT
                      ${debug} eval "LD_PRELOAD=${rel_ext_path_llvm}/${ext_lib} ${nmodl_exe} ${nmodl_common_args} ${nmodl_llvm_args} 2>&1 | tee ${output_dir}/${benchmark_ext_jit_desc}.log"
                      benchmark_time+=($(grep "Average compute time" ${output_dir}/${benchmark_ext_jit_desc}.log | awk '{print $NF}'))
                      benchmark_variance+=($(grep "Compute time variance" ${output_dir}/${benchmark_ext_jit_desc}.log | awk '{print $NF}'))

                    done
                    module unload llvm/13.0.0
                fi

            done
		fi
        echo "|  |  NMODL JIT"
        for fast_math in false true; do
            if $fast_math; then
                fast_math_flag="--fmf nnan contract afn"
                fast_math_opt="nnancontractafn"
            else
                fast_math_flag=""
                fast_math_opt="nonfastmath"
            fi
            echo "|  |  |  options: ${fast_math_flag}"
            if [[ $nmodl_architecture != "nvptx64" ]]; then
                for math_lib in SVML SLEEF;
                do
                    echo "|  |  |  |  math library: ${math_lib}"
                    if [[ $math_lib == "SVML" ]]; then
                        math_lib_path=${svml_lib}
                    else
                        math_lib_path=${sleef_lib}
                    fi
                    benchmark_nmodl_desc=${kernel_target}_nmodl-jit_${math_lib}_${nmodl_architecture}_v${vec_width}_${fast_math_opt}
                    benchmark_description+=("${benchmark_nmodl_desc}")
                    nmodl_args="${kernels_path}/${kernel_target}.mod --output ${output_dir}/${benchmark_nmodl_desc} llvm --ir ${fast_math_flag} --opt-level-ir 3 cpu --name ${nmodl_architecture} --vector-width ${vec_width} --math-library ${math_lib} benchmark --run --instance-size ${kernel_inst_size} --repeat ${num_exp} --opt-level-codegen 3 --libs ${math_lib_path}"
                    # runs only kernel generated by LLVM IR
                    ${debug} eval "${nmodl_exe} ${nmodl_args} 2>&1 | tee ${output_dir}/${benchmark_nmodl_desc}.log"
                    benchmark_time+=($(grep "Average compute time" ${output_dir}/${benchmark_nmodl_desc}.log | awk '{print $NF}' | tail -n 1))
                    benchmark_variance+=($(grep "Compute time variance" ${output_dir}/${benchmark_nmodl_desc}.log | awk '{print $NF}' | tail -n 1))
                done
            else
                benchmark_nmodl_desc=${kernel_target}_nmodl-cuda-jit_libdevice_${nmodl_architecture}_v${vec_width}_${fast_math_opt}
                benchmark_description+=("${benchmark_nmodl_desc}")
                nmodl_args="${kernels_path}/${kernel_target}.mod --output ${output_dir}/${benchmark_nmodl_desc} llvm --ir ${fast_math_flag} --opt-level-ir 3 gpu --name ${nmodl_architecture} --target-arch \"sm_70\" --math-library libdevice benchmark --run --instance-size ${kernel_inst_size} --repeat ${num_exp} --opt-level-codegen 3 --libs ${libdevice_lib} --grid-dim-x 16384 --block-dim-x 512"
                # runs only kernel generated by LLVM IR
                if [[ $ncu_exec != "" ]]; then
                    nvidia_profile="${ncu_exec} --set detailed -f -o ${kernel_target}_${fast_math_opt}_detailed"
                fi
                if [[ $nsys_exec != "" ]]; then
                    nvidia_profile="${nsys_exec} profile --stats=true --force-overwrite=true -o ${output_dir}/${kernel_target}_${fast_math_opt}_nsys"
                fi
                ${debug} eval "${nvidia_profile} ${nmodl_exe} ${nmodl_args} 2>&1 | tee ${output_dir}/${benchmark_nmodl_desc}.log"
                benchmark_time+=($(grep "Average compute time" ${output_dir}/${benchmark_nmodl_desc}.log | awk '{print $NF}' | tail -n 1))
                benchmark_variance+=($(grep "Compute time variance" ${output_dir}/${benchmark_nmodl_desc}.log | awk '{print $NF}' | tail -n 1))
            fi
        done
    done
done

OUTPUT_FILE=${output_dir}/output_${KERNEL_TARGETS//[[:blank:]]/}_${ARCHITECTURES//[[:blank:]]/}.txt
rm -f ${OUTPUT_FILE}
for index in ${!benchmark_description[@]}; do
    echo -e "${benchmark_description[$index]}\t${benchmark_time[$index]}\t${benchmark_variance[$index]}" &>> ${OUTPUT_FILE}
done

cat ${OUTPUT_FILE}
