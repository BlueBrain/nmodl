#!/bin/bash
# set -x
#
# Driver for nmodl-llvm benchmarking
#
set -e
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
-ext, --external-kernel    Runs external kernels.
-o, --output-directory     Sets the output directory.
-d, --dry-run              Debug run.
-h, --help                 Display this usage information.
-V, --version              Show version and exit.
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
nmodl_exe="/gpfs/bbp.cscs.ch/data/scratch/proj16/magkanar/nmodl/build_benchmark/install/bin/nmodl"

# external kernel
nmodl_src_path="/gpfs/bbp.cscs.ch/data/scratch/proj16/magkanar/nmodl"
kernels_path=${nmodl_src_path}/"test/benchmark/kernels"
ext_lib="libextkernel.so"
modfile_directory=${kernels_path}

mkdir -p ${output_dir}

intel_path="/gpfs/bbp.cscs.ch/ssd/apps/hpc/jenkins/deploy/compilers/2021-01-06/linux-rhel7-x86_64/gcc-4.8.5/intel-20.0.2-ilowey/bin"
# compilers
icpc_exe=${intel_path}/icpc
declare -a icpc_flags_skylake_avx512=(
    "-O2 -march=skylake-avx512 -mtune=skylake -prec-div -fimf-use-svml"
    "-O2 -march=skylake-avx512 -mtune=skylake -prec-div -fimf-use-svml -fopenmp"
    )

declare -a icpc_flags_broadwell=(
    "-O2 -march=broadwell -mtune=broadwell -prec-div -fimf-use-svml"
    )

declare -a icpc_flags_nehalem=(
    "-O2 -msse2 -prec-div -fimf-use-svml"
    )

llvm_path="/gpfs/bbp.cscs.ch/apps/hpc/llvm-install/0621"
llvm_lib=${llvm_path}/lib
clang_exe=${llvm_path}/bin/clang++
llc_exe=${llvm_path}/bin/llc
declare -a clang_flags_skylake_avx512=(
    "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -fveclib=SVML"
    "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -fopenmp -fveclib=SVML"
    )

declare -a clang_flags_broadwell=(
    "-O3 -march=broadwell -mtune=broadwell -ffast-math -fopenmp -fveclib=SVML"
    )

declare -a clang_flags_nehalem=(
    "-O3 -march=nehalem -mtune=nehalem -ffast-math -fopenmp -fveclib=SVML"
    )

gcc_bin_path="/gpfs/bbp.cscs.ch/ssd/apps/hpc/jenkins/deploy/compilers/2021-01-06/linux-rhel7-x86_64/gcc-4.8.5/gcc-9.3.0-45gzrp/bin"
gcc_exe=${gcc_bin_path}/g++
declare -a gcc_flags_skylake_avx512=(
    "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -ftree-vectorize -mveclibabi=svml"
    "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -ftree-vectorize -mveclibabi=svml -fopenmp"
    )

declare -a gcc_flags_broadwell=(
    "-O3 -march=broadwell -mtune=broadwell -ffast-math -ftree-vectorize -mveclibabi=svml -fopenmp"
    )

declare -a gcc_flags_nehalem=(
    "-O3 -march=nehalem -mtune=nehalem -ffast-math -ftree-vectorize -mveclibabi=svml -fopenmp"
    )

declare -a benchmark_description
declare -a benchmark_time
declare -a benchmark_variance
# also get variance
KERNEL_TARGETS="compute-bound memory-bound hh"
#KERNEL_TARGETS="compute-bound"
#KERNEL_TARGETS="hh"
ARCHITECTURES="skylake_avx512 broadwell nehalem"
#ARCHITECTURES="avx512"
#ARCHITECTURES="skylake_avx512"
# set cpu option in jit according to the architecture
COMPILERS="icpc clang cpp"
#COMPILERS="clang"

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

	                ${debug} mkdir -p ${rel_ext_path_cpp}
	                ${debug} cd ${rel_ext_path_cpp}
	                ext_path=$(pwd)
	                ${debug} ${!compiler_exe} ${flags} ${kernels_path}/${kernel_target}.cpp \
	                -shared -fpic -o ${ext_lib}
	                ${debug} eval "objdump ${ext_lib} -d > ${ext_lib::-1}"
	                ${debug} cd ..

                    # add --fmf nnan contract afn here to generate .ll file similar to the fast-math options from external compilers
                    nmodl_args="${kernels_path}/${kernel_target}.mod llvm --ir --vector-width ${vec_width} --veclib SVML --opt-level-ir 3 --fmf nnan contract afn benchmark --run --instance-size ${kernel_inst_size} --repeat ${num_exp} --opt-level-codegen 3 --cpu ${nmodl_architecture} --libs ${vec_lib_path}/${vec_lib}"
                    nmodl_args="${nmodl_args} --external"

                    benchmark_ext_desc=ext_${kernel_target}_${compiler}_${nmodl_architecture}_v${vec_width}_${flags//[[:blank:]]/}
                    benchmark_description+=("${benchmark_ext_desc}")
                    # runs only external kernel
                    ${debug} eval "LD_LIBRARY_PATH=${ext_path}:${vec_lib_path}:${llvm_lib} ${nmodl_exe} ${nmodl_args} &> ${output_dir}/${benchmark_ext_desc}.log"
                    benchmark_time+=($(grep "Average compute time" ${output_dir}/${benchmark_ext_desc}.log | awk '{print $NF}'))
                    benchmark_variance+=($(grep "Compute time variance" ${output_dir}/${benchmark_ext_desc}.log | awk '{print $NF}'))

                    if [ "$compiler" == "clang" ]; then
                        rel_ext_path_llvm=${kernel_target}_${spec}_llvm
                        ${debug} mkdir -p ${rel_ext_path_llvm}
                        # Generate external library from LLVM IR of JIT
                        ${debug} sed 's/nrn_state_hh/_Z16nrn_state_hh_extPv/g' v${vec_width}_${kernel_target}_opt.ll > ${output_dir}/v${vec_width}_${kernel_target}_opt_ext.ll
                        ${debug} ${!compiler_exe} ${flags} -shared -fpic ${output_dir}/v${vec_width}_${kernel_target}_opt_ext.ll -o ${rel_ext_path_llvm}/${ext_lib} &>/dev/null # overwrites previous ${ext_lib}

                        benchmark_ext_jit_desc=ext_jit_${kernel_target}_${compiler}_${nmodl_architecture}_v${vec_width}_${flags//[[:blank:]]/}
                        benchmark_description+=("${benchmark_ext_jit_desc}")
                        # run external library generated by the LLVM IR code of JIT
                        ${debug} eval "LD_LIBRARY_PATH=${rel_ext_path_llvm}:${vec_lib_path}:${llvm_lib} ${nmodl_exe} ${nmodl_args} &> ${output_dir}/${benchmark_ext_jit_desc}.log"
                        benchmark_time+=($(grep "Average compute time" ${output_dir}/${benchmark_ext_jit_desc}.log | awk '{print $NF}'))
                        benchmark_variance+=($(grep "Compute time variance" ${output_dir}/${benchmark_ext_jit_desc}.log | awk '{print $NF}'))
                    fi
                done
            done
		fi
        echo "|  |  NMODL JIT"
        for fast_math in true false; do
            if $fast_math; then
                fast_math_flag="--fmf nnan contract afn"
                fast_math_opt="nnancontractafn"
            else
                fast_math_flag=""
                fast_math_opt="nonfastmath"
            fi
            for assume_may_alias in true false; do
                if $assume_may_alias; then
                    assume_may_alias_flag="--assume-may-alias"
                    assume_may_alias_opt="alias"
                else
                    assume_may_alias_flag=""
                    assume_may_alias_opt="noalias"
                fi
                echo "|  |  options: ${fast_math_flag} ${assume_may_alias_flag}"
                nmodl_args="${kernels_path}/${kernel_target}.mod llvm --ir ${fast_math_flag} ${assume_may_alias_flag} --vector-width ${vec_width} --veclib SVML --opt-level-ir 3 benchmark --run --instance-size ${kernel_inst_size} --repeat ${num_exp} --opt-level-codegen 3 --cpu ${nmodl_architecture} --libs ${vec_lib_path}/${vec_lib}"
                benchmark_nmodl_desc=nmodl_${kernel_target}_${nmodl_architecture}_v${vec_width}_${fast_math_opt}_${assume_may_alias_opt}
                benchmark_description+=("${benchmark_nmodl_desc}")
                # runs only kernel generated by LLVM IR
                ${debug} eval "LD_LIBRARY_PATH=${ext_path}:${vec_lib_path}:${llvm_lib} ${nmodl_exe} ${nmodl_args} &> ${output_dir}/${benchmark_nmodl_desc}.log"
                benchmark_time+=($(grep "Average compute time" ${output_dir}/${benchmark_nmodl_desc}.log | awk '{print $NF}'))
                benchmark_variance+=($(grep "Compute time variance" ${output_dir}/${benchmark_nmodl_desc}.log | awk '{print $NF}'))
            done
        done
    done
done

OUTPUT_FILE=${output_dir}/output_${KERNEL_TARGETS//[[:blank:]]/}_${ARCHITECTURES//[[:blank:]]/}.txt
rm -f ${OUTPUT_FILE}
for index in ${!benchmark_description[@]}; do
    echo -e "${benchmark_description[$index]}\t${benchmark_time[$index]}\t${benchmark_variance[$index]}" &>> ${OUTPUT_FILE}
done

cat ${OUTPUT_FILE}
