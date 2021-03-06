#!/bin/bash

set -xe

git show HEAD

source /gpfs/bbp.cscs.ch/apps/hpc/jenkins/config/modules.sh
module use /gpfs/bbp.cscs.ch/apps/tools/modules/tcl/linux-rhel7-x86_64/

module load archive/2020-10 cmake bison flex python-dev doxygen
module list

function bb5_pr_setup_virtualenv() {
    # latest version of breathe from 21st April has issue with 4.13.0, see https://github.com/michaeljones/breathe/issues/431
    # temporary workaround
    virtualenv venv
    . venv/bin/activate
    pip3 install "breathe<=4.12.0"
    pip3 install "cmake-format==0.6.13"
}

function find_clang_format() {
    module load llvm
    clang_format_exe=$(which clang-format)
    module unload llvm
}

function build_with() {
    compiler="$1"
    module load $compiler
    . venv/bin/activate
    find_clang_format

    echo "Building NMODL with $compiler"
    module load $compiler
    rm -rf build_$compiler
    mkdir build_$compiler
    pushd build_$compiler
    cmake .. -DCMAKE_C_COMPILER=$MPICC_CC \
             -DCMAKE_CXX_COMPILER=$MPICXX_CXX \
             -DPYTHON_EXECUTABLE=$(which python3) \
             -DNMODL_FORMATTING:BOOL=ON \
             -DClangFormat_EXECUTABLE=$clang_format_exe \
             -DLLVM_DIR=/gpfs/bbp.cscs.ch/apps/hpc/jenkins/merge/deploy/externals/latest/linux-rhel7-x86_64/gcc-9.3.0/llvm-11.0.0-kzl4o5/lib/cmake/llvm
    make -j6
    popd
}

function test_with() {
    compiler="$1"
    module load $compiler
    . venv/bin/activate
    cd build_$compiler
    env CTEST_OUTPUT_ON_FAILURE=1 make test
}

function make_target() {
    compiler=$1
    target=$2
    . venv/bin/activate
    cd build_$compiler
    make $target
}

function bb5_pr_cmake_format() {
    make_target intel check-cmake-format
}

function bb5_pr_clang_format() {
    make_target intel check-clang-format
}

function bb5_pr_build_gcc() {
    build_with gcc
}

function bb5_pr_build_intel() {
    build_with intel
}

function bb5_pr_build_pgi() {
    build_with pgi
}

function bb5_pr_test_gcc() {
    test_with gcc
}

function bb5_pr_test_intel() {
    test_with intel
}

function bb5_pr_test_pgi() {
    test_with pgi
}

function bb5_pr_build_llvm() {
    build_with llvm
}

function bb5_pr_test_llvm() {
    test_with llvm
}

action=$(echo "$CI_BUILD_NAME" | tr ' ' _)
if [ -z "$action" ] ; then
    for action in "$@" ; do
        "bb5_pr_$action"
    done
else
    "bb5_pr_$action"
fi
