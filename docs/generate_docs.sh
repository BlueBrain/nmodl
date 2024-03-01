#!/usr/bin/env bash

# script for generating documentation for NMODL

set -eu

if [ $# -lt 2 ]
then
    echo "Usage: $(basename "$0") output_dir python_exe [use_virtual_env]"
    exit 1
fi

# the dir where we put the temporary build and the docs
output_dir="$1"
# path to the Python executable
python_exe="$2"
use_venv="${3:-}"

if ! [ -d "${output_dir}" ]
then
    mkdir -p "${output_dir}"
fi

echo "== Building documentation files in: ${output_dir}/_docs =="
echo "== Temporary project build directory is: ${output_dir}/_build =="

if [ "${use_venv}" != "false" ]
then
    venv_name="$(mktemp -d)"
    ${python_exe} -m venv "${venv_name}"
    . "${venv_name}/bin/activate"
    python_exe="$(command -v python)"
    ${python_exe} -m pip install -U pip
fi
${python_exe} -m pip install ".[docs]" -C build-dir="${output_dir}/_build"

# the abs dir where this script is located (so we can call it from wherever)
script_dir="$(cd "$(dirname "$0")"; pwd -P)"

cd "${script_dir}"
doxygen Doxyfile
cd -
sphinx-build docs/ "${output_dir}/_docs"

if [ "${use_venv}" != "false" ]
then
    deactivate
fi
