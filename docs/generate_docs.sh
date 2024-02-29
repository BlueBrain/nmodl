#!/usr/bin/env bash

# script for generating documentation for NMODL

set -xeu

# in order to create the docs, we first need to build NMODL in some dir
build_dir="${1:-"$(mktemp -d)"}"
if ! [ -d "${build_dir}" ]
then
    mkdir -p "${build_dir}"
fi

# the Python executable to use (default: whatever `python3` is)
python_exe="${2:-"$(command -v python3)"}"

# should we use a separate venv?
use_venv="${3:-}"

if [ "${use_venv}" != "false" ]
then
    venv_name="$(mktemp -d)"
    ${python_exe} -m venv "${venv_name}"
    . "${venv_name}/bin/activate"
    python_exe="$(command -v python)"
    ${python_exe} -m pip install -U pip
else
    python_exe="$(command -v python3)"
fi
${python_exe} -m pip install ".[docs]" -C build-dir="$(cd "${build_dir}"; pwd -P)"

# the abs dir where this script is located (so we can call it from wherever)
script_dir="$(cd "$(dirname "$0")"; pwd -P)"

cd "${script_dir}"
doxygen Doxyfile
sphinx-build . "${script_dir}/../public"
cd -

if [ "${use_venv}" != "false" ]
then
    deactivate
fi
