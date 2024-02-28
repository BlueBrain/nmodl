#!/usr/bin/env bash

# script for generating documentation for NMODL
# note that the NMODL Python wheel must be installed
# for the script to work properly

set -xeu

# in order to create the docs, we first need to build NMODL
build_dir="$(mktemp -d)"
wheel_dir="$(mktemp -d)"
pip wheel . --no-deps --wheel-dir "$(cd "${wheel_dir}"; pwd -P)" -C build-dir="$(cd "${build_dir}"; pwd -P)"

# the abs dir where this script is located (so we can call it from wherever)
script_dir="$(cd "$(dirname "$0")"; pwd -P)"

cd "${script_dir}"
doxygen Doxyfile
sphinx-build . "${script_dir}/../public"
cd -
