#!/usr/bin/env bash

# script for generating documentation for NMODL
# note that the NMODL Python wheel must be installed
# for the script to work properly

set -xe

# the abs dir where this script is located (so we can call it from wherever)
script_dir="$(realpath "$(dirname "$0")")"

cd "${script_dir}"
doxygen Doxyfile
sphinx-build . "${script_dir}/../public"
cd -
