#! /usr/bin/env bash
set -eu

if [[ $# -ne 3 ]]
then
  echo "Usage: $0 NMODL USECASE_DIR OUTPUT_DIR"
  exit -1
fi

nmodl="$1"
usecase_dir="$2"
output_dir="$3"

"${nmodl}" "${usecase_dir}"/*.mod --neuron -o "${output_dir}"/neuron
"${nmodl}" "${usecase_dir}"/*.mod -o "${output_dir}"/coreneuron
