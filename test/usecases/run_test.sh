#! /usr/bin/env bash
set -eu

if [[ $# -ne 2 ]]
then
  echo "Usage: $0 NMODL USECASE_DIR"
fi

nmodl="$1"
output_dir="$(uname -m)"
usecase_dir="$2"

pushd "${usecase_dir}"

for file in ./simulate*.py
do

    # NRN + nocmodl
    rm -r "${output_dir}" tmp || true
    nrnivmodl
    "$(uname -m)/special" -nogui "${file}"

    # NRN + NMODL
    rm -r "${output_dir}" tmp || true
    nrnivmodl -nmodl "${nmodl}"
    "$(uname -m)/special" -nogui "${file}"
done

popd
