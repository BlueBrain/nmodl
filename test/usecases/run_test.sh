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

# NRN + nocmodl
rm -r "${output_dir}" tmp || true
nrnivmodl
python simulate.py nocmodl.txt

# NRN + NMODL
rm -r "${output_dir}" tmp || true
nrnivmodl -nmodl "${nmodl}"
python simulate.py nmodl.txt

# if files are generated, compare them, then remove them

if [[ -f 'nmodl.txt' ]] && [[ -f 'nocmodl.txt' ]]
then
    # diff will report a non-zero exit code if they differ
    diff nocmodl.txt nmodl.txt
    rm -fr nocmodl.txt nmodl.txt
fi

popd
