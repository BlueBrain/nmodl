#! /usr/bin/env bash
set -eu

function run_tests() {
  for f in test_*.py simulate.py
  do
    if [[ -f "$f" ]]
    then
      python "$f"
    fi
  done
}

if [[ $# -ne 2 ]]
then
  echo "Usage: $0 NMODL USECASE_DIR"
  exit -1
fi

nmodl="$1"
output_dir="$(uname -m)"
usecase_dir="$2"

pushd "${usecase_dir}"

# NRN + nocmodl
echo "-- Running NRN+nocmodl ------"
rm -r "${output_dir}" tmp || true
nrnivmodl
run_tests


# NRN + NMODL
echo "-- Running NRN+NMODL --------"
rm -r "${output_dir}" tmp || true
nrnivmodl -nmodl "${nmodl}"
run_tests

popd
