#! /usr/bin/env bash
set -e

nmodl="$1"
usecase_dir="$2"

pushd "${usecase_dir}"

rm -r x86_64 tmp || true

nrnivmodl -nmodl "${nmodl}"
x86_64/special simulate.py

popd
