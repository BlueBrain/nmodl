#!/usr/bin/env bash
# A simple set of tests checking if a wheel is working correctly
set -eux

this_dir="$(dirname "$(realpath "$0")")"

if [ "$#" -lt 2 ]; then
    echo "Usage: $(basename "$0") python_exe python_wheel [use_virtual_env]"
    exit 1
fi

# cli parameters
python_exe=$1
python_wheel=$2
use_venv=$3 #if $3 is not "false" then use virtual environment

python_ver=$("$python_exe" -c "import sys; print('%d%d' % tuple(sys.version_info)[:2])")


test_wheel () {
    # sample mod file for nrnivmodl check
    TEST_DIR="$(mktemp -d)"
    OUTPUT_DIR="$(mktemp -d)"
    cp "${this_dir}/../nmodl/ext/example/"*.mod "$TEST_DIR/"
    cp "${this_dir}/../test/integration/mod/cabpump.mod" "${this_dir}/../test/integration/mod/var_init.inc" "$TEST_DIR/"
    for mod in "${TEST_DIR}/"*.mod
    do
        nmodl -o "${OUTPUT_DIR}" "${mod}" sympy --analytic
        $python_exe -c "import nmodl; driver = nmodl.NmodlDriver(); driver.parse_file('${mod}')"
    done
    $python_exe -m pytest -vvv "${this_dir}/../test/"
}

echo "== Testing $python_wheel using $python_exe ($python_ver) =="

# creat python virtual environment and use `python` as binary name
# because it will be correct one from venv.
if [[ "$use_venv" != "false" ]]; then
  echo " == Creating virtual environment == "
  venv_name="$(mktemp -d)/nmodl_test_venv_${python_ver}"
  $python_exe -m venv "$venv_name"
  . "$venv_name/bin/activate"
  python_exe="$(command -v python)"
else
  echo " == Using global install == "
fi

# install nmodl
$python_exe -m pip install -U pip
$python_exe -m pip install "${python_wheel}" pytest
$python_exe -m pip show nmodl || $python_exe -m pip show nmodl-nightly

# run tests
test_wheel "$(command -v python)"

# cleanup
if [[ "$use_venv" != "false" ]]; then
  deactivate
fi
