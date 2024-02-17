#!/bin/bash
# A simple set of tests checking if a wheel is working correctly
set -xe


if [ -z "${VIRTUAL_ENV}" ]
then
    echo "ERROR: you do not appear to be in a virtual environment, please use one before running tests"
    exit 1
fi

test_wheel () {
    # the path to the root directory
    rootdir="$1"
    # sample mod file for nrnivmodl check
    test_dir="$(mktemp -d)"
    cp "${rootdir}"/nmodl/ext/example/*.mod "${test_dir}"
    cp "${rootdir}/test/integration/mod/cabpump.mod" "${rootdir}/test/integration/mod/var_init.inc" "${test_dir}"
    for mod in "${test_dir}"/*.mod
    do
        nmodl "$mod" sympy --analytic
    done
    python -c "import nmodl; driver = nmodl.NmodlDriver(); driver.parse_file('${test_dir}/hh.mod')"
}

# run tests
test_wheel "$@"
