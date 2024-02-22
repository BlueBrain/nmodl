#!/bin/bash
# A simple set of tests checking if a wheel is working correctly
set -xe

python -m pytest "$1/test/unit/pybind"
find "$1/test/" "$1/nmodl/ext/" \
    -name "*.mod" \
    -exec nmodl '{}' sympy --analytic \; \
    -exec python -c "import nmodl; driver = nmodl.NmodlDriver(); driver.parse_file('{}')" \;
