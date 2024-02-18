#!/bin/bash
# A simple set of tests checking if a wheel is working correctly
set -xe

find "$1/test/" "$1/nmodl/ext/" \
    -name "*.mod" \
    -exec python -c "import nmodl; driver = nmodl.NmodlDriver(); driver.parse_file('{}')" \;
