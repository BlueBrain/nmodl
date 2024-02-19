#!/bin/bash
# A simple set of tests checking if the NMODL CLI (called from Python) is
# working correctly
set -xe

find "$1/test/" "$1/nmodl/ext/" \
    -name "*.mod" \
    -exec nmodl '{}' sympy --analytic \;
