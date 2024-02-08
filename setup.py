# Copyright 2023 Blue Brain Project, EPFL.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0
import sys

from skbuild import setup


cmake_args = ["-DPYTHON_EXECUTABLE=" + sys.executable]
if "bdist_wheel" in sys.argv:
    cmake_args.append("-DLINK_AGAINST_PYTHON=FALSE")
    cmake_args.append("-DNMODL_ENABLE_TESTS=FALSE")

setup(
    packages=["nmodl"],
    include_package_data=True,
    cmake_minimum_required_version="3.15.0",
    cmake_args=cmake_args,
)
