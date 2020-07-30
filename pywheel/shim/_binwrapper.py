#!/usr/bin/env python
"""
A generic wrapper to access nmodl binaries from a python installation
Please create a softlink with the binary name to be called.
"""

import os
import sys
import stat
from pkg_resources import working_set
from find_libpython import find_libpython


def _config_exe(exe_name):
    """Sets the environment to run the real executable (returned)"""

    package_name = "nmodl"

    assert (
        package_name in working_set.by_key
    ), "NMODL package not found! Verify PYTHONPATH"

    NMODL_PREFIX = os.path.join(working_set.by_key[package_name].location, "nmodl")
    NMODL_HOME = os.path.join(NMODL_PREFIX, ".data")
    NMODL_BIN = os.path.join(NMODL_HOME, "bin")
    NMODL_LIB = os.path.join(NMODL_HOME, "lib")

    # add pywrapper path to environment
    if sys.platform == "darwin":
        os.environ["NMODL_WRAPLIB"] = os.path.join(NMODL_LIB, "libpywrapper.dylib")
    else:
        os.environ["NMODL_WRAPLIB"] = os.path.join(NMODL_LIB, "libpywrapper.so")

    # add libpython*.so path to environment
    os.environ["NMODL_PYLIB"] = find_libpython()

<<<<<<< HEAD
    # add nmodl home to environment (i.e. necessary for nrnunits.lib)
    os.environ["NMODLHOME"] = NMODL_HOME

    return os.path.join(NMODL_BIN, exe_name)

=======
    # TODO remove
    print("NMODL_WRAPLIB")
    print(os.environ["NMODL_WRAPLIB"])
    print("NMODL_PYLIB")
    print(os.environ["NMODL_PYLIB"])

    return os.path.join(NMODL_PREFIX_DATA, exe_name)
>>>>>>> !squash check shim paths and venv nmodl tree

if __name__ == "__main__":
    """Set the pointed file as executable"""
    exe = _config_exe(os.path.basename(sys.argv[0]))
    st = os.stat(exe)
    os.chmod(exe, st.st_mode | stat.S_IEXEC)
    os.execv(exe, sys.argv)
