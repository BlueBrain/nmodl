import os
import sys

if sys.version_info >= (3, 9):
    from importlib.resources import files
    from importlib.metadata import version, PackageNotFoundError
else:
    from importlib_resources import files
    from importlib_metadata import version, PackageNotFoundError

from find_libpython import find_libpython

# add libpython*.so path to environment
os.environ["NMODL_PYLIB"] = find_libpython()

# add nmodl home to environment (i.e. necessary for nrnunits.lib)
os.environ["NMODLHOME"] = str(files("nmodl") / ".data")


try:
    # Try importing but catch exception in case bindings are not available
    from ._nmodl import NmodlDriver, to_json, to_nmodl  # noqa

    try:
        __version__ = version("nmodl")
    except PackageNotFoundError:
        # package is not installed
        pass

    __all__ = ["NmodlDriver", "to_json", "to_nmodl"]
except ImportError:
    print("[NMODL] [warning] :: Python bindings are not available")
