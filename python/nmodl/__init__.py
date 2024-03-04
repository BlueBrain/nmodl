import os
import sys

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files

from find_libpython import find_libpython

# try to add libpython*.so path to environment if not already set
try:
    os.environ["NMODL_PYLIB"] = os.environ.get(
        "NMODL_PYLIB",
        find_libpython(),
    )
except TypeError as exc:
    raise RuntimeError(
        "find_libpython was unable to find the Python library on this platform; "
        "please make sure that the Python library is installed correctly\n"
        "You can also try to manually set the NMODL_PYLIB environmental variable "
        "to the Python library"
    ) from exc

# add nmodl home to environment (i.e. necessary for nrnunits.lib) if not
# already set
# `files` will automatically raise a `ModuleNotFoundError`
os.environ["NMODLHOME"] = os.environ.get(
    "NMODLHOME",
    str(files("nmodl") / ".data"),
)


try:
    # Try importing but catch exception in case bindings are not available
    from ._nmodl import NmodlDriver, to_json, to_nmodl, __version__  # noqa

    __all__ = ["NmodlDriver", "to_json", "to_nmodl"]
except ImportError:
    print("[NMODL] [warning] :: Python bindings are not available")
