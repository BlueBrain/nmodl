#!/usr/bin/env python
"""
A generic wrapper to access nmodl binaries from a python installation
Please create a softlink with the binary name to be called.
"""
import os
from pathlib import Path
import subprocess
import stat
import sys

if sys.version_info >= (3, 9):
    from importlib.metadata import metadata, PackageNotFoundError
    from importlib.resources import files
else:
    from importlib_metadata import metadata, PackageNotFoundError
    from importlib_resources import files

from find_libpython import find_libpython


def main():
    """Sets the environment to run the real executable (returned)"""

    try:
        metadata("nmodl-nightly")
        print("INFO : Using nmodl-nightly Package (Developer Version)")
    except PackageNotFoundError:
        pass

    prefix = files("nmodl")
    exe = prefix / ".data" / "bin" / Path(sys.argv[0]).name

    if os.name == "nt":
        exe = exe.with_suffix(".exe")
    else:
        st = os.stat(exe)
        os.chmod(exe, st.st_mode | stat.S_IEXEC)

    env = dict(os.environ)
    # add libpython*.so path to environment
    env["NMODL_PYLIB"] = find_libpython()
    # set PYTHONPATH for embedded python to properly find the nmodl module
    env["PYTHONPATH"] = str(prefix.parent)
    if pth := os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] += os.pathsep + pth

    cmd = [exe] + sys.argv[1:]
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError:
        sys.exit(1)
