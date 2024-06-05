"""
Helper functions
"""
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def compile_modfile(
    modfile_path: Path,
    compiler_path: Optional[Path] = None,
):
    """
    Compiler a given mod file (or directory with mod files).
    """
    compiler_args = [str(shutil.which("nrnivmodl"))]
    if compiler_path:
        compiler_args += ["-nmodl", str(compiler_path)]

    # the dir where we will run nrnivmodl
    dir_to_run_in = path.parent if (path := Path(modfile_path)).is_file() else path

    subprocess.run(
        compiler_args + [modfile_path],
        check=True,
        cwd=dir_to_run_in,
    )
