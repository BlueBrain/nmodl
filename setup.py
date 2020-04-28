# ***********************************************************************
# Copyright (C) 2018-2019 Blue Brain Project
#
# This file is part of NMODL distributed under the terms of the GNU
# Lesser General Public License. See top-level LICENSE file for details.
# ***********************************************************************

import inspect
import os
import os.path as osp
import platform
import re
import subprocess
import sys
import sysconfig
from distutils.version import LooseVersion

from distutils.dir_util import copy_tree
from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext


class lazy_dict(dict):
    """When the value associated to a key is a function, then returns
    the function call instead of the function.
    """

    def __getitem__(self, item):
        value = dict.__getitem__(self, item)
        if inspect.isfunction(value):
            return value()
        return value


def get_sphinx_command():
    """Lazy load of Sphinx distutils command class
    """
    from sphinx.setup_command import BuildDoc

    return BuildDoc

class Docs(Command):
    description = "Generate & optionally upload documentation to docs server"
    user_options = [("upload", None, "Upload to docs server")]
    finalize_options = lambda self: None

    def initialize_options(self):
        self.upload = False

    def run(self):
        # The extensions must be created inplace to inspect docs
        self.reinitialize_command('build_ext', inplace=1)
        self.run_command('build_ext')
        self.run_command("doctest")
        self.run_command("buildhtml")
        if self.upload:
            self._upload()

    def _upload(self):
        pass


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = osp.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = LooseVersion(
            re.search(r"version\s*([\d.]+)", out.decode()).group(1)
        )
        if cmake_version < "3.3.0":
            raise RuntimeError("CMake >= 3.3.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def get_egg_paths(self):
        eggs_basepath = osp.join(osp.dirname(osp.abspath(__file__)), '.eggs')
        eggs = [osp.join(eggs_basepath, egg) for egg in os.listdir(eggs_basepath)]
        return eggs

    def build_extension(self, ext):
        extdir = osp.abspath(osp.dirname(self.get_ext_fullpath(ext.name)))
        extdir = osp.join(extdir, ext.name)
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j{}".format(max(1, os.cpu_count() - 3))]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        env["PYTHONPATH"] = '{}:{}'.format(
                ':'.join(self.get_egg_paths()),
                env.get("PYTHONPATH",""))
        if not osp.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp, env=env
        )

        # copy nmodl module with shared library to extension directory
        copy_tree(os.path.join(self.build_temp, 'nmodl'), extdir)


install_requirements = [
        "PyYAML>=3.13",
        "sympy>=1.3",
        ]

setup(
    name="NMODL",
    version="0.2",
    author="Blue Brain Project",
    author_email="bbp-ou-hpc@groupes.epfl.ch",
    description="NEURON Modelling Language Source-to-Source Compiler Framework",
    long_description="",
    packages=["nmodl"],
    include_package_data=True,
    ext_modules=[CMakeExtension("nmodl")],
    cmdclass=lazy_dict(
        build_ext=CMakeBuild,
        docs=Docs,
        doctest=get_sphinx_command,
        buildhtml=get_sphinx_command,
    ),
    zip_safe=False,
    setup_requires=[
        "jinja2>=2.9.3",
        "jupyter-client",
        "m2r",
        "mistune<2", # prevents a version conflict with nbconvert
        "nbconvert<6.0", # prevents issues with nbsphinx
        "nbsphinx>=0.3.2",
        "pytest>=3.7.2",
        "sphinx-rtd-theme",
        "sphinx>=2.0",
        "sphinx<3.0",
        ] + install_requirements,
    install_requires=install_requirements,
)
