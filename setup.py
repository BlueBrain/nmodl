# ***********************************************************************
# Copyright (C) 2018-2019 Blue Brain Project
#
# This file is part of NMODL distributed under the terms of the GNU
# Lesser General Public License. See top-level LICENSE file for details.
# ***********************************************************************

import inspect
import os
import subprocess
import sys
import sysconfig

from distutils.cmd import Command
from skbuild import setup
from setuptools.command.test import test


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


class InstallDoc(Command):
    description = 'Install Sphinx documentation'
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass

    def run(self):
        self.run_command("test")
        self.run_command("doctest")
        self.run_command("buildhtml")

class NMODLTest(test):
    """Custom disutils command that acts like as a replacement
    for the "test" command.

    It first executes the standard "test" command, then runs the
    C++ tests and finally runs the "doctest" to also validate
    code snippets in the sphinx documentation.
    """

    def distutils_dir_name(self, dname):
        """Returns the name of a distutils build directory"""
        dir_name = "{dirname}.{platform}-{version[0]}.{version[1]}"
        return dir_name.format(
            dirname=dname, platform=sysconfig.get_platform(), version=sys.version_info
        )

    def run(self):
        super().run()
        subprocess.check_call(
            [
                "cmake",
                "--build",
                os.path.join("build", self.distutils_dir_name("temp")),
                "--target",
                "test",
            ]
        )


install_requirements = ["jinja2>=2.9.3", "PyYAML>=3.13", "sympy>=1.3"]

setup(
    name="NMODL",
    version="0.2",
    author="Blue Brain Project",
    author_email="bbp-ou-hpc@groupes.epfl.ch",
    description="NEURON Modelling Language Source-to-Source Compiler Framework",
    long_description="",
    packages=["nmodl"],
    include_package_data=True,
    cmake_minimum_required_version="3.3.0",
    cmake_args=["-DPYTHON_EXECUTABLE=" + sys.executable],
    cmdclass=lazy_dict(
        test=NMODLTest,
        install_doc=InstallDoc,
        doctest=get_sphinx_command,
        buildhtml=get_sphinx_command,
    ),
    zip_safe=False,
    setup_requires=["nbsphinx>=0.3.2", "mistune<2.0", "m2r", "sphinx-rtd-theme", "sphinx>=2.0", "sphinx<3.0"]
    + install_requirements,
    install_requires=install_requirements,
    tests_require=["pytest>=3.7.2"],
)
