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

import subprocess
import os

from setuptools import Command
from skbuild import setup


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

    def run(self, *args, **kwargs):
        """ The scikit-architecture builds python-c++ projects in this way:

        - skbuild.setup reads all the arguments and their options (i.e. build_ext --inplace)
        - decides, based on the arguments (i.e build_ext) if calling cmake is appropriate
        - copies files around if necessary (i.e. build_ext --inplace)
        - calls the arguments as usual

        The problem with calling commands inside a custom one (like here) is that this is
        happens after setup(). In other words, setup reads the arguments (i.e. docs), does nothing (it is not
        aware that there will be a build_ext at this point) and then calls the custom command that calls only
        build_ext for example. In order to force cmake we can pass the --force-cmake option
        to setup. However, the inplace is taken from the command build_ext in setup and things
        are copied around still in setup. Thus, we either change build_ext before setup is called
        or we recall setup inside the custom command.

        There is an open issue here to find a solution with scikit-build developers:

        https://github.com/scikit-build/scikit-build/issues/489

        Katta
        """
        subprocess.run(["python3", "setup.py", "-G", "Unix Makefiles", "build_ext", "--inplace", "-j", str(max(1, os.cpu_count() - 4))]) # workaround
        self.run_command("doctest")
        self.run_command("buildhtml")
        if self.upload:
            self._upload()

    def _upload(self):
        pass



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
    cmake_minimum_required_version="3.3.0",
    cmake_args=["-DPYTHON_EXECUTABLE=" + sys.executable],
    cmdclass=lazy_dict(
        test=NMODLTest,
        install_doc=InstallDoc,
        doctest=get_sphinx_command,
        buildhtml=get_sphinx_command,
    ),
    zip_safe=False,
    setup_requires=[
        "jinja2>=2.9.3",
        "jupyter",
        "m2r",
        "mistune<2", # prevents a version conflict with nbconvert
        "nbconvert<6.0", # prevents issues with nbsphinx
        "nbsphinx>=0.3.2",
        "pytest>=3.7.2",
        "sphinx-rtd-theme",
        "sphinx>=2.0",
        "sphinx<3.0", # prevents issue with m2r where m2r uses an old API no more supported with sphinx>=3.0
        ] + install_requirements,
    install_requires=install_requirements,
)
