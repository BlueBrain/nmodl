# ***********************************************************************
# Copyright (C) 2018-2019 Blue Brain Project
#
# This file is part of NMODL distributed under the terms of the GNU
# Lesser General Public License. See top-level LICENSE file for details.
# ***********************************************************************

import inspect
import sys

from distutils.cmd import Command
from skbuild import setup
from skbuild.command.build_ext import build_ext


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
        cmd_obj = self.distribution.get_command_obj('build_ext')
        cmd_obj.inplace = 1
        self.run_command('build_ext')
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
