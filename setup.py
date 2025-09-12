# Copyright (c) PEMD development team.
# Distributed under the terms of the MIT License.

"""Setup.py for PEMD."""

import codecs
import os
from setuptools import setup, find_packages

# these things are needed for the README.md show on pypi (if you dont need delete it)
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.0.0'
DESCRIPTION = 'a python package for molecular dynamics simulations of polymers electrolytes'
INSTALL_REQUIRES = [
    "rdkit == 2024.3.6",
]
setup(
    name="PEMD",
    version=VERSION,
    author="mdgo development team",
    author_email="jcy23@mails.tsinghua.edu.cn",
    description=DESCRIPTION,
    license="MIT",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    keywords=[
            "Gromacs",
            "Molecular dynamics",
            "polymer",
            "quantumn calculations",
            "charge",
            "materials",
            "science",
            "solvation",
            "diffusion",
            "transport",
            "conductivity",
            "force field"
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Unix",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
)
