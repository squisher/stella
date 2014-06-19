#!/usr/bin/env python
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "Marsenne Twist for Python",
    ext_modules = cythonize('mtpy.pyx'),
)
