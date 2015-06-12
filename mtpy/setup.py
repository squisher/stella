#!/usr/bin/env python
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="mtpy",
    version="0.2",
    long_description="Marsenne Twist for Python",
    ext_modules=cythonize('mtpy.pyx'),
)
