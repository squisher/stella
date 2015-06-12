#!/usr/bin/env python
# Copyright 2013-2015 David Mohr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from setuptools import setup


def README():
    return open(os.path.join(os.path.dirname(__file__), 'README.md')).read()


classifiers = ['Development Status :: 3 - Alpha',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: Apache 2.0 License',
               'Operating System :: POSIX',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: MacOS :: MacOS X',
               'Topic :: Software Development :: Libraries',
               'Topic :: Scientific/Engineering'] + [
              ('Programming Language :: Python :: %s' % x) for x in
                  '3.2 3.4'.split()]


setup(
    name="stella",
    version="0.2",
    packages=['stella'],
    author='David Mohr',
    author_email='dmohr@cs.unm.edu',
    long_description=README(),
    classifiers=classifiers,
    install_requires=['llvmlite', 'numpy'],
    extras_require={
        'test': ['pytest', 'pystache', 'mtpy']
    },
)
