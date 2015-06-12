Stella: A Low-Level Language Embedded into Python to Easily Structure Fast Simulations
======================================================================================

Stella has a relatively narrow focus: Python programs with a medium size core
that needs to be executed very efficiently. Simulations are a prime example:
In particular stochastic simulations need to be sampled many times to get
relevant results which are statistically relevant, but even regular
simulations often are extremely slow when run in the regular Python
interpreter.

The key features of Stella are:
* compiled with LLVM for execution times similar to C programs
* statically typed; ensured correctness for the computationally expensive
  parts
* no interaction with the Python run-time to avoid any unintended performance
  penalties
* completely valid Python code for verification and debugging purposes
* easy integration with C libraries with the use of Cython
* excellent functional test coverage
* compatible with numpy arrays

**Note:**
Stella is currently in an alpha stage: it is a research project, albeit a well
tested one. But there is no guarantee that the program you want to run will be
completely supported. Please do give it a try, and report back!


Installation
------------

Stella requires numpy and llvmlite (which in turn requires LLVM to be
installed). I recommend setting up a *virtualenv*.

```shell
git clone https://github.com/squisher/stella.git/
cd stella

# grab branch of llvmlite known to work with stella
git clone https://github.com/squisher/llvmlite.git/
cd llvmlite
git checkout stella
# At least Debian does not create a llvm-config symlink, so explicitly tell
# llvmlite which binary to use. Debian jessie has `llvm-3.5', while wheezy
# only ships with llvm-3.0, which is rather old. You may want to upgrade to
# jessie or install LLVM by hand.
export LLVM_CONFIG=llvm-config-3.5
pip install .
cd ..

# install stella in editable mode
pip install -e .
```

If you intend to run the unit tests, skip the last line and instead run:

```shell
cd mtpy
# make sure cython is installed
python setup.py install
cd ..

pip install -e '.[test]'
```


Basic Usage
-----------

You can execute any Python function in Stella, given that it only uses
supported features. Stella will automatically consider the call graph, and
ensure that until the entry method returns, everything is executed in Stella.
Otherwise an exception will be raised.

```python
def some_math(x, y):
    return sqrt(x**2 + y**2 - 42)

import stella
result = stella.wrap(some_math)(13, 7)
```

Or a slightly more complex example:

```python
# Just an example, for illustration only
def compute(x):
    return (x+2)/5

def stuff(a, b):
    for i in range(len(a)):
        b[i] = compute(a[i])

import numpy, stella
a = numpy.array(range(9))
b = numpy.zeros(a.shape, dtype=a.dtype)

stuff(a, b)

print (b)
```


Integrating C Libraries
-----------------------

Stella expects all Python function names to be identical to their C version,
and the module must implement a function **getCSignatures()** which returns a
dictionary mapping the function name to a ctypes-style signature.


mtpy
----
The *mtpy* library is small C library and demonstrates how to support Stella.
It is an implementation of the Mersenne Twist random number generator by Geoff
Kuenning. It is licensed under the LGPL. See the included README for more
details.

Note that *mtpy* is used by some of the tests, and serves as an example, but
it is NOT part of Stella.


Tests
-----

Testing is done with the absolutely great *py.test* library. If you also want
to run the benchmarks, then *pystache* is additionally required. Some tests
and benchmarks use the aforementioned *mtpy* library.

```python
import stella
stella.run_tests()
```


Contact and Feedback
--------------------

Please file an issue at github or send me an email at dmohr@cs.unm.edu

Thanks for trying Stella!


License
-------

Copyright 2013-2015 David Mohr

Unless otherwise noted files are under the following license:

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
