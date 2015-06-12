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

from random import randint
import numpy as np
from . import *  # noqa


def return_bool():
    return True


def return_arg(x):
    return x


def numpy_return_element(a):
    return a[2]


def equality(a, b):
    return a == b


def cast_float(x):
    return float(x)


def cast_int(x):
    return int(x)


def cast_bool(x):
    return bool(x)


def test1():
    make_eq_test(return_bool, ())


@mark.parametrize('arg', single_args([True, False, 0, 1, 42.0, -42.5]))
def test2(arg):
    make_eq_test(return_arg, arg)


@mark.parametrize('args', [(True, True), (1, 1), (42.0, 42.0), (1, 2), (2.0, -2.0), (True, False),
                           (randint(0, 10000000), randint(-10000, 1000000))])
def test3(args):
    make_eq_test(equality, args)


@mark.parametrize('args', [(False, 1), (False, 0), (True, 1), (42.0, True), (1, 1.0),
                           (randint(0, 10000000), float(randint(-10000, 1000000)))])
@mark.xfail()
def test3fail(args):
    make_eq_test(equality, args)


@mark.parametrize('args', single_args([np.zeros(5, dtype=int),
                                       np.array([1, 2, 3, 4, 5], dtype=int)]))
@mark.parametrize('f', [numpy_return_element])
@mark.xfail()
def test4fail(f, args):
    make_eq_test(f, args)


@mark.parametrize('f', [cast_float, cast_int])
@mark.parametrize('args', single_args([1, 42, -3, -5.5, 0, 3.14, randint(0, 10000000)]))
def test5(f, args):
    make_eq_test(f, args)


@mark.parametrize('f', [cast_bool])
@mark.parametrize('args', single_args([1, 42, -3, -5.5, 0, 3.14, randint(0, 10000000)]))
@unimplemented
def test5u(f, args):
    make_eq_test(f, args)
