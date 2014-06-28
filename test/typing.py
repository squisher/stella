#!/usr/bin/env python

from random import randint
import numpy as np
from test import *  # noqa


def return_bool():
    return True


def return_arg(x):
    return x


def numpy_return_element(a):
    return a[2]


def equality(a, b):
    return a == b


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
@mark.parameterize('f', numpy_return_element)
@mark.xfail()
def test4fail(f, args):
    make_eq_test(f, args)
