#!/usr/bin/env python

from stella import exc, wrap
from test import *  # noqa
from stella.intrinsics.python import zeros


def undefined1():
    if False:
        r = 1
    return r


def undefined2():
    if False:
        x = 1
    y = 0 + x  # noqa
    return True


def zeros_no_type():
    a = zeros(5)  # noqa


@mark.parametrize('f', [undefined1, undefined2])
def test_undefined(f):
    make_exc_test(f, (), UnboundLocalError, exc.UndefinedError)


def third(t):
    return t[2]


def callThird():
    t = (4, 2)
    return third(t)


def array_alloc_const_index_out_of_bounds():
    a = zeros(5, dtype=int)
    a[5] = 42


def array_alloc_var_index_out_of_bounds():
    """This tests causes a segmentation fault."""
    a = zeros(5, dtype=int)
    i = 5
    a[i] = 42


@mark.parametrize('f', [callThird, array_alloc_const_index_out_of_bounds])
def test_indexerror(f):
    make_exc_test(f, (), IndexError, exc.IndexError)


@mark.parametrize('f', [array_alloc_var_index_out_of_bounds])
@unimplemented
def test_indexerror_segfault(f):
    """Would crash"""
    make_exc_test(f, (), IndexError, exc.IndexError)


class TestException(Exception):
    pass


def raise_exc():
    raise TestException('foo')


@mark.parametrize('f', [raise_exc])
def test_exception(f):
    """
    Note: this isn't a real test. The NotImplementedError is thrown during
    _compile-time_, not _run-time_!
    """
    with raises(Exception):
        f()

    with raises(NotImplementedError):
        wrap(f)()
