#!/usr/bin/env python

from stella import exc
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
