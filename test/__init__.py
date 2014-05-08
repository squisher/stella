from stella import wrap
import sys
from pytest import mark
from pytest import raises

def single_args(l):
    return list(map (lambda x: (x,), l))

def make_eq_test(f, args):
    x = f(*args)
    y = wrap(f)(*args)
    assert x == y and type(x) == type(y)

def make_eq_kw_test(f, args):
    x = f(**args)
    y = wrap(f)(**args)
    assert x == y and type(x) == type(y)

def make_delta_test(f, args, delta = 1e-7):
    x = f(*args)
    y = wrap(f)(*args)
    assert x-y < delta and type(x) == type(y)

def make_exc_test(f, args, py_exc, stella_exc):
    with raises(py_exc):
        x = f(*args)

    with raises(stella_exc):
        y = wrap(f)(*args)

    assert True

