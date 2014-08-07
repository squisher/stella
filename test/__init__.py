import numpy as np
from functools import wraps
import time

from stella import wrap
import pytest
from pytest import mark
from pytest import raises


def single_args(l):
    return list(map(lambda x: (x,), l))


def make_eq_test(f, args):
    args1 = []
    args2 = []
    for a in args:
        if type(a) == np.ndarray:
            args1.append(np.copy(a))
            args2.append(np.copy(a))
        else:
            args1.append(a)
            args2.append(a)
    x = f(*args1)
    y = wrap(f)(*args2)
    assert x == y and type(x) == type(y)


def make_eq_kw_test(f, args):
    x = f(**args)
    y = wrap(f)(**args)
    assert x == y and type(x) == type(y)


def make_delta_test(f, args, delta=1e-7):
    x = f(*args)
    y = wrap(f)(*args)
    assert x - y < delta and type(x) == type(y)


def make_exc_test(f, args, py_exc, stella_exc):
    with raises(py_exc):
        x = f(*args)  # noqa

    with raises(stella_exc):
        y = wrap(f)(*args)  # noqa

    assert True


unimplemented = mark.xfail(reason="Unimplemented", run=False)
bench = mark.bench


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kw_args):
        start = time.time()
        r = f(*args, **kw_args)
        end = time.time()
        print("{0}({1}, {2}) took {3:0.2f}s".format(
            f.__name__, args, kw_args, end - start))
        return r
    return wrapper


@pytest.fixture
def report():
    pass
