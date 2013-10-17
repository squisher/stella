from stella import stella
import sys
from pytest import mark

def single_args(l):
    return list(map (lambda x: (x,), l))

def make_eq_test(f, args):
    x = f(*args)
    y = stella(f)(*args)
    assert x == y and type(x) == type(y)

def make_delta_test(f, args, delta = 0.0000001):
    x = f(*args)
    y = stella(f)(*args)
    assert x-y < delta and type(x) == type(y)
