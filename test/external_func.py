import numpy as np
from random import randint

import mtpy

from test import *
from stella import zeros
from .basicmath import addition, subtraction
from . import basicmath


def seed_const():
    mtpy.mt_seed32new(42)

def seed(s):
    mtpy.mt_seed32new(s)

def drand_const():
    mtpy.mt_seed32new(42)
    return mtpy.mt_drand()

def drand(s):
    mtpy.mt_seed32new(s)
    return mtpy.mt_drand() + mtpy.mt_drand()

@mark.parametrize('f', [seed_const, drand_const])
def test1(f):
    make_eq_test(f, ())

@mark.parametrize('arg', single_args([1,2,42,1823828, randint(1, 10000000), randint(1, 10000000)]))
@mark.parametrize('f', [seed,drand])
def test2(f, arg):
    make_eq_test(f, arg)
