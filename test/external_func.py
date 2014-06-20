from test import *
from stella import zeros
from .basicmath import addition, subtraction
from . import basicmath
import numpy as np

import mtpy

def seed():
    mtpy.mt_seed32new(42)

def drand():
    mtpy.mt_seed32new(42)
    return mtpy.drand()

#@mark.parametrize('args', [(40,2), (43, -1), (41, 1)])
@mark.parametrize('f', [])
def test1(f,args):
    make_eq_test(f, ())

@mark.parametrize('f', [seed,drand])
@unimplemented
def test1b(f):
    make_eq_test(f, ())
