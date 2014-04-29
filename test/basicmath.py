#!/usr/bin/env python

from random import randint
import sys
from test import *

def addition(a,b):       return a+b
def subtraction(a,b):    return a-b
def multiplication(a,b): return a*b
def division(a,b):       return a/b
def floor_division(a,b): return a//b
def modulo(a,b):         return a%b
def power(a,b):          return a**b
def chained(a,b):        return (a-b)/b*a
def inplace(a,b):
    x = a
    x += b
    x /= b
    x -= b
    x *= b
    return x

arglist1 = [(-1,0), (84, -42), (1.0, 1), (0,1), (randint(0, 1000000), randint(0, 1000000)), (-1*randint(0, 1000000), randint(0, 1000000))]
@mark.parametrize('args', arglist1)
@mark.parametrize('f', [addition, subtraction, multiplication])
def test1(f,args):
    make_eq_test(f, args)

arglist2 = [(0,1), (5,2), (5.2,2), (4.0,4), (-5,2), (5.0,-2), (3,1.5), (randint(0, 1000000), randint(1, 1000000)), (341433, 673069)]
@mark.parametrize('args', arglist2)
@mark.parametrize('f', [division, floor_division])
def test2(f, args):
    make_delta_test(f, args)

@mark.parametrize('args', arglist2)
@mark.parametrize('f', [chained, inplace])
def test_accuracy(f, args):
    """Note: Lower accuracy"""
    make_delta_test(f, args, delta=1e-6)

@mark.parametrize('args', filter(lambda e: e[0] >= 0, arglist2))
def test_modulo(args):
    """Note: Lower accuracy"""
    make_delta_test(modulo, args, delta=1e-6)

@mark.parametrize('args', filter(lambda e: e[0] < 0, arglist2))
@mark.xfail()
def test_semantics_modulo(args):
    """Semantic difference:
    Modulo always has the sign of the divisor in Python, unlike C where it is
    the sign of the dividend.
    """
    make_delta_test(modulo, args)

arglist3 = [(0,42), (42,0), (2,5.0), (2.0,5), (1.2,2), (4,7.5), (-4,2)]
@mark.parametrize('args', arglist3)
@mark.parametrize('f', [power])
def test3(f,args):
    make_delta_test(f, args)

@mark.parametrize('args', [(4,-2)])
@mark.xfail()
def test_semantics_power(args):
    """Semantic difference:
    4**2 returns an integer, but 4**-2 returns a float.
    """
    make_delta_test(power, args)
