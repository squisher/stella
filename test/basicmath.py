#!/usr/bin/env python

from stella import stella
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
@mark.parametrize('f', [addition, subtraction, multiplication])
@mark.parametrize('args', arglist1)
def test1(f,args):
    make_eq_test(f, args)


arglist2 = [(0,1), (5,2), (5.2,2), (4.0,4), (-5,2), (5.0,-2), (3,1.5), (randint(0, 1000000), randint(1, 1000000)), (341433, 673069)]
@mark.parametrize('f', [division, floor_division, modulo, chained, inplace])
@mark.parametrize('args', arglist2)
def test2(f, args):
    make_delta_test(f, args)

arglist3 = [(0,42), (42,0), (2,5.0), (2.0,5), (1.2,2), (4,7.5), (4,-2), (-4,2)]
@mark.parametrize('f', [power])
@mark.parametrize('args', arglist3)
def test3(f,args):
    make_delta_test(f, args)

if __name__ == '__main__':
    print(stella(addition)(41, 1))
    print(stella(addition)(43, -1))
    print(stella(subtraction)(44,2))
    print(stella(multiplication)(21,2))
    print(stella(division)(42,10))
    print(stella(floor_division)(85,2.0))
    print(stella(floor_division)(85,-2.0))
    x = chained(5,2)
    y = stella(chained)(5,2)
    print("chained difference: {0:.10f}".format(x-y))
