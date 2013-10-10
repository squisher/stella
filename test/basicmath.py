#!/usr/bin/env python

from stella import stella
from random import randint
import sys

def addition(a,b):       return a+b
def subtraction(a,b):    return a-b
def multiplication(a,b): return a*b
def division(a,b):       return a/b
def floor_division(a,b): return a//b

arglist = [(-1,0), (84, -42), (1.0, 1), (0,1), (randint(0, 1000000), randint(0, 1000000)), (-1*randint(0, 1000000), randint(0, 1000000))]
def make_test(f):
    def perform_test(f):
        for (a,b) in arglist:
            x = f(a,b)
            y = stella(f)(a,b)
            assert x == y and type(x) == type(y)
    sys.modules[__name__].__dict__["test_" + f.__name__] = lambda: perform_test(f)
for f in [addition, subtraction, multiplication]:
    make_test(f)

delta = 0.0000001
arglist = [(0,1), (5,2), (5.0,2), (4.0,4), (-5,2), (5.0,-2), (randint(0, 1000000), randint(1, 1000000))]
def make_test(f):
    def perform_test(f):
        for (a,b) in arglist:
            x = f(a,b)
            y = stella(f)(a,b)
            assert x-y < delta and type(x) == type(y)
    sys.modules[__name__].__dict__["test_" + f.__name__] = lambda: perform_test(f)
for f in [division, floor_division]:
    make_test(f)

if __name__ == '__main__':
    print(stella(addition)(41, 1))
    #print(stella(addition)(43, -1))
    print(stella(subtraction)(44,2))
    print(stella(multiplication)(21,2))
    print(stella(division)(42,10))
    print(stella(floor_division)(85,2.0))
    print(stella(floor_division)(85,-2.0))
