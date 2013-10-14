#!/usr/bin/env python

from stella import stella
from random import randint
import sys

def addition(a,b):       return a+b
def subtraction(a,b):    return a-b
def multiplication(a,b): return a*b
def division(a,b):       return a/b
def floor_division(a,b): return a//b
def chained(a,b): return (a-b)/b*a;

arglist1 = [(-1,0), (84, -42), (1.0, 1), (0,1), (randint(0, 1000000), randint(0, 1000000)), (-1*randint(0, 1000000), randint(0, 1000000))]
def make_test(f, arglist):
    for i in range(len(arglist)):
        def perform_test(f, args):
            a, b = args
            x = f(a,b)
            y = stella(f)(a,b)
            assert x == y and type(x) == type(y)
        sys.modules[__name__].__dict__["test_" + f.__name__+"_"+str(i)] = lambda: perform_test(f, arglist[i])
for f in [addition, subtraction, multiplication]:
    make_test(f, arglist1)

delta = 0.0000001
arglist2 = [(0,1), (5,2), (5.0,2), (4.0,4), (-5,2), (5.0,-2), (randint(0, 1000000), randint(1, 1000000)), (341433, 673069)]
def make_test(f, arglist):
    for i in range(len(arglist)):
        def perform_test(f, args):
            a, b = args
            x = f(a,b)
            y = stella(f)(a,b)
            assert x-y < delta and type(x) == type(y)
        sys.modules[__name__].__dict__["test_" + f.__name__+"_"+str(i)] = lambda: perform_test(f, arglist[i])
for f in [division, floor_division]:
    make_test(f, arglist2)

#delta = 0.00001
make_test(chained, arglist2)

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
