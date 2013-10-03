#!/usr/bin/env python

from stella import stella
from random import randint
import sys

def addition(a,b):       return a+b
def subtraction(a,b):    return a-b
def multiplication(a,b): return a*b
def division(a,b):       return a/b

for f in [addition, subtraction, multiplication]:
    def template():
        for (a,b) in [(0,1), (randint(0, 1000000), randint(0, 1000000))]:
            assert f(a,b) == stella(f)(a,b)
    sys.modules[__name__].__dict__["test_" + f.__name__] = template

if __name__ == '__main__':
    print(stella(addition)(41, 1))
    #print(stella(addition)(43, -1))
    print(stella(subtraction)(44,2))
    print(stella(multiplication)(21,2))
