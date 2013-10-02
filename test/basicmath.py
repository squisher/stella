#!/usr/bin/env python

from stella import stella
from random import randint

def addition(a,b):
    return a+b

def subtraction(a,b):
    return a-b

def test_addition():
    for (a,b) in [(0,0), (randint(0, 1000000), randint(0, 1000000))]:
        assert addition(a,b) == stella(addition)(a,b)

def test_subtraction():
    for (a,b) in [(0,0), (randint(0, 1000000), randint(0, 1000000))]:
        assert subtraction(a,b) == stella(subtraction)(a,b)

if __name__ == '__main__':
    print(stella(addition)(41, 1))
    print(stella(addition)(43, -1))
    print(stella(subtraction)(44,2))
