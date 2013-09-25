#!/usr/bin/env python

from stella import stella
from random import randint

def addition(a,b):
    return a+b

def test_addition():
    for (a,b) in [(0,0), (randint(0, 1000000), randint(0, 1000000))]:
        assert addition(a,b) == stella(addition)(a,b)
