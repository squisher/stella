#!/usr/bin/env python

from stella import stella
from random import randint
from test import *

def return_bool(): return True
def return_arg(x): return x
def equality(a,b): return a==b

make_eq_test(__name__, return_bool, [])
make_eq_test(__name__, return_arg, single_args([True, False, 0, 1, 42.0, -42.5]))
make_eq_test(__name__, equality, [(True, True), (1,1), (42.0, 42.0), (1,2), (2.0, -2.0), (True, False), (randint(0, 10000000), randint(-10000 , 1000000))])

if __name__ == '__main__':
    print(stella(return_bool)())
