#!/usr/bin/env python

from stella import stella
from random import randint
from test import *

def return_bool(): return True
def return_arg(x): return x
def equality(a,b): return a==b

def test1():
    make_eq_test(return_bool, ())

@mark.parametrize('arg', single_args([True, False, 0, 1, 42.0, -42.5]))
def test2(arg):
    make_eq_test(return_arg, arg)

@mark.parametrize('args', [(True, True), (1,1), (42.0, 42.0), (1,2), (2.0, -2.0), (True, False), (randint(0, 10000000), randint(-10000 , 1000000))])
def test3(args):
    make_eq_test(equality, args)

@mark.parametrize('args', [(False, 1), (42.0, True), (1, 1.0), (randint(0, 10000000), float(randint(-10000 , 1000000)))])
@mark.xfail()
def test3fail(args):
    make_eq_test(equality, args)

if __name__ == '__main__':
    print(stella(return_bool)())
